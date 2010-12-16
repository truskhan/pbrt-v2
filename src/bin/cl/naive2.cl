#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.002f

__kernel void Intersection (
    const __global float* vertex, const __global float* dir, const __global float* o, const __global float* bounds, const __global float* uvs,
    __global float* tHit, __global float* tutv, __global float* dp,
    __global int* index, int count, int size, int offset   ) {
    // find position in global arrays
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= count) return;

    // process all geometry
    float4 e1, e2, s1, s2, d;
    float divisor, invDivisor, b1, b2, t;
    float4 v1, v2, v3, rayd, rayo;

    rayd = (float4)(dir[3*iGID], dir[3*iGID+1], dir[3*iGID+2], 0);
    rayo = (float4)(o[3*iGID], o[3*iGID+1], o[3*iGID+2], 0);

    for ( int i = 0; i < size; i++) {
        v1 = (float4)(vertex[9*i], vertex[9*i+1], vertex[9*i+2], 0);
        v2 = (float4)(vertex[9*i + 3], vertex[9*i + 4], vertex[9*i + 5], 0);
        v3 = (float4)(vertex[9*i + 6], vertex[9*i + 7], vertex[9*i + 8], 0);
        e1 = v2 - v1;
        e2 = v3 - v1;

        s1 = cross(rayd, e2);
        divisor = dot(s1, e1);
        if ( divisor == 0) continue;
        invDivisor = 1.0f/ divisor;

        // compute first barycentric coordinate
        d = rayo - v1;
        b1 = dot(d, s1) * invDivisor;
        if ( b1 < 0 || b1 > 1) continue;

        // compute second barycentric coordinate
        s2 = cross(d, e1);
        b2 = dot(rayd, s2) * invDivisor;
        if ( b2 < 0 || (b1 + b2) > 1 ) continue;

        // Compute _t_ to intersection point
        t = dot(e2, s2) * invDivisor;
        if (t < bounds[i*2]) continue;

        if (tHit[iGID] != INFINITY && tHit[iGID] != NAN && t > tHit[iGID]) continue;

        tHit[iGID] = t;
        index[iGID] = i + offset;

        float du1 = uvs[6*i]   - uvs[6*i+4];
        float du2 = uvs[6*i+2] - uvs[6*i+4];
        float dv1 = uvs[6*i+1] - uvs[6*i+5];
        float dv2 = uvs[6*i+3] - uvs[6*i+5];
        float4 dp1 = v1 - v3;
        float4 dp2 = v2 - v3;
        float determinant = du1 * dv2 - dv1 * du2;
        if ( determinant > -EPS && determinant < EPS ) {
            float4 temp = normalize(cross(e2, e1));
            if ( fabs(temp.x) > fabs(temp.y)) {
                float invLen = rsqrt(temp.x*temp.x + temp.z*temp.z);
                dp[6*iGID] = -temp.z*invLen;
                dp[6*iGID+1] = 0.f;
                dp[6*iGID+2] = temp.x*invLen;
            } else {
                float invLen = rsqrt(temp.y*temp.y + temp.z*temp.z);
                dp[6*iGID] = 0.f;
                dp[6*iGID+1] = temp.z*invLen;
                dp[6*iGID+2] = -temp.y*invLen;
            }
            float4 help = cross(temp, (float4)(dp[6*iGID], dp[6*iGID+1], dp[6*iGID+2],0));
            dp[6*iGID+3] = help.x;
            dp[6*iGID+4] = help.y;
            dp[6*iGID+5] = help.z;
        } else {
            float invdet = 1.f / determinant;
            float4 help = (dv2 * dp1 - dv1 * dp2) * invdet;
            dp[6*iGID] = help.x;
            dp[6*iGID+1] = help.y;
            dp[6*iGID+2] = help.z;
            help = (-du2 * dp1 + du1 * dp2) * invdet;
            dp[6*iGID+3] = help.x;
            dp[6*iGID+4] = help.y;
            dp[6*iGID+5] = help.z;
        }

        float b0 = 1 - b1 - b2;
        tutv[2*iGID] = b0*uvs[6*i] + b1*uvs[6*i+2] + b2*uvs[6*i+4];
        tutv[2*iGID+1] = b0*uvs[6*i+1] + b1*uvs[6*i+3] + b2*uvs[6*i+5];
    }
}

__kernel void Intersection2 (
    const __global float* vertex, const __global float* dir, const __global float* o, const __global float* bounds, const __global float* uvs,
    __global float* tHit, __global float* tutv, __global float* dp,
    __global int* index, int count, int size, int offset   ) {
    // find position in global arrays
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= count) return;

    // process all geometry
    float4 e1, e2, s1, s2, d;
    float divisor, invDivisor, b1, b2, t;
    float4 v1, v2, v3, rayd, rayo;

    rayd = (float4)(dir[3*iGID], dir[3*iGID+1], dir[3*iGID+2], 0);
    if ( rayd.x < -5 ) return;
    rayo = (float4)(o[3*iGID], o[3*iGID+1], o[3*iGID+2], 0);

    for ( int i = 0; i < size; i++) {
        v1 = (float4)(vertex[9*i], vertex[9*i+1], vertex[9*i+2], 0);
        v2 = (float4)(vertex[9*i + 3], vertex[9*i + 4], vertex[9*i + 5], 0);
        v3 = (float4)(vertex[9*i + 6], vertex[9*i + 7], vertex[9*i + 8], 0);
        e1 = v2 - v1;
        e2 = v3 - v1;

        s1 = cross(rayd, e2);
        divisor = dot(s1, e1);
        if ( divisor < EPS && divisor > -EPS) continue;
        invDivisor = 1.0f/ divisor;

        // compute first barycentric coordinate
        d = rayo - v1;
        b1 = dot(d, s1) * invDivisor;
        if ( b1 < -1e-3f - EPS || b1 > 1.+1e-3f + EPS) continue;

        // compute second barycentric coordinate
        s2 = cross(d, e1);
        b2 = dot(rayd, s2) * invDivisor;
        if ( b2 < -1e-3f - EPS || (b1 + b2) > 1.+1e-3f + EPS) continue;

        // Compute _t_ to intersection point
        t = dot(e2, s2) * invDivisor;
        if (t < bounds[i*2]) continue;

        if (tHit[iGID] != INFINITY && tHit[iGID] != NAN && t > tHit[iGID]) continue;

        tHit[iGID] = t;
        index[iGID] = i + offset;

        float du1 = uvs[6*i]   - uvs[6*i+4];
        float du2 = uvs[6*i+2] - uvs[6*i+4];
        float dv1 = uvs[6*i+1] - uvs[6*i+5];
        float dv2 = uvs[6*i+3] - uvs[6*i+5];
        float4 dp1 = v1 - v3;
        float4 dp2 = v2 - v3;
        float determinant = du1 * dv2 - dv1 * du2;
        if ( determinant > -EPS && determinant < EPS ) {
            float4 temp = normalize(cross(e2, e1));
            if ( fabs(temp.x) > fabs(temp.y)) {
                float invLen = rsqrt(temp.x*temp.x + temp.z*temp.z);
                dp[6*iGID] = -temp.z*invLen;
                dp[6*iGID+1] = 0.f;
                dp[6*iGID+2] = temp.x*invLen;
            } else {
                float invLen = rsqrt(temp.y*temp.y + temp.z*temp.z);
                dp[6*iGID] = 0.f;
                dp[6*iGID+1] = temp.z*invLen;
                dp[6*iGID+2] = -temp.y*invLen;
            }
            float4 help = cross(temp, (float4)(dp[6*iGID], dp[6*iGID+1], dp[6*iGID+2],0));
            dp[6*iGID+3] = help.x;
            dp[6*iGID+4] = help.y;
            dp[6*iGID+5] = help.z;
        } else {
            float invdet = 1.f / determinant;
            float4 help = (dv2 * dp1 - dv1 * dp2) * invdet;
            dp[6*iGID] = help.x;
            dp[6*iGID+1] = help.y;
            dp[6*iGID+2] = help.z;
            help = (-du2 * dp1 + du1 * dp2) * invdet;
            dp[6*iGID+3] = help.x;
            dp[6*iGID+4] = help.y;
            dp[6*iGID+5] = help.z;
        }

        float b0 = 1 - b1 - b2;
        tutv[2*iGID] = b0*uvs[6*i] + b1*uvs[6*i+2] + b2*uvs[6*i+4];
        tutv[2*iGID+1] = b0*uvs[6*i+1] + b1*uvs[6*i+3] + b2*uvs[6*i+5];
    }
}

__kernel void IntersectionP (
const __global float* vertex, const __global float* dir, const __global float* o, const __global float* bounds,
__global unsigned char* tHit, int count, int size)
{
    int iGID = get_global_id(0);
    if (iGID >= count) return;

    // process all geometry
    float4 e1, e2, s1, s2, d;
    float divisor, invDivisor, b1, b2, t;

    float4 v1, v2, v3, rayd, rayo;
   rayd = (float4)(dir[3*iGID], dir[3*iGID+1], dir[3*iGID+2], 0);
   rayo = (float4)(o[3*iGID], o[3*iGID+1], o[3*iGID+2], 0);

    for ( int i = 0; i < size; i++){
      v1 = (float4)(vertex[9*i], vertex[9*i+1], vertex[9*i+2], 0);
      v2 = (float4)(vertex[9*i + 3], vertex[9*i + 4], vertex[9*i + 5], 0);
      v3 = (float4)(vertex[9*i + 6], vertex[9*i + 7], vertex[9*i + 8], 0);
      e1 = v2 - v1;
      e2 = v3 - v1;

      s1 = cross(rayd, e2);
      divisor = dot(s1, e1);
      if ( divisor == 0.0) continue;
      invDivisor = 1.0f/ divisor;

      // compute first barycentric coordinate
      d = rayo - v1;
      b1 = dot(d, s1) * invDivisor;
      if ( b1 < -1e-3f  || b1 > 1.+1e-3f) continue;

      // compute second barycentric coordinate
      s2 = cross(d, e1);
      b2 = dot(rayd, s2) * invDivisor;
      if ( b2 < -1e-3f || (b1 + b2) > 1.+1e-3f) continue;

      // Compute _t_ to intersection point
      t = dot(e2, s2) * invDivisor;
      if (t < bounds[2*iGID] || t > bounds[2*iGID +1]) continue;
      tHit[iGID] = '1';
      //break; //Causing CL_INVALID_COMMAND_QUEUE

     }
}

