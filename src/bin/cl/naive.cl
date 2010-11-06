#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void Intersection (
    const __global float* vertex, const __global float* dir, const __global float* o, const __global float* bounds, const __global float* uvs,
    __global float* tHit, __global float* tutv, __global float* dp,
    __global int* index, int count, int size, int offset   ) {
    // find position in global arrays
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= size) return;

    // process all geometry
    float4 e1, e2, s1, s2, d;
    float divisor, invDivisor, b1, b2, t;

    float4 v1, v2, v3, rayd, rayo;
        v1 = (float4)(vertex[9*iGID], vertex[9*iGID+1], vertex[9*iGID+2], 0);
        v2 = (float4)(vertex[9*iGID + 3], vertex[9*iGID + 4], vertex[9*iGID + 5], 0);
        v3 = (float4)(vertex[9*iGID + 6], vertex[9*iGID + 7], vertex[9*iGID + 8], 0);
        e1 = v2 - v1;
        e2 = v3 - v1;


    for ( int i = 0; i < count; i++) {
    rayd = (float4)(dir[3*i], dir[3*i+1], dir[3*i+2], 0);
    rayo = (float4)(o[3*i], o[3*i+1], o[3*i+2], 0);

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
        if (t < bounds[i*2]) continue;

        if (tHit[i] != INFINITY && tHit[i] != NAN && t > tHit[i]) continue;

        tHit[i] = t;
        index[i] = iGID + offset;

        float du1 = uvs[6*iGID]   - uvs[6*iGID+4];
        float du2 = uvs[6*iGID+2] - uvs[6*iGID+4];
        float dv1 = uvs[6*iGID+1] - uvs[6*iGID+5];
        float dv2 = uvs[6*iGID+3] - uvs[6*iGID+5];
        float4 dp1 = v1 - v3;
        float4 dp2 = v2 - v3;
        float determinant = du1 * dv2 - dv1 * du2;
        if ( determinant == 0.f ) {
            float4 temp = normalize(cross(e2, e1));
            if ( fabs(temp.x) > fabs(temp.y)) {
                float invLen = rsqrt(temp.x*temp.x + temp.z*temp.z);
                dp[6*i] = -temp.z*invLen;
                dp[6*i+1] = 0.f;
                dp[6*i+2] = temp.x*invLen;
            } else {
                float invLen = rsqrt(temp.y*temp.y + temp.z*temp.z);
                dp[6*i] = 0.f;
                dp[6*i+1] = temp.z*invLen;
                dp[6*i+2] = -temp.y*invLen;
            }
            float4 help = cross(temp, (float4)(dp[6*i], dp[6*i+1], dp[6*i+2],0));
            dp[6*i+3] = help.x;
            dp[6*i+4] = help.y;
            dp[6*i+5] = help.z;
        } else {
            float invdet = 1.f / determinant;
            float4 help = (dv2 * dp1 - dv1 * dp2) * invdet;
            dp[6*i] = help.x;
            dp[6*i+1] = help.y;
            dp[6*i+2] = help.z;
            help = (-du2 * dp1 + du1 * dp2) * invdet;
            dp[6*i+3] = help.x;
            dp[6*i+4] = help.y;
            dp[6*i+5] = help.z;
        }

        float b0 = 1 - b1 - b2;
        tutv[2*i] = b0*uvs[6*iGID] + b1*uvs[6*iGID+2] + b2*uvs[6*iGID+4];
        tutv[2*i+1] = b0*uvs[6*iGID+1] + b1*uvs[6*iGID+3] + b2*uvs[6*iGID+5];
    }
}

__kernel void IntersectionP (
const __global float* vertex, const __global float* dir, const __global float* o, const __global float* bounds,
__global unsigned char* tHit, int count, int size)
{
    int iGID = get_global_id(0);
    if (iGID >= size) return;

    // process all geometry
    float4 e1, e2, s1, s2, d;
    float divisor, invDivisor, b1, b2, t;

    float4 v1, v2, v3, rayd, rayo;
    v1 = (float4)(vertex[9*iGID], vertex[9*iGID+1], vertex[9*iGID+2], 0);
    v2 = (float4)(vertex[9*iGID + 3], vertex[9*iGID + 4], vertex[9*iGID + 5], 0);
    v3 = (float4)(vertex[9*iGID + 6], vertex[9*iGID + 7], vertex[9*iGID + 8], 0);
    e1 = v2 - v1;
    e2 = v3 - v1;

    for ( int i = 0; i < count; i++){
       if ( tHit[i] == '1') continue; //already know it is occluded
       rayd = (float4)(dir[3*i], dir[3*i+1], dir[3*i+2], 0);
       rayo = (float4)(o[3*i], o[3*i+1], o[3*i+2], 0);
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
        if (t < bounds[2*i] || t > bounds[2*i +1]) continue;
        tHit[i] = '1';
     }
}

