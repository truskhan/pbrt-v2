#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

bool intersectsNode(float4 omin, float4 omax, float2 uvmin, float2 uvmax, float4 bmin, float4 bmax) {
 float4 ocenter = (float4)0;
 float4 ray;
 float2 uv;
 float2 tmin, tmax;

//Minkowski sum of the two boxes (sum the widths/heights and position it at boxB_pos - boxA_pos).
 ray = (omax - omin)/2;
 bmin -= ray;
 bmax += ray;

 ocenter = ray + omin;
 ocenter.w = 0;

 ray = (float4)0;
 ray = normalize((float4)(bmin.x, bmin.y, bmin.z,0) - ocenter);
 tmin.x = tmax.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 tmin.y = tmax.y = ray.z;

 ray = normalize((float4)(bmax.x, bmax.y, bmax.z,0) - ocenter);
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmin.x, bmax.y, bmin.z,0) - ocenter);
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmin.x, bmax.y, bmax.z,0) - ocenter);
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmin.x, bmin.y, bmax.z,0) - ocenter);
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmax.x, bmin.y, bmin.z,0) - ocenter);
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmax.x, bmax.y, bmin.z,0) - ocenter);
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmax.x, bmin.y, bmax.z,0) - ocenter);
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 if ( ( max(tmin.x, uvmin.x) < min(tmax.x, uvmax.x)) && (max(tmin.y, uvmin.y) < min(tmax.y, uvmax.y)))
  return true;

  return false;
}

__kernel void IntersectionP (
const __global float* vertex, const __global float* dir, const __global float* o,
 const __global float* cones, const __global int* pointers, const __global float* bounds,
__global char* tHit, __local int* stack, int size, int height,unsigned int threadsCount, unsigned int chunk
#ifdef STAT_PRAY_TRIANGLE
 ,__global int* stat_rayTriangle
#endif
)
{
    int iGID = get_global_id(0);
    int iLID = get_local_id(0);
    if (iGID >= size) return;

    // process all geometry
    float4 e1, e2;

    float4 v1, v2, v3;
    v1 = vload4(0, vertex + 9*iGID);
    v2 = vload4(0, vertex + 9*iGID + 3);
    v3.x = v2.w;
    v3.y = vertex[9*iGID + 7];
    v3.z = vertex[9*iGID + 8];
    v1.w = 0; v2.w = 0; v3.w = 0;
    e1 = v2 - v1;
    e2 = v3 - v1;

    float4 s1, s2, d, rayd, rayo;
    float divisor, invDivisor, t, b1, b2;

    //calculate bounding box
    float4 bmin, bmax;

    bmin = min(v1,v2);
    bmin = min(bmin, v3);
    bmax = max(v1,v2);
    bmax = max(bmax, v3);


    //find number of elements in top level of the ray hieararchy
    uint levelcount = threadsCount; //end of level0
    uint num = 0;

    for ( int i = 1; i <= height; i++){
        num += levelcount;
        levelcount = (levelcount+1)/2; //number of elements in level
    }

    int SPindex = 0;
    int wbeginStack = (2 + height*(height+1)/2)*iLID;
    uint begin,rindex,temp;
    int i = 0;
    int2 child;

    //3D bounding box of the origin
    float4 omin, omax;
    //2D bounding box for uv
    float2 uvmin, uvmax;

    begin = 11*num;
    for ( int j = 0; j < levelcount; j++){
      // get node description
      omin = vload4(0, cones + begin + 11*j);
      omax = vload4(0, cones + begin + 11*j + 3);
      uvmin = vload2(0, cones + begin + 11*j + 6);
      uvmax = vload2(0, cones + begin + 11*j + 8);
      child = vload2(0, pointers + (begin/11 + j)*2);

      // check if triangle intersects cone
      if ( intersectsNode( omin, omax , uvmin, uvmax, bmin, bmax ))
      {
        //store child to the stack
        stack[wbeginStack + SPindex++] = child.x;
        if ( child.y != -1)
          stack[wbeginStack + SPindex++] = child.y;

        while ( SPindex > 0 ){
          //take the cones from the stack and check them
          --SPindex;
          i = stack[wbeginStack + SPindex];
          omin = vload4(0, cones + i);
          omax = vload4(0, cones + i + 3);
          uvmin = vload2(0, cones + i + 6);
          uvmax = vload2(0, cones + i + 8);
          child = vload2(0, pointers + (i/11)*2);

          if ( intersectsNode( omin, omax , uvmin, uvmax, bmin, bmax ))
          {
            //if the cones is at level 0 - check leaves
            if ( child.x == -2){
              rindex = (i/11)*chunk;
              //rindex = computeRIndex(i,cones, pointers, count);
              // process all rays in the cone
              for ( int k = 0; k < child.y; k++){
                #ifdef STAT_PRAY_TRIANGLE
                 ++stat_rayTriangle[rindex + k];
                #endif

                rayd.x = dir[3*rindex + 3*k];
                rayd.y = dir[3*rindex + 3*k + 1];
                rayd.z = dir[3*rindex + 3*k + 2];

                rayo.x = o[3*rindex + 3*k];
                rayo.y = o[3*rindex + 3*k + 1];
                rayo.z = o[3*rindex + 3*k + 2];
                rayd.w = 0; rayo.w = 0;

                s1 = cross(rayd, e2);
                divisor = dot(s1, e1);
                if ( divisor == 0.0f) continue;
                invDivisor = 1.0f/ divisor;

                // compute first barycentric coordinate
                d = rayo - v1;
                b1 = dot(d, s1) * invDivisor;
                if ( b1 < -1e-3f  || b1 > 1+1e-3f) continue;

                // compute second barycentric coordinate
                s2 = cross(d, e1);
                b2 = dot(rayd, s2) * invDivisor;
                if ( b2 < -1e-3f || (b1 + b2) > 1+1e-3f) continue;

                // Compute _t_ to intersection point
                t = dot(e2, s2) * invDivisor;
                if (t < bounds[2*rindex + k*2] || t > bounds[2*rindex + k*2 +1]) continue;

                tHit[rindex + k] = '1';
              }

            }
            else {
              //save the intersected cone to the stack
              stack[wbeginStack + SPindex++] = child.x;
              if ( child.y != -1)
                stack[wbeginStack + SPindex++] = child.y;
            }
          }
        }
      }

    }

}
