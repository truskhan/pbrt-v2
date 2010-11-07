#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define EPS 0.000002f

void intersectPAllLeaves (const __global float* dir, const __global float* o, const __global float* bounds,
__global char* tHit, float4 v1, float4 v2, float4 v3, float4 e1, float4 e2, int chunk, int rindex
#ifdef STAT_PRAY_TRIANGLE
 ,__global int* stat_rayTriangle
#endif
){
    float4 s1, s2, d, rayd, rayo;
    float divisor, invDivisor, t, b1, b2;
    // process all rays in the cone
    for ( int i = 0; i < chunk; i++){
      #ifdef STAT_PRAY_TRIANGLE
      atom_add(stat_rayTriangle + rindex + i, 1);
       //++stat_rayTriangle[rindex + i];
      #endif

      rayd = vload4(0, dir + 3*rindex + 3*i);
      rayo = vload4(0, o + 3*rindex + 3*i);
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
      if (t < bounds[2*rindex + i*2] || t > bounds[2*rindex + i*2 + 1]) continue;

      tHit[rindex+i] = '1';
    }
}

bool intersectsNode(float4 center, float2 uvmin, float2 uvmax, float4 o, float radius) {
  float2 uv;
  float4 ray = o - center;
  float len = length(ray);
  ray = ray/len;
  uv.x = (ray.x == 0)? 0: atan(ray.y/ray.x);
  uv.y = acos(ray.z);

  len = atan(radius/len);

  if ( max(uv.x - len, uvmin.x) < min(uv.x + len, uvmax.x) &&
    max(uv.y - len, uvmin.y) < min(uv.y + len, uvmax.y))
    return true;

  return false;
}

__kernel void IntersectionP (
const __global float* vertex, const __global float* dir, const __global float* o,
 const __global float* cones, const __global int* pointers, const __global float* bounds,
__global char* tHit, __local int* stack, int size, int height,unsigned int threadsCount, unsigned int chunk
#ifdef STAT_PRAY_TRIANGLE
 , __global int* stat_rayTriangle
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
    v3 = vload4(0, vertex + 9*iGID + 6);
    v1.w = 0; v2.w = 0; v3.w = 0;
    e1 = v2 - v1;
    e2 = v3 - v1;

    float4 center; float radius;
    center = (v1+v2+v3)/3;
    radius = length(v1-center);

    float4 a,x;
    float fi;
    float len;

    //find number of elements in top level of the ray hieararchy
    uint levelcount = threadsCount; //end of level0
    uint num = 0;
    uint lastlevelnum = 0;

    for ( int i = 1; i <= height; i++){
        lastlevelnum = levelcount;
        num += levelcount;
        levelcount = (levelcount+1)/2; //number of elements in level
    }

    int SPindex = 0;
    int wbeginStack = (2 + height*(height+1)/2)*iLID;
    uint begin,rindex;
    int i = 0;
    int2 child;

    float4 center1;
    float radius1;
    float2 u, v;

    begin = 9*num;
    for ( int j = 0; j < levelcount; j++){
      // get node description
      center1 = vload4(0, cones + begin + 9*j);
      //extend the triangle bounding sphere radius
      radius1 = center1.w + radius;
      center1.w = 0;
      u = vload2(0, cones + begin + 9*j + 4);
      v = vload2(0, cones + begin + 9*j + 6);
      child = vload2(0, pointers + (begin/9 + j)*2);

      // check if triangle intersects cone
      if ( intersectsNode(center1, u, v, center, radius1 ))
      {
        //store child to the stack
        stack[wbeginStack + SPindex++] = child.x;
        if ( child.y != -1)
          stack[wbeginStack + SPindex++] = child.y;

        while ( SPindex > 0 ){
          //take the cones from the stack and check them
          --SPindex;
          i = stack[wbeginStack + SPindex];
          center1 = vload4(0, cones + i);
          radius1 = center1.w + radius;
          center1.w = 0;
          u = vload2(0, cones + i + 4);
          v = vload2(0, cones + i + 6);
          child = vload2(0, pointers + (i/9)*2);

          if ( intersectsNode(center1, u, v, center, radius1 ))
          {
            //if the cones is at level 0 - check leaves
            if ( child.x == -2){
              rindex = (i/9)*chunk;
              intersectPAllLeaves( dir, o, bounds, tHit, v1,v2,v3,e1,e2,child.y,rindex
              #ifdef STAT_PRAY_TRIANGLE
               ,stat_rayTriangle
              #endif
              );
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
