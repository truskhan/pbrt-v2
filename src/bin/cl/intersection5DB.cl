#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

void intersectAllLeaves (const __global float* dir, const __global float* o,
const __global float* bounds, __global int* index, __global float* tHit, float4 v1, float4 v2, float4 v3,
float4 e1, float4 e2, int chunk, int rindex
#ifdef STAT_RAY_TRIANGLE
, __global int* stat_rayTriangle
#endif
 ){
    float4 s1, s2, d, rayd, rayo;
    float divisor, invDivisor, t, b1, b2;
    // process all rays in the cone

    for ( int i = 0; i < chunk; i++){
      rayd = vload4(0, dir + 3*rindex + 3*i);
      rayo = vload4(0, o + 3*rindex + 3*i);
      rayd.w = 0; rayo.w = 0;
      #ifdef STAT_RAY_TRIANGLE
       ++stat_rayTriangle[rindex + i];
      #endif
      s1 = cross(rayd, e2);
      divisor = dot(s1, e1);
      if ( divisor == 0.0f) continue; //degenarate triangle
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

      if (t < bounds[2*rindex + i*2]) continue;

      if (t > tHit[rindex + i]) continue;
        tHit[rindex + i] = t;
        index[rindex + i] = get_global_id(0);


    }

}

int computeChild (unsigned int threadsCount, int i){
  int index = 0;
  int levelcount = threadsCount;
  int temp;

  if ( i < 11*levelcount)
    return -1; // level 0, check rays

  while ( (index + 11*levelcount) <= i){
    temp = levelcount;
    index += 11*levelcount;
    levelcount = (levelcount+1)/2;
  }
  int offset = i - index;

  return (index - 11*temp) + 2*offset;
}

int computeRIndex (unsigned int j, const __global float* cones){
  int rindex = 0;
  for ( int i = 0; i < j; i += 11){
      rindex += cones[i + 10];
  }
  return rindex;
}

bool intersectsNode(float2 ox, float2 oy, float2 oz, float2 u, float2 v, float2 bx, float2 by, float2 bz) {
 float4 ocenter = (0);
 float2 bu, bv, uv;
 float2 tu, tv;

//Minkowski sum of the two boxes (sum the widths/heights and position it at boxB_pos - boxA_pos).
 float len = (ox.y - ox.x)/2;
 ocenter.x = len + ox.x;
 bx.x -= len; bx.y += len;
 len = (oy.y - oy.x)/2;
 ocenter.y = len + oy.x;
 by.x -= len; by.y += len;
 len = (oz.y - oz.x)/2;
 ocenter.z = len + oz.x;
 bz.x -= len; bz.y += len;

 float4 ray = (0);
 ray = normalize((float4)(bx.x, by.x, bz.x,0) - ocenter);
 tu.x = tu.y = uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 tv.x = tv.y = uv.y = ray.z;

 ray = normalize((float4)(bx.y, by.y, bz.y,0) - ocenter);
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tu.x = min(tu.x, uv.x);
 tu.y = max(tu.y, uv.x);
 tv.x = min(tv.x, uv.y);
 tv.y = max(tv.y, uv.y);

 ray = normalize((float4)(bx.x, by.y, bz.x,0) - ocenter);
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tu.x = min(tu.x, uv.x);
 tu.y = max(tu.y, uv.x);
 tv.x = min(tv.x, uv.y);
 tv.y = max(tv.y, uv.y);

 ray = normalize((float4)(bx.x, by.y, bz.y,0) - ocenter);
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tu.x = min(tu.x, uv.x);
 tu.y = max(tu.y, uv.x);
 tv.x = min(tv.x, uv.y);
 tv.y = max(tv.y, uv.y);

 ray = normalize((float4)(bx.x, by.x, bz.y,0) - ocenter);
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tu.x = min(tu.x, uv.x);
 tu.y = max(tu.y, uv.x);
 tv.x = min(tv.x, uv.y);
 tv.y = max(tv.y, uv.y);

 ray = normalize((float4)(bx.y, by.x, bz.x,0) - ocenter);
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tu.x = min(tu.x, uv.x);
 tu.y = max(tu.y, uv.x);
 tv.x = min(tv.x, uv.y);
 tv.y = max(tv.y, uv.y);

 ray = normalize((float4)(bx.y, by.y, bz.x,0) - ocenter);
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tu.x = min(tu.x, uv.x);
 tu.y = max(tu.y, uv.x);
 tv.x = min(tv.x, uv.y);
 tv.y = max(tv.y, uv.y);

 ray = normalize((float4)(bx.y, by.x, bz.y,0) - ocenter);
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tu.x = min(tu.x, uv.x);
 tu.y = max(tu.y, uv.x);
 tv.x = min(tv.x, uv.y);
 tv.y = max(tv.y, uv.y);

 if ( ( max(tu.x, u.x) < min(tu.y, u.y)) && (max(tv.x, v.x) < min(tv.y, v.y)))
  return true;

  return false;
}

__kernel void IntersectionR (
    const __global float* vertex, const __global float* dir, const __global float* o,
    const __global float* cones, const __global float* bounds, __global float* tHit,
    __global int* index,
#ifdef STAT_TRIANGLE_CONE
 __global int* stat_triangleCone,
#endif
#ifdef STAT_RAY_TRIANGLE
 __global int* stat_rayTriangle,
#endif
    __local int* stack,
     int count, int size, int height, unsigned int threadsCount
) {
    // find position in global and shared arrays
    int iGID = get_global_id(0);
    int iLID = get_local_id(0);
    #ifdef STAT_TRIANGLE_CONE
    stat_triangleCone[iGID] = 0;
    #endif

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= size) return;

    // find geometry for the work-item
    float4 e1, e2;

    float4 v1, v2, v3;
    v1 = vload4(0, vertex + 9*iGID);
    v2 = vload4(0, vertex + 9*iGID + 3);
    v3 = vload4(0, vertex + 9*iGID + 6);
    v1.w = 0; v2.w = 0; v3.w = 0;
    e1 = v2 - v1;
    e2 = v3 - v1;

    //calculate bounding box
    float2 bx, by, bz;
    bx.x = min(v1.x, v2.x);
    bx.x = min(bx.x, v3.x);
    bx.y = max(v1.x, v2.x);
    bx.y = max(bx.y, v3.x);

    by.x = min(v1.y, v2.y);
    by.x = min(by.x, v3.y);
    by.y = max(v1.y, v2.y);
    by.y = max(by.y, v3.y);

    bz.x = min(v1.z, v2.z);
    bz.x = min(bz.x, v3.z);
    bz.y = max(v1.z, v2.z);
    bz.y = max(bz.y, v3.z);

    //find number of elements in top level of the ray hieararchy
    uint levelcount = threadsCount; //end of level0
    uint num = 0;
    uint lastlevelnum = 0;

    for ( int i = 1; i < height; i++){
        lastlevelnum = levelcount;
        num += levelcount;
        levelcount = (levelcount+1)/2; //number of elements in level
    }

    int SPindex = 0;
    uint begin, rindex;
    int i = 0;
    int child;

    float2 ox, oy, oz, u, v;

    begin = 11*num;
    for ( int j = 0; j < levelcount; j++){
      // get node description
      ox = vload2(0, cones + begin + 11*j);
      oy = vload2(0, cones + begin + 11*j + 2);
      oz = vload2(0, cones + begin + 11*j + 4);
      u = vload2(0, cones + begin + 11*j + 6);
      v = vload2(0, cones + begin + 11*j + 8);

      // check if triangle intersects cone
      if ( intersectsNode(ox, oy, oz, u, v, bx, by, bz ))
      {
        #ifdef STAT_TRIANGLE_CONE
         ++stat_triangleCone[iGID];
        #endif
        //store child to the stack
        stack[iLID*(height) + SPindex++] = begin - 11*lastlevelnum + 22*j;
        while ( SPindex > 0 ){
          //take the cones from the stack and check them
          --SPindex;
          i = stack[iLID*(height) + SPindex];
          ox = vload2(0, cones + i);
          oy = vload2(0, cones + i + 2);
          oz = vload2(0, cones + i + 4);
          u = vload2(0, cones + i + 6);
          v = vload2(0, cones + i + 8);

          if ( intersectsNode(ox, oy, oz, u, v, bx, by, bz ))
          {
            #ifdef STAT_TRIANGLE_CONE
             ++stat_triangleCone[iGID];
            #endif
            child = computeChild(threadsCount,i);
            //if the cones is at level 0 - check leaves
            if ( child < 0) {
              rindex = computeRIndex(i, cones);
              intersectAllLeaves( dir, o, bounds, index, tHit, v1,v2,v3,e1,e2,cones[i+10], rindex
              #ifdef STAT_RAY_TRIANGLE
                , stat_rayTriangle
              #endif
              );
            }
            else {
              //save the intersected cone to the stack
              stack[iLID*(height) + SPindex++] = child;
            }
          }
          ox = vload2(0, cones + i + 11);
          oy = vload2(0, cones + i + 13);
          oz = vload2(0, cones + i + 15);
          u = vload2(0, cones + i + 17);
          v = vload2(0, cones + i + 19);

          if ( intersectsNode(ox, oy, oz, u, v, bx, by, bz ))
         {
            #ifdef STAT_TRIANGLE_CONE
             ++stat_triangleCone[iGID];
            #endif
            child = computeChild (threadsCount, i+11);
            //if the cone is at level 0 - check leaves
            if ( child < 0) {
              rindex = computeRIndex(i + 11, cones);
              intersectAllLeaves( dir, o, bounds, index, tHit, v1,v2,v3,e1,e2,cones[i+21],rindex
              #ifdef STAT_RAY_TRIANGLE
                , stat_rayTriangle
              #endif
              );
            }
            else {
              stack[iLID*(height) + SPindex++] = child;
            }
          }
        }
      }

    }


}
