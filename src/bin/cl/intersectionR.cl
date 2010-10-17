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

  if ( i < 8*levelcount)
    return -1; // level 0, check rays

  while ( (index + 8*levelcount) <= i){
    temp = levelcount;
    index += 8*levelcount;
    levelcount = (levelcount+1)/2;
  }
  int offset = i - index;

  return (index - 8*temp) + 2*offset;
}

int computeRIndex (unsigned int j, const __global float* cones){
  int rindex = 0;
  for ( int i = 0; i < j; i += 8){
      rindex += cones[i + 7];
  }
  return rindex;
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

    //calculate bounding sphere
    //vizualizace pruseciku s paprskem a kuzel,trojuhelnikem, ktery se pocitaly

    float4 center; float radius;
    //bounding sphere center - center of mass
    center = (v1+v2+v3)/3;
    radius = length(v1-center);

    float4 a,x;
    float fi;

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
    uint begin, rindex;
    int i = 0;
    int child;
    float len;

    begin = 8*num;
    for ( int j = 0; j < levelcount; j++){
      // get cone description
      a = vload4(0, cones + begin+8*j);
      x = vload4(0, cones + begin+8*j + 3);
      fi = x.w;
      a.w = 0; x.w = 0;
      // check if triangle intersects cone
      len = length(center-a);
      if ( acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
      {
        #ifdef STAT_TRIANGLE_CONE
         ++stat_triangleCone[iGID];
        #endif
        //store child to the stack
        stack[wbeginStack + SPindex++] = begin - 8*lastlevelnum + 16*j;
        stack[wbeginStack + SPindex++] = begin - 8*lastlevelnum + 16*j + 8;
        while ( SPindex > 0 ){
          //take the cones from the stack and check them
          --SPindex;
          i = stack[wbeginStack + SPindex];
          a = vload4(0, cones+i);
          x = vload4(0, cones+i+3);
          fi = x.w;
          a.w = 0; x.w = 0;
          len = length(center-a);
          if ( len < EPS || acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
          {
            #ifdef STAT_TRIANGLE_CONE
             ++stat_triangleCone[iGID];
            #endif
            child = computeChild(threadsCount,i);
            //if the cones is at level 0 - check leaves
            if ( child < 0) {
              rindex = computeRIndex(i, cones);
              intersectAllLeaves( dir, o, bounds, index, tHit, v1,v2,v3,e1,e2,cones[i+7], rindex
              #ifdef STAT_RAY_TRIANGLE
                , stat_rayTriangle
              #endif
              );
            }
            else {
              //save the intersected cone to the stack
              stack[wbeginStack + SPindex++] = child;
              stack[wbeginStack + SPindex++] = child+8;
            }
          }

        }
      }

    }


}
