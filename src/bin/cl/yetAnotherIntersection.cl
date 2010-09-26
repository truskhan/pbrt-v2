#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

void yetAnotherIntersectAllLeaves (const __global float* dir, const __global float* o,
const __global float* bounds, __global int* index, __global float* tHit, __global int* changed,
float4 v1, float4 v2, float4 v3,
float4 e1, float4 e2, int chunk, int rindex ){
    float4 s1, s2, d, rayd, rayo;
    float divisor, invDivisor, t, b1, b2;
    // process all rays in the cone

    for ( int i = 0; i < chunk; i++){
      rayd = (float4)(dir[3*rindex + 3*i], dir[3*rindex + 3*i+1], dir[3*rindex + 3*i+2],0);
      rayo = (float4)(o[3*rindex + 3*i], o[3*rindex +3*i+1], o[3*rindex + 3*i+2],0);

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
      if (t < bounds[2*rindex + i*2]) continue;

      //if (t >= tHit[rindex + i]) continue;
      if ( t > tHit[rindex + i] || index[rindex+i] == get_global_id(0)) continue;
        tHit[rindex + i] = t;
        index[rindex + i] = get_global_id(0);
        changed[get_global_id(0)] = rindex + i;
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


__kernel void YetAnotherIntersection (
    const __global float* vertex, const __global float* dir, const __global float* o,
    const __global float* cones, const __global float* bounds, __global float* tHit,
    __global int* index, __global int* changed,
    __local int* stack,
     int count, int size, int height, unsigned int threadsCount
) {
    // find position in global and shared arrays
    int iGID = get_global_id(0);
    int iLID = get_local_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= size) return;

    // find geometry for the work-item
    float4 e1, e2;

    float4 v1, v2, v3;
    v1 = (float4)(vertex[9*iGID], vertex[9*iGID+1], vertex[9*iGID+2],0);
    v2 = (float4)(vertex[9*iGID + 3], vertex[9*iGID + 4], vertex[9*iGID + 5],0);
    v3 = (float4)(vertex[9*iGID + 6], vertex[9*iGID + 7], vertex[9*iGID + 8],0);
    e1 = v2 - v1;
    e2 = v3 - v1;

    //calculate bounding sphere
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

    for ( int i = 1; i < height; i++){
        lastlevelnum = levelcount;
        num += levelcount;
        levelcount = (levelcount+1)/2; //number of elements in level
    }

    int SPindex = 0;
    uint begin, rindex;
    int i = 0;
    int child;
    float len;

    begin = 8*num;
    for ( int j = 0; j < levelcount; j++){
      // get cone description
      a = (float4)(cones[begin + 8*j],cones[begin + 8*j+1],cones[begin + 8*j+2],0);
      x = (float4)(cones[begin + 8*j+3],cones[begin + 8*j+4],cones[begin + 8*j+5],0);
      fi = cones[begin + 8*j+6];
      // check if triangle intersects cone
      len = length(center-a);
      if ( acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
      {
        //store child to the stack
        stack[iLID*(height) + SPindex++] = begin - 8*lastlevelnum + 16*j;
        while ( SPindex > 0 ){
          //take the cones from the stack and check them
          --SPindex;
          i = stack[iLID*(height) + SPindex];
          a = (float4)(cones[i],cones[i+1],cones[i+2],0);
          x = (float4)(cones[i+3],cones[i+4],cones[i+5],0);
          fi = cones[i+6];
          len = length(center-a);
          if ( len < EPS || acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
          {
            child = computeChild(threadsCount,i);
            //if the cones is at level 0 - check leaves
            if ( child < 0) {
              rindex = computeRIndex(i, cones);
              yetAnotherIntersectAllLeaves( dir, o, bounds, index, tHit, changed, v1,v2,v3,e1,e2,cones[i+7], rindex);
            }
            else {
              //save the intersected cone to the stack
              stack[iLID*(height) + SPindex++] = child;
            }
          }
          a = (float4)(cones[i+8],cones[i+9],cones[i+10],0);
          x = (float4)(cones[i+11],cones[i+12],cones[i+13],0);
          fi = cones[i+14];
          if ( len < EPS || acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
         {
            child = computeChild (threadsCount, i+8);
            //if the cone is at level 0 - check leaves
            if ( child < 0) {
              rindex = computeRIndex(i + 8, cones);
              yetAnotherIntersectAllLeaves( dir, o, bounds, index, tHit, changed, v1,v2,v3,e1,e2,cones[i+15],rindex);
            }
            else {
              stack[iLID*(height) + SPindex++] = child;
            }
          }
        }
      }

    }


}