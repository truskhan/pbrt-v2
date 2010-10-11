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

bool intersectsNode(float4 omin, float4 omax, float2 uvmin, float2 uvmax, float4 bmin, float4 bmax) {
 float4 ocenter = (float4)0;
 float4 ray;
 float2 uv;
 float2 tmin, tmax;

//Minkowski sum of the two boxes (sum the widths/heights and position it at boxB_pos - boxA_pos).
 ray = omax - omin;
 ocenter = ray/2 + omin;
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

 ray = normalize((float4)(bmin.x, bmax.y, bmin.x,0) - ocenter);
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
    v1 = vload4(0, vertex + 9*iGID);
    v2 = vload4(0, vertex + 9*iGID + 3);
    v3 = vload4(0, vertex + 9*iGID + 6);
    v1.w = 0; v2.w = 0; v3.w = 0;
    e1 = v2 - v1;
    e2 = v3 - v1;

    //calculate bounding box
    float4 bmin, bmax;

    bmin = min(v1,v2);
    bmin = min(bmin, v3);
    bmax = max(v1,v2);
    bmax = max(bmax, v3);

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

      // check if triangle intersects cone
      if ( intersectsNode( omin, omax , uvmin, uvmax, bmin, bmax ))
      {
        //store child to the stack
        stack[iLID*(height) + SPindex++] = begin - 11*lastlevelnum + 22*j;
        stack[iLID*(height) + SPindex++] = begin - 11*lastlevelnum + 22*j + 11;
        while ( SPindex > 0 ){
          //take the cones from the stack and check them
          --SPindex;
          i = stack[iLID*(height) + SPindex];
          omin = vload4(0, cones + i);
          omax = vload4(0, cones + i + 3);
          uvmin = vload2(0, cones + i + 6);
          uvmax = vload2(0, cones + i + 8);

          if ( intersectsNode( omin, omax , uvmin, uvmax, bmin, bmax ))
          {
            child = computeChild(threadsCount,i);
            //if the cones is at level 0 - check leaves
            if ( child < 0) {
              rindex = computeRIndex(i, cones);
              yetAnotherIntersectAllLeaves( dir, o, bounds, index, tHit, changed, v1,v2,v3,e1,e2,cones[i+10], rindex);
            }
            else {
              //save the intersected cone to the stack
              stack[iLID*(height) + SPindex++] = child;
              stack[iLID*(height) + SPindex++] = child + 11;
            }
          }

        }
      }

    }


}