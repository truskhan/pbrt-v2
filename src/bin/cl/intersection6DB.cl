#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

void intersectAllLeaves (const __global float* dir, const __global float* o,
const __global float* bounds, __global int* index, __global float* tHit, float4 v1, float4 v2, float4 v3,
float4 e1, float4 e2, int chunk, int rindex, const unsigned int offsetGID
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
        index[rindex + i] = get_global_id(0) + offsetGID;


    }

}

int computeRIndex( unsigned int j, const __global float* cones, const __global int* pointers){
  int rindex = 0;
  for ( int i = 0; i < j; i += 13){
    rindex += pointers[(int)(cones[i+12]) + 1];
  }
  return rindex;
}

bool intersectsNode(float4 omin, float4 omax, float4 uvmin, float4 uvmax, float4 bmin, float4 bmax) {
 float4 ocenter = (float4)0;
 float4 ray;
 float4 tmin, tmax;

//Minkowski sum of the two boxes (sum the widths/heights and position it at boxB_pos - boxA_pos).
 ray = omax - omin;
 ocenter = ray/2 + omin;
 ocenter.w = 0;

 uvmin.w = uvmax.w = 0;

 ray = (float4)0;
 ray = normalize((float4)(bmin.x, bmin.y, bmin.z,0) - ocenter);
 tmin = ray;
 tmax = ray;

 ray = normalize((float4)(bmin.x, bmin.y, bmax.z,0) - ocenter);
 tmin = min(tmin, ray);
 tmax = max(tmax, ray);

 ray = normalize((float4)(bmin.x, bmax.y, bmin.z,0) - ocenter);
 tmin = min(tmin, ray);
 tmax = max(tmax, ray);

 ray = normalize((float4)(bmin.x, bmax.y, bmax.z,0) - ocenter);
 tmin = min(tmin, ray);
 tmax = max(tmax, ray);

 ray = normalize((float4)(bmax.x, bmin.y, bmin.z,0) - ocenter);
 tmin = min(tmin, ray);
 tmax = max(tmax, ray);

 ray = normalize((float4)(bmax.x, bmax.y, bmin.z,0) - ocenter);
 tmin = min(tmin, ray);
 tmax = max(tmax, ray);

 ray = normalize((float4)(bmax.x, bmin.y, bmax.z,0) - ocenter);
 tmin = min(tmin, ray);
 tmax = max(tmax, ray);

 ray = normalize((float4)(bmax.x, bmax.y, bmax.z,0) - ocenter);
 tmin = min(tmin, ray);
 tmax = max(tmax, ray);

 tmin = max(tmin, uvmin);
 tmax = min(tmax, uvmax);

 if ( tmin.x < tmax.x && tmin.y < tmax.y && tmin.z < tmax.z )
   return true;

  return false;
}

__kernel void IntersectionR (
    const __global float* vertex, const __global float* dir, const __global float* o,
    const __global float* cones, const __global int* pointers, const __global float* bounds,
     __global float* tHit, __global int* index,  __local int* stack,
     int count, int size, int height, unsigned int threadsCount, const unsigned int offsetGID
#ifdef STAT_TRIANGLE_CONE
 ,__global int* stat_triangleCone
#endif
#ifdef STAT_RAY_TRIANGLE
 ,__global int* stat_rayTriangle
#endif
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
    float4 bmin, bmax;

    bmin = min(v1,v2);
    bmin = min(bmin, v3);
    bmax = max(v1,v2);
    bmax = max(bmax, v3);

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
    int2 child;

    //3D bounding box of the origin
    float4 omin, omax;
    //2D bounding box for uv
    float4 uvmin, uvmax;

    begin = 13*num;
    for ( int j = 0; j < levelcount; j++){
      // get node description
      omin = vload4(0, cones + begin + 13*j);
      omax = vload4(0, cones + begin + 13*j + 3);
      uvmin = vload4(0, cones + begin + 13*j + 6);
      uvmax = vload4(0, cones + begin + 13*j + 9);
      child = vload2(0, pointers + (begin/13 + j)*2);

      // check if triangle intersects cone
      if ( intersectsNode( omin, omax , uvmin, uvmax, bmin, bmax ))
      {
        #ifdef STAT_TRIANGLE_CONE
         ++stat_triangleCone[iGID];
        #endif
        //store childs to the stack
        stack[wbeginStack + SPindex++] = child.x;
        if ( child.y != -1)
          stack[wbeginStack + SPindex++] = child.y;
        while ( SPindex > 0 ){
          //take the cones from the stack and check them
          --SPindex;
          i = stack[wbeginStack + SPindex];
          omin = vload4(0, cones + i);
          omax = vload4(0, cones + i + 3);
          uvmin = vload4(0, cones + i + 6);
          uvmax = vload4(0, cones + i + 9);
          child = vload2(0, pointers + (i/13)*2);

          if ( intersectsNode( omin, omax , uvmin, uvmax, bmin, bmax ))
          {
            #ifdef STAT_TRIANGLE_CONE
             ++stat_triangleCone[iGID];
            #endif
            //if the cones is at level 0 - check leaves
            if ( child.x == -2) {
              rindex = computeRIndex(i, cones, pointers);
              intersectAllLeaves( dir, o, bounds, index, tHit, v1,v2,v3,e1,e2, child.y, rindex, offsetGID
              #ifdef STAT_RAY_TRIANGLE
                , stat_rayTriangle
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
