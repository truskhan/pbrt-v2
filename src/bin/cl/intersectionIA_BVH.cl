#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
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
    rayd.w = rayo.w = 0;

    for ( int i = 0; i < chunk; i++){
      rayd = vload4(0, dir + 3*rindex + 3*i);
      rayo = vload4(0, o + 3*rindex + 3*i);
      rayd.w = 0; rayo.w = 0;
      #ifdef STAT_RAY_TRIANGLE
      atom_add(stat_rayTriangle + rindex + i, 1);
      // ++stat_rayTriangle[rindex + i];
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




bool intersectsNode ( float4 bmin, float4 bmax, float4 omin, float4 omax, float4 dmin, float4 dmax){
  //compute (Bx-Ox)*(1/Vx)
  float2 s,t,u;
  float4 temp;
  //compute (Bx-0x)
  temp = omin;
  omin = bmin - omax;
  omax = bmax - temp;
  //compute (1/Vx)
  if ( dmin.x <= 0 && dmax.x >= 0){
    return true;
  } else {
    temp.x = dmin.x;
    dmin.x = 1/dmax.x;
    dmax.x = 1/temp.x;
  }
  if ( dmin.y <= 0 && dmax.y >= 0){
    return true;
  } else {
    temp.x = dmin.y;
    dmin.y = 1/dmax.y;
    dmax.y = 1/temp.x;
  }
  if ( dmin.z <= 0 && dmax.z >= 0){
    return true;
  } else {
    temp.x = dmin.z;
    dmin.z = 1/dmax.z;
    dmax.z = 1/temp.x;
  }

  temp.x = omin.x*dmin.x;
  temp.y = omax.x*dmin.x;
  temp.z = omax.x*dmax.x;
  temp.w = omin.x*dmax.x;
  s.x = min(temp.x, temp.y);
  s.x = min(s.x, temp.z);
  s.x = min(s.x, temp.w);
  s.y = max(temp.x, temp.y);
  s.y = max(s.y, temp.z);
  s.y = max(s.y, temp.w);

  temp.x = omin.y*dmin.y;
  temp.y = omax.y*dmin.y;
  temp.z = omax.y*dmax.y;
  temp.w = omin.y*dmax.y;
  t.x = min(temp.x, temp.y);
  t.x = min(t.x, temp.z);
  t.x = min(t.x, temp.w);
  t.y = max(temp.x, temp.y);
  t.y = max(t.y, temp.z);
  t.y = max(t.y, temp.w);

  temp.x = omin.z*dmin.z;
  temp.y = omax.z*dmin.z;
  temp.z = omax.z*dmax.z;
  temp.w = omin.z*dmax.z;
  u.x = min(temp.x, temp.y);
  u.x = min(u.x, temp.z);
  u.x = min(u.x, temp.w);
  u.y = max(temp.x, temp.y);
  u.y = max(u.y, temp.z);
  u.y = max(u.y, temp.w);

  s.x = max(s.x, t.x);
  s.x = max(s.x, u.x);
  s.y = min(s.y, t.y);
  s.y = min(s.y, u.y);

  return (s.x < s.y);
}

int computeRIndex( unsigned int j, const __global float* cones, const __global int* pointers){
  int rindex = 0;
  int temp = 1;
  for ( int i = 0; i < j; i += 13){
    rindex += pointers[temp];
    temp += 2;
  }
  return rindex;
}

__kernel void IntersectionR (
    const __global float* vertex, const __global float* dir, const __global float* o,
    const __global float* cones, const __global int* pointers, const __global float* bounds, __global float* tHit,
    __global int* index,  __local int* stack, __local float* local_cones, __local float* local_pointers,
     int count, int size, const int height, unsigned int threadsCount, unsigned int offsetGID
#ifdef STAT_TRIANGLE_CONE
 , __global int* stat_triangleCone
#endif
#ifdef STAT_RAY_TRIANGLE
 , __global int* stat_rayTriangle
#endif
) {
    // find position in global and shared arrays
    int iGID = get_global_id(0);
    int iLID = get_local_id(0);
    #ifdef STAT_TRIANGLE_CONE
    stat_triangleCone[iGID + offsetGID] = 0;
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
    bmin.w = bmax.w = 0;
    bmin = min(v1,v2);
    bmin = min(bmin, v3);
    bmax = max(v1,v2);
    bmax = max(bmax, v3);

    float4 omin, omax, dmin, dmax;

    //find number of elements in top level of the ray hieararchy
    uint levelcount = threadsCount; //end of level0
    uint num = 0;
    uint lastlevelnum = 0;

    int2 child;
    for ( int i = 1; i <= height; i++){
        lastlevelnum = levelcount;
        num += levelcount;
        levelcount = (levelcount+1)/2; //number of elements in level
    }

    int SPindex = 0;
    int wbeginStack = (2 + height*(height+1)/2)*iLID;
    uint begin;
    int rindex;
    int i = 0;
    begin = 13*num;

    /*j = iGID;
    while ( j < 13*levelcount){
       local_cones[j] = cones[begin+j];
       j += get_local_size(0);
    }

    barrier(CLK_LOCAL_MEM_FENCE);*/
    int j, fn;
    for ( j = 0; j < levelcount; j++){
      fn = iGID;
      while ( fn < 13*get_local_size(0)){
        local_cones[fn] = cones[begin+fn+13*j];
        fn += get_local_size(0);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for ( fn = 0; fn < get_local_size(0) && j < levelcount; fn++, j++) {
      // get IA description
      omin.x = local_cones[13*j];
      omin.y = local_cones[13*j+1];
      omin.z = local_cones[13*j+2];
      omax.x = local_cones[13*j+3];
      omax.y = local_cones[13*j+4];
      omax.z = local_cones[13*j+5];
      dmin.x = local_cones[13*j+6];
      dmin.y = local_cones[13*j+7];
      dmin.z = local_cones[13*j+8];
      dmax.x = local_cones[13*j+9];
      dmax.y = local_cones[13*j+10];
      dmax.z = local_cones[13*j+11];

      /*omin = vload4(0, cones + begin+13*j);
      omax = vload4(0, cones + begin+13*j + 3);
      dmin = vload4(0, cones + begin+13*j + 6);
      dmax = vload4(0, cones + begin+13*j + 9);*/
      child = vload2(0, pointers + (begin/13 + j)*2);

      // check if triangle intersects IA node
      if ( intersectsNode(bmin, bmax, omin, omax, dmin, dmax) )
      {
        #ifdef STAT_TRIANGLE_CONE
         ++stat_triangleCone[iGID + offsetGID];
        #endif
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
        dmin = vload4(0, cones + i + 6);
        dmax = vload4(0, cones + i + 9);
        child = vload2(0, pointers + (i/13)*2);

        if ( intersectsNode(bmin, bmax, omin, omax, dmin, dmax))
        {
          #ifdef STAT_TRIANGLE_CONE
           ++stat_triangleCone[iGID + offsetGID];
          #endif
          //if the cones is at level 0 - check leaves
          if ( child.x == -2) {
            rindex = computeRIndex(i,cones,pointers);
            intersectAllLeaves( dir, o, bounds, index, tHit, v1,v2,v3,e1,e2,child.y, rindex, offsetGID
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
      ++j;
      }
      }

    }



}