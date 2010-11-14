#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define EPS 0.000002f

sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

void intersectAllLeaves (
  __read_only image2d_t dir, __read_only image2d_t o,
  __read_only image2d_t bounds, __global char* tHit, float4 v1, float4 v2, float4 v3,
float4 e1, float4 e2, const int totalWidth, const int lheight, const int lwidth, const int x, const int y
#ifdef STAT_PRAY_TRIANGLE
, __global int* stat_rayTriangle
#endif
 ){
  float4 s1, s2, d, rayd, rayo;
  float divisor, invDivisor, t, b1, b2;
  // process all rays in the cone

  //read the tile
  for ( int i = 0; i < lheight; i++){
    for ( int j = 0; j < lwidth; j++) {
      rayd = read_imagef(dir, imageSampler, (int2)(x + j, y + i));
      if ( rayd.w < 0 ) continue; //not a valid ray
      rayd.w = 0;
      rayo = read_imagef(o, imageSampler, (int2)(x + j, y + i));

      #ifdef STAT_PRAY_TRIANGLE
      atom_add(stat_rayTriangle + totalWidth*(y + i) + x + j, 1);
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

      s1 = read_imagef(bounds, imageSampler, (int2)(x + j, y + i));
      if (t < s1.x || t > s1.y ) continue;

      tHit[totalWidth*(y + i) + x + j] = '1';
    }
  }


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
  const __global float* vertex, __read_only image2d_t dir, __read_only image2d_t o,
  __read_only image2d_t nodes, __read_only image2d_t validity,
  __read_only image2d_t bounds, __global char* tHit,
  __global int* stack,
  int roffsetX, int xWidth, int yWidth,
  const int lwidth, const int lheight,
    int size,  int stackSize //, __write_only image2d_t kontrola
#ifdef STAT_PRAY_TRIANGLE
 , __global int* stat_rayTriangle
#endif
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

    int SPindex = 0;
    int wbeginStack = stackSize*iGID;

    //3D bounding box of the origin
    float4 omin, omax, uv;
    //2D bounding box for uv
    float2 uvmin, uvmax;
    int4 valid;

    int tempOffsetX, tempWidth, tempHeight;
    int tempX, tempY;
    for ( int j = 0; j < yWidth; j++){
      for ( int k = 0; k < xWidth; k++){
        valid = read_imageui(validity, imageSampler, (int2)(roffsetX + k, j));
        if ( valid.x == 0) continue;
        omax = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,          yWidth + j));
        uv = read_imagef(nodes, imageSampler, (int2)(roffsetX + xWidth + k, yWidth + j));
        omin = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,          j));

        uvmin.x = uv.x;
        uvmin.y = uv.z;
        uvmax.x = uv.y;
        uvmax.y = uv.w;

        // check if triangle intersects node
        if ( intersectsNode(omin, omax, uvmin, uvmax, bmin, bmax) )
        {
          //store all 4 children to the stack (one is enough, the other 3 are nearby)
          stack[wbeginStack + SPindex] = xWidth*2 ;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k;
          stack[wbeginStack + SPindex + 4] = 2*j;
          SPindex += 5;

          stack[wbeginStack + SPindex] = xWidth*2 ;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k + 1;
          stack[wbeginStack + SPindex + 4] = 2*j;
          SPindex += 5;

          stack[wbeginStack + SPindex] = xWidth*2 ;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k + 1;
          stack[wbeginStack + SPindex + 4] = 2*j + 1;
          SPindex += 5;

          stack[wbeginStack + SPindex] = xWidth*2 ;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k;
          stack[wbeginStack + SPindex + 4] = 2*j + 1;
          SPindex += 5;

          while ( SPindex > 0) {
            SPindex -= 5;
            tempWidth = stack[wbeginStack + SPindex];
            tempHeight = stack[wbeginStack + SPindex + 1];
            tempOffsetX = stack[wbeginStack + SPindex + 2];
            tempX = stack[wbeginStack + SPindex + 3];
            tempY = stack[wbeginStack + SPindex + 4];

            valid = read_imageui(validity, imageSampler, (int2)(tempOffsetX + tempX, tempY));
            if ( valid.x == 0) continue;
            omax = read_imagef(nodes, imageSampler, (int2)(tempOffsetX + tempX,             tempHeight + tempY));
            uv = read_imagef(nodes, imageSampler, (int2)(tempOffsetX + tempWidth + tempX, tempHeight + tempY));
            omin = read_imagef(nodes, imageSampler, (int2)(tempOffsetX + tempX,             tempY));
            uvmin.x = uv.x;
            uvmin.y = uv.z;
            uvmax.x = uv.y;
            uvmax.y = uv.w;

            if ( intersectsNode(omin, omax, uvmin, uvmax, bmin, bmax) )
            {
              //if it is a leaf node
              if ( tempOffsetX == 0) {
                intersectAllLeaves( dir, o, bounds, tHit, v1,v2,v3,e1,e2,
                      tempWidth*lwidth, lheight, lwidth, tempX*lwidth, tempY*lheight
                      #ifdef STAT_PRAY_TRIANGLE
                      , stat_rayTriangle
                      #endif
                      );
              } else {
                //store the children to the stack
                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX;
                stack[wbeginStack + SPindex + 4] = 2*tempY;
                SPindex += 5;

                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX + 1;
                stack[wbeginStack + SPindex + 4] = 2*tempY;
                SPindex += 5;

                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX + 1;
                stack[wbeginStack + SPindex + 4] = 2*tempY + 1;
                SPindex += 5;

                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX;
                stack[wbeginStack + SPindex + 4] = 2*tempY + 1;
                SPindex += 5;
              }
            }


          }
        }
      }
    }

}
