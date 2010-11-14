#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define EPS 0.000002f

sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

void intersectAllLeaves (
  __read_only image2d_t dir, __read_only image2d_t o,
const __global float* bounds, __global int* index, __global float* tHit, __global int* changed,
float4 v1, float4 v2, float4 v3, float4 e1, float4 e2, const int totalWidth, const int lheight,
const int lwidth, const int x, const int y, const unsigned int offsetGID
#ifdef STAT_RAY_TRIANGLE
, __global int* stat_rayTriangle
#endif
 ){
  float4 s1, s2, d, rayd, rayo;
  float divisor, invDivisor, t, b1, b2;
  // process all rays in the cone

  //read the tile
  for ( int i = 0; i < lheight; i++){
    for ( int j = 0; j < lwidth; j++) {
      #ifdef STAT_RAY_TRIANGLE
      atom_add(stat_rayTriangle + totalWidth*(y + i) + x + j, 1);
      #endif
      rayd = read_imagef(dir, imageSampler, (int2)(x + j, y + i));
      rayo = read_imagef(o, imageSampler, (int2)(x + j, y + i));

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

      if (t < bounds[2*(totalWidth*(y + i) + x + j)]) continue;

      if ( t > tHit[totalWidth*(y + i) + x + j]
        || index[totalWidth*(y + i) + x + j] == (get_global_id(0)+ offsetGID)) continue;

        tHit[totalWidth*(y + i) + x + j] = t;
        index[totalWidth*(y + i) + x + j] = get_global_id(0) + offsetGID;
        changed[get_global_id(0)] = totalWidth*(y + i) + x + j;
    }
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

__kernel void YetAnotherIntersection (
  const __global float* vertex, __read_only image2d_t dir, __read_only image2d_t o,
  __read_only image2d_t nodes, const __global float* bounds, __global float* tHit,
  __global int* index, __global int* changed, __global int* stack,
  int roffsetX, int xWidth, int yWidth,
  const int lwidth, const int lheight,
    int size, unsigned int offsetGID, int stackSize //, __write_only image2d_t kontrola
#ifdef STAT_RAY_TRIANGLE
 , __global int* stat_rayTriangle
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

    //calculate bounding sphere
    float4 center; float radius;
    //bounding sphere center - center of mass
    center = (v1+v2+v3)/3;
    radius = length(v1-center);

    int SPindex = 0;
    int wbeginStack = stackSize*iGID;

    float4 center1, uv;
    float radius1;
    float2 u, v;

    int tempOffsetX, tempWidth, tempHeight;
    int tempX, tempY;
    for ( int j = 0; j < yWidth; j++){
      for ( int k = 0; k < xWidth; k++){
        uv = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,          yWidth + j));
        center1 = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,          j));
        //extend the triangle bounding sphere radius
        radius1 = center1.w + radius;
        center1.w = 0;

        u.x = uv.x;
        u.y = uv.z;
        v.x = uv.y;
        v.y = uv.w;

        // check if triangle intersects node
        if ( intersectsNode(center1, u, v, center, radius1) )
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

            uv = read_imagef(nodes, imageSampler, (int2)(tempOffsetX + tempX,             tempHeight + tempY));
            center1 = read_imagef(nodes, imageSampler, (int2)(tempOffsetX + tempX,             tempY));
            //extend the triangle bounding sphere radius
            radius1 = center1.w + radius;
            center1.w = 0;

            u.x = uv.x;
            u.y = uv.z;
            v.x = uv.y;
            v.y = uv.w;

            if ( intersectsNode(center1, u, v, center, radius1) )
            {
              //if it is a leaf node
              if ( tempOffsetX == 0) {
                intersectAllLeaves( dir, o, bounds, index, tHit, changed, v1,v2,v3,e1,e2,
                      tempWidth*lwidth, lheight, lwidth, tempX*lwidth, tempY*lheight , offsetGID
                      #ifdef STAT_RAY_TRIANGLE
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
