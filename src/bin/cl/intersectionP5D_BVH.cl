#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define EPS 0.002f

typedef struct
{
  float ax, ay, az, bx, by, bz;
  uint primOffset;
  uint nPrimitives; //number of primitives, if it is interior node -> n = 0
} GPUNode;

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
      if ( stat_rayTriangle[totalWidth*(y+i) + x + j] < 80479)
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
  const __global float* vertex, __read_only image2d_t dir, __read_only image2d_t o,
  __read_only image2d_t nodes, __read_only image2d_t validity,
  __read_only image2d_t bounds, __global char* tHit,
  __global int* stack, __global GPUNode* bvh,
  int roffsetX, int xWidth, int yWidth,
  const int lwidth, const int lheight,
  const unsigned int lowerBound, const unsigned int upperBound,
   int stackSize, int topLevelNodes
#ifdef STAT_PRAY_TRIANGLE
 , __global int* stat_rayTriangle
#endif
) {
    // find position in global and shared arrays
    uint iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= topLevelNodes) return;
    GPUNode bvhElem = bvh[iGID];
    //an empty node
    if ( !bvhElem.nPrimitives && !bvhElem.primOffset) return;

    //bounding box
    float4 bmin, bmax;
    bmin.x = bvhElem.ax;
    bmin.y = bvhElem.ay;
    bmin.z = bvhElem.az;
    bmax.x = bvhElem.bx;
    bmax.y = bvhElem.by;
    bmax.z = bvhElem.bz;
    bmin.w = bmax.w = 0;

    float4 v1,v2,v3,e1,e2;

    //calculate bounding sphere
    float4 center; float radius;
    float4 temp_center; float temp_radius;
    //bounding sphere center - center of mass
    center = (bmin+bmax)/2;
    radius = length(bmax-center);

    int SPindex = 0;
    int wbeginStack = stackSize*iGID;

    float4 center1, uv;
    float radius1;
    float2 u, v;
    int4 valid;

    int tempOffsetX, tempWidth, tempHeight;
    int tempX, tempY;
    for ( int j = 0; j < yWidth; j++){
      for ( int k = 0; k < xWidth; k++){
        valid = read_imageui(validity, imageSampler, (int2)(roffsetX + k, j));
        if ( valid.x == 0) continue;
        uv = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,          yWidth + j));
        center1 = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,          j));
        //extend the triangle bounding sphere radius
        radius1 = center1.w + radius;
        center1.w = 0;
        SPindex = 0;
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
          stack[wbeginStack + SPindex + 5] = get_global_id(0);
          SPindex += 6;

          stack[wbeginStack + SPindex] = xWidth*2 ;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k + 1;
          stack[wbeginStack + SPindex + 4] = 2*j;
          stack[wbeginStack + SPindex + 5] = get_global_id(0);
          SPindex += 6;

          stack[wbeginStack + SPindex] = xWidth*2 ;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k + 1;
          stack[wbeginStack + SPindex + 4] = 2*j + 1;
          stack[wbeginStack + SPindex + 5] = get_global_id(0);
          SPindex += 6;

          stack[wbeginStack + SPindex] = xWidth*2 ;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k;
          stack[wbeginStack + SPindex + 4] = 2*j + 1;
          stack[wbeginStack + SPindex + 5] = get_global_id(0);
          SPindex += 6;

          while ( SPindex > 0) {
            SPindex -= 6;
            tempWidth = stack[wbeginStack + SPindex];
            tempHeight = stack[wbeginStack + SPindex + 1];
            tempOffsetX = stack[wbeginStack + SPindex + 2];
            tempX = stack[wbeginStack + SPindex + 3];
            tempY = stack[wbeginStack + SPindex + 4];
            iGID = stack[wbeginStack + SPindex + 5];
            bvhElem = bvh[iGID];
            //en empty BVH node
            if ( !bvhElem.nPrimitives && !bvhElem.primOffset) continue;
            bmin.x = bvhElem.ax;
            bmin.y = bvhElem.ay;
            bmin.z = bvhElem.az;
            bmax.x = bvhElem.bx;
            bmax.y = bvhElem.by;
            bmax.z = bvhElem.bz;
            bmin.w = bmax.w = 0;
            temp_center = (bmin+bmax)/2;
            temp_radius = length(bmax-temp_center);

            valid = read_imageui(validity, imageSampler, (int2)(tempOffsetX + tempX, tempY));
            if ( valid.x == 0) continue;
            uv = read_imagef(nodes, imageSampler, (int2)(tempOffsetX + tempX,             tempHeight + tempY));
            center1 = read_imagef(nodes, imageSampler, (int2)(tempOffsetX + tempX,             tempY));
            //extend the triangle bounding sphere radius
            radius1 = center1.w + temp_radius;
            center1.w = 0;

            u.x = uv.x;
            u.y = uv.z;
            v.x = uv.y;
            v.y = uv.w;

            if ( intersectsNode(center1, u, v, temp_center, radius1) )
            {
              //if it is a rayhierarchy leaf node and bvh leaf node
              if ( !tempOffsetX && bvhElem.nPrimitives){
                  if ( bvhElem.primOffset < lowerBound
                      || (bvhElem.primOffset + bvhElem.nPrimitives) >= upperBound) continue;
                  for ( int f = 0; f < bvhElem.nPrimitives; f++){
                    v1 = vload4(0, vertex + 9*(bvhElem.primOffset+f - lowerBound));
                    v2 = vload4(0, vertex + 9*(bvhElem.primOffset+f - lowerBound) + 3);
                    v3 = vload4(0, vertex + 9*(bvhElem.primOffset+f - lowerBound) + 6);
                    v1.w = 0; v2.w = 0; v3.w = 0;
                    e1 = v2 - v1;
                    e2 = v3 - v1;
                    intersectAllLeaves( dir, o, bounds, tHit, v1,v2,v3,e1,e2,
                          tempWidth*lwidth, lheight, lwidth, tempX*lwidth, tempY*lheight
                          #ifdef STAT_PRAY_TRIANGLE
                          , stat_rayTriangle
                          #endif
                          );
                  }
              }
              //it is a rayhierarchy leaf node but BVH inner node - traverse BVH
              if ( !tempOffsetX  && !bvhElem.nPrimitives){
                stack[wbeginStack + SPindex] = tempWidth;
                stack[wbeginStack + SPindex + 1] = tempHeight;
                stack[wbeginStack + SPindex + 2] = tempOffsetX ;
                stack[wbeginStack + SPindex + 3] = tempX;
                stack[wbeginStack + SPindex + 4] = tempY;
                stack[wbeginStack + SPindex + 5] = bvhElem.primOffset;
                SPindex += 6;

                stack[wbeginStack + SPindex] = tempWidth;
                stack[wbeginStack + SPindex + 1] = tempHeight;
                stack[wbeginStack + SPindex + 2] = tempOffsetX;
                stack[wbeginStack + SPindex + 3] = tempX;
                stack[wbeginStack + SPindex + 4] = tempY;
                stack[wbeginStack + SPindex + 5] = bvhElem.primOffset+1;
                SPindex += 6;
              }
              //interior nodes
              if ( tempOffsetX && !bvhElem.nPrimitives){
                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX;
                stack[wbeginStack + SPindex + 4] = 2*tempY;
                stack[wbeginStack + SPindex + 5] = bvhElem.primOffset;
                SPindex += 6;

                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX;
                stack[wbeginStack + SPindex + 4] = 2*tempY;
                stack[wbeginStack + SPindex + 5] = bvhElem.primOffset+1;
                SPindex += 6;

                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX + 1;
                stack[wbeginStack + SPindex + 4] = 2*tempY;
                stack[wbeginStack + SPindex + 5] = bvhElem.primOffset;
                SPindex += 6;

                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX + 1;
                stack[wbeginStack + SPindex + 4] = 2*tempY;
                stack[wbeginStack + SPindex + 5] = bvhElem.primOffset+1;
                SPindex += 6;

                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX + 1;
                stack[wbeginStack + SPindex + 4] = 2*tempY + 1;
                stack[wbeginStack + SPindex + 5] = bvhElem.primOffset;
                SPindex += 6;

                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX + 1;
                stack[wbeginStack + SPindex + 4] = 2*tempY + 1;
                stack[wbeginStack + SPindex + 5] = bvhElem.primOffset+1;
                SPindex += 6;

                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX;
                stack[wbeginStack + SPindex + 4] = 2*tempY + 1;
                stack[wbeginStack + SPindex + 5] = bvhElem.primOffset;
                SPindex += 6;

                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX;
                stack[wbeginStack + SPindex + 4] = 2*tempY + 1;
                stack[wbeginStack + SPindex + 5] = bvhElem.primOffset+1;
                SPindex += 6;
              }
              //rayhierarchy inner node and BVH leaf node
              if ( tempOffsetX && bvhElem.nPrimitives){
                //store the children to the stack
                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX;
                stack[wbeginStack + SPindex + 4] = 2*tempY;
                stack[wbeginStack + SPindex + 5] = iGID;
                SPindex += 6;

                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX + 1;
                stack[wbeginStack + SPindex + 4] = 2*tempY;
                stack[wbeginStack + SPindex + 5] = iGID;
                SPindex += 6;

                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX + 1;
                stack[wbeginStack + SPindex + 4] = 2*tempY + 1;
                stack[wbeginStack + SPindex + 5] = iGID;
                SPindex += 6;

                stack[wbeginStack + SPindex] = tempWidth*2;
                stack[wbeginStack + SPindex + 1] = tempHeight*2;
                stack[wbeginStack + SPindex + 2] = tempOffsetX - tempWidth*2;
                stack[wbeginStack + SPindex + 3] = 2*tempX;
                stack[wbeginStack + SPindex + 4] = 2*tempY + 1;
                stack[wbeginStack + SPindex + 5] = iGID;
                SPindex += 6;
              }
            }


          }
        }
      }
    }

}
