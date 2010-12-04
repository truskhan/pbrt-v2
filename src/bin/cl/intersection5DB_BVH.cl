#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
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
  __read_only image2d_t bounds, __global int* index, __global float* tHit, float4 v1, float4 v2, float4 v3,
float4 e1, float4 e2, const int totalWidth, const int lheight, const int lwidth, const int x, const int y,
const unsigned int GID
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

      s1 = read_imagef(bounds, imageSampler, (int2)(x + j, y + i));
      if (t < s1.x  ) continue;

      if (t > tHit[totalWidth*(y + i) + x + j]) continue;
        tHit[totalWidth*(y + i) + x + j] = t;
        index[totalWidth*(y + i) + x + j] = GID;
    }
  }
}

bool intersectsNode(float4 omin, float4 omax, float2 uvmin, float2 uvmax, float4 bmin, float4 bmax) {
 float4 ocenter = (float4)0;
 float4 ray;
 float2 uv;
 float2 tmin, tmax;
 int range = 0;

//Minkowski sum of the two boxes (sum the widths/heights and position it at boxB_pos - boxA_pos).
 ray = (omax - omin)/2;
 ocenter = ray + omin;
 ocenter.w = 0;

 bmin -= ray;
 bmax += ray;

 ray = (float4)0;
 ray = normalize((float4)(bmin.x, bmin.y, bmin.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 tmin.x = tmax.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 tmin.y = tmax.y = ray.z;

 ray = normalize((float4)(bmax.x, bmax.y, bmax.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmin.x, bmax.y, bmin.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmin.x, bmax.y, bmax.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmin.x, bmin.y, bmax.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmax.x, bmin.y, bmin.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmax.x, bmax.y, bmin.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmax.x, bmin.y, bmax.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 if ( (range & 0x1100 == 0x1100) || (range & 0x0011 == 0x0011)) return true;

 return (( max(tmin.x, uvmin.x) < min(tmax.x, uvmax.x)) + EPS && (max(tmin.y, uvmin.y) < min(tmax.y, uvmax.y) + EPS));

}

__kernel void IntersectionR (
  const __global float* vertex, __read_only image2d_t dir, __read_only image2d_t o,
  __read_only image2d_t nodes, __read_only image2d_t bounds, __global float* tHit,
  __global int* index,  __global int* stack, __global GPUNode* bvh,
  int roffsetX, int xWidth, int yWidth,
  const int lwidth, const int lheight,
  int stackSize, int topLevelNodes, const int offsetGID
#ifdef STAT_RAY_TRIANGLE
 , __global int* stat_rayTriangle
#endif
) {
    // find position in global and shared arrays
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= topLevelNodes) return;
    GPUNode bvhElem = bvh[iGID];
    if ( !bvhElem.nPrimitives && !bvhElem.primOffset) return;

    // find geometry for the work-item
    float4 e1, e2;
    float4 v1, v2, v3;
    //calculate bounding box
    float4 bmin, bmax, temp_bmin, temp_bmax;

    bmin.x = bvhElem.ax;
    bmin.y = bvhElem.ay;
    bmin.z = bvhElem.az;
    bmax.x = bvhElem.bx;
    bmax.y = bvhElem.by;
    bmax.z = bvhElem.bz;

    int SPindex = 0;
    int wbeginStack = stackSize*iGID;

    //3D bounding box of the origin
    float4 omin, omax, uv;
    //2D bounding box for uv
    float2 uvmin, uvmax;

    int tempOffsetX, tempWidth, tempHeight;
    int tempX, tempY;
    for ( int j = 0; j < yWidth; j++){
      for ( int k = 0; k < xWidth; k++){
        omax = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,          yWidth + j));
        uv = read_imagef(nodes, imageSampler, (int2)(roffsetX + xWidth + k, yWidth + j));
        omin = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,          j));
        SPindex = 0;
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
            temp_bmin.x = bvhElem.ax;
            temp_bmin.y = bvhElem.ay;
            temp_bmin.z = bvhElem.az;
            temp_bmax.x = bvhElem.bx;
            temp_bmax.y = bvhElem.by;
            temp_bmax.z = bvhElem.bz;

            omax = read_imagef(nodes, imageSampler, (int2)(tempOffsetX + tempX,             tempHeight + tempY));
            uv = read_imagef(nodes, imageSampler, (int2)(tempOffsetX + tempWidth + tempX, tempHeight + tempY));
            omin = read_imagef(nodes, imageSampler, (int2)(tempOffsetX + tempX,             tempY));
            uvmin.x = uv.x;
            uvmin.y = uv.z;
            uvmax.x = uv.y;
            uvmax.y = uv.w;

            if ( intersectsNode(omin, omax, uvmin, uvmax, temp_bmin, temp_bmax) )
            {
              //if it is a rayhierarchy leaf node and bvh leaf node
              if ( !tempOffsetX && bvhElem.nPrimitives){
                   for ( int f = 0; f < bvhElem.nPrimitives; f++){
                    v1 = vload4(0, vertex + 9*(bvhElem.primOffset+f));
                    v2 = vload4(0, vertex + 9*(bvhElem.primOffset+f) + 3);
                    v3 = vload4(0, vertex + 9*(bvhElem.primOffset+f) + 6);
                    v1.w = 0; v2.w = 0; v3.w = 0;
                    e1 = v2 - v1;
                    e2 = v3 - v1;
                    intersectAllLeaves( dir, o, bounds, index, tHit, v1,v2,v3,e1,e2,
                          tempWidth*lwidth, lheight, lwidth, tempX*lwidth, tempY*lheight , offsetGID+bvhElem.primOffset+f
                          #ifdef STAT_RAY_TRIANGLE
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
