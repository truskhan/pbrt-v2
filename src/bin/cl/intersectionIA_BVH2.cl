#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define EPS 0.000002f

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
      if ( rayd.w < 0 ) continue; //not a valid ray
      rayd.w = 0;
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
      if (t < s1.x ) continue;

      if (t > tHit[totalWidth*(y + i) + x + j]) continue;
        tHit[totalWidth*(y + i) + x + j] = t;
        index[totalWidth*(y + i) + x + j] = GID;
    }
  }


}

bool intersectsNode ( float4 bmin, float4 bmax, float4 t_omin, float4 t_omax, float4 t_dmin, float4 t_dmax){
  //compute (Bx-Ox)*(1/Vx)
  float2 s,t,u;
  float4 temp;
  //compute (Bx-0x)
  float4 omin = bmin - t_omax;
  float4 omax = bmax - t_omin;
  float4 dmin;
  float4 dmax;
  //compute (1/Vx)
  if ( (dmin.x <= 0 && dmax.x >= 0) || (dmin.y <= 0 && dmax.y >= 0) || (dmin.z <= 0 && dmax.z >= 0) )
     return true;

  dmin.x = 1/t_dmax.x;
  dmax.x = 1/t_dmin.x;
  dmin.y = 1/t_dmax.y;
  dmax.y = 1/t_dmin.y;
  dmin.z = 1/t_dmax.z;
  dmax.z = 1/t_dmin.z;

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


__kernel void IntersectionR2 (
  const __global float* vertex, __read_only image2d_t dir, __read_only image2d_t o,
  __read_only image2d_t nodes, __read_only image2d_t validity, __read_only image2d_t bounds,
  __global float* tHit, __global int* index,  __global int* stack, __global GPUNode* bvh,
  int roffsetX, int xWidth, int yWidth,
  const int lwidth, const int lheight,
  int stackSize, int topLevelNodes, int offsetGID
#ifdef STAT_RAY_TRIANGLE
 , __global int* stat_rayTriangle
#endif
) {
    // find position in global and shared arrays
    int iGID = get_global_id(0);
    __local float bbox[6*64];

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= topLevelNodes) return;
    GPUNode bvhElem = bvh[iGID];
    if ( !bvhElem.nPrimitives && !bvhElem.primOffset) return;

    // find geometry for the work-item
    float4 e1, e2;
    float4 v1, v2, v3;
    //calculate bounding box
    float4 bmin, bmax;

    bmin.x = bvhElem.ax;
    bmin.y = bvhElem.ay;
    bmin.z = bvhElem.az;
    bmax.x = bvhElem.bx;
    bmax.y = bvhElem.by;
    bmax.z = bvhElem.bz;
    bbox[get_local_id(0)*6] = bmin.x;
    bbox[get_local_id(0)*6+1] = bmin.y;
    bbox[get_local_id(0)*6+2] = bmin.z;
    bbox[get_local_id(0)*6+3] = bmax.x;
    bbox[get_local_id(0)*6+4] = bmax.y;
    bbox[get_local_id(0)*6+5] = bmax.z;

    float4 omin, omax, dmin, dmax;
    int4 valid;
    int SPindex = 0;
    int wbeginStack = stackSize*iGID;

    int tempOffsetX, tempWidth, tempHeight;
    int tempX, tempY;
    for ( int j = 0; j < yWidth; j++){
      for ( int k = 0; k < xWidth; k++){
        valid = read_imageui(validity, imageSampler, (int2)(roffsetX + k, j));
        if ( valid.x == 0) continue;
        dmax = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,          yWidth + j));
        omin = read_imagef(nodes, imageSampler, (int2)(roffsetX + xWidth + k, yWidth + j));
        dmin = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,          j));
        omax.x = omin.w;
        omax.y = dmax.w;
        omax.z = dmin.w;
        dmax.w = omin.w = dmin.w = omax.w = 0;
        SPindex = 0;

        bmin.x = bbox[get_local_id(0)*6] ;
        bmin.y = bbox[get_local_id(0)*6+1];
        bmin.z = bbox[get_local_id(0)*6+2] ;
        bmax.x = bbox[get_local_id(0)*6+3] ;
        bmax.y = bbox[get_local_id(0)*6+4] ;
        bmax.z = bbox[get_local_id(0)*6+5] ;

        // check if triangle intersects node
        if ( intersectsNode(bmin, bmax, omin, omax, dmin, dmax) )
        {
          //store all 4 children to the stack
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
            valid = read_imageui(validity, imageSampler, (int2)(tempOffsetX + tempX, tempY));
            if ( valid.x == 0) continue;

            bvhElem = bvh[iGID];
            //en empty BVH node
            if ( !bvhElem.nPrimitives && !bvhElem.primOffset) continue;
            bmin.x = bvhElem.ax;
            bmin.y = bvhElem.ay;
            bmin.z = bvhElem.az;
            bmax.x = bvhElem.bx;
            bmax.y = bvhElem.by;
            bmax.z = bvhElem.bz;

            dmax = read_imagef(nodes, imageSampler, (int2)(tempOffsetX + tempX,             tempHeight + tempY));
            omin = read_imagef(nodes, imageSampler, (int2)(tempOffsetX + tempWidth + tempX, tempHeight + tempY));
            dmin = read_imagef(nodes, imageSampler, (int2)(tempOffsetX + tempX,             tempY));
            omax.x = omin.w;
            omax.y = dmax.w;
            omax.z = dmin.w;

            if ( intersectsNode(bmin, bmax, omin, omax, dmin, dmax) )
            {
              //if it is a rayhierarchy leaf node and bvh leaf node
              if ( !tempOffsetX && bvhElem.nPrimitives){
                  for ( int f = 0; f < bvhElem.nPrimitives; f++){
                    v1 = vload4(0, vertex + 9*(bvhElem.primOffset+f ));
                    v2 = vload4(0, vertex + 9*(bvhElem.primOffset+f ) + 3);
                    v3 = vload4(0, vertex + 9*(bvhElem.primOffset+f ) + 6);
                    v1.w = 0; v2.w = 0; v3.w = 0;
                    e1 = v2 - v1;
                    e2 = v3 - v1;
                    intersectAllLeaves( dir, o, bounds, index, tHit, v1,v2,v3,e1,e2,
                          tempWidth*lwidth, lheight, lwidth, tempX*lwidth, tempY*lheight , offsetGID + bvhElem.primOffset+f
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
