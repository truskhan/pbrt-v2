__kernel void IntersectionR2 (
  const __global float* vertex, __read_only image2d_t dir, __read_only image2d_t o,
  __read_only image2d_t nodes,  __read_only image2d_t validity, __read_only image2d_t bounds,
  __global float* tHit, __global int* index,  __global int* stack, __global GPUNode* bvh,
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
    int4 valid;
    int tempX, tempY;
    for ( int j = 0; j < yWidth; j++){
      for ( int k = 0; k < xWidth; k++){
        valid = read_imageui(validity, imageSampler, (int2)(roffsetX + k, j));
        if ( valid.x == 0) continue;
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
            valid = read_imageui(validity, imageSampler, (int2)(tempOffsetX + tempX, tempY));
            if ( valid.x == 0) continue;

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
                    intersectAllLeaves2( dir, o, bounds, index, tHit, v1,v2,v3,e1,e2,
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