/**
 * @file yetAntoherIntersection6DB_BVH.cl
 * @author: Hana Truskova hana.truskova@seznam.cz
**/
/** method for finding primary ray-triangle intersections - control method*/
__kernel void YetAnotherIntersection (
  const __global float* vertex, __read_only image2d_t dir, __read_only image2d_t o,
  __read_only image2d_t nodes, __read_only image2d_t bounds, __global float* tHit,
  __global int* index,  __global int* stack, __global GPUNode* bvh, __global int* changed,
  int roffsetX, int xWidth, int yWidth,
  const int lwidth, const int lheight,
  int stackSize, int topLevelNodes, int offsetGID
#ifdef STAT_RAY_TRIANGLE
 , __global int* stat_rayTriangle
#endif
) {
    // find position in global and shared arrays
    int iGID = get_global_id(0);
    return;

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= topLevelNodes) return;
    GPUNode bvhElem ;

    // find geometry for the work-item
    float4 e1, e2;
    float4 v1, v2, v3;

    //calculate bounding box
    float4 bmin, bmax;
    float4 omin, omax, dmin, dmax;

    int SPindex = 0;
    int wbeginStack = 6*iGID;

    int k;
    int j;
    for ( j = 0; j < yWidth; j++){
      for ( k = 0; k < xWidth; k++){
        stack[wbeginStack + SPindex] = xWidth ;
        stack[wbeginStack + SPindex + 1] = yWidth;
        stack[wbeginStack + SPindex + 2] = roffsetX;
        stack[wbeginStack + SPindex + 3] = k;
        stack[wbeginStack + SPindex + 4] = j;
        stack[wbeginStack + SPindex + 5] = iGID;
        SPindex += stackSize;
      }
    }

    while ( SPindex >= stackSize ) {
      SPindex -= stackSize;
      xWidth = stack[wbeginStack + SPindex];
      yWidth = stack[wbeginStack + SPindex + 1];
      roffsetX = stack[wbeginStack + SPindex + 2];
      k = stack[wbeginStack + SPindex + 3];
      j = stack[wbeginStack + SPindex + 4];
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

      dmax = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,             yWidth + j));
      omin = read_imagef(nodes, imageSampler, (int2)(roffsetX + xWidth + k, yWidth + j));
      dmin = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,             j));
      omax.x = omin.w;
      omax.y = dmax.w;
      omax.z = dmin.w;

      if ( intersectsNode(omin, omax, dmin, dmax, bmin, bmax) )
      {
        //if it is a rayhierarchy leaf node and bvh leaf node
        if ( !roffsetX && bvhElem.nPrimitives){
            for ( int f = 0; f < bvhElem.nPrimitives; f++){
              v1 = vload4(0, vertex + 9*(bvhElem.primOffset+f));
              v2 = vload4(0, vertex + 9*(bvhElem.primOffset+f) + 3);
              v3 = vload4(0, vertex + 9*(bvhElem.primOffset+f) + 6);
              v1.w = 0; v2.w = 0; v3.w = 0;
              e1 = v2 - v1;
              e2 = v3 - v1;
              yetAnotherIntersectAllLeaves( dir, o, bounds, index, tHit, changed, v1,v2,v3,e1,e2,
                    xWidth*lwidth, lheight, lwidth, k*lwidth, j*lheight , offsetGID + bvhElem.primOffset+f,bvhElem.primOffset+f
                    #ifdef STAT_RAY_TRIANGLE
                    , stat_rayTriangle
                    #endif
                    );
            }
        }
        //it is a rayhierarchy leaf node but BVH inner node - traverse BVH
        if ( !roffsetX  && !bvhElem.nPrimitives){
          stack[wbeginStack + SPindex] = xWidth;
          stack[wbeginStack + SPindex + 1] = yWidth;
          stack[wbeginStack + SPindex + 2] = roffsetX ;
          stack[wbeginStack + SPindex + 3] = k;
          stack[wbeginStack + SPindex + 4] = j;
          stack[wbeginStack + SPindex + 5] = bvhElem.primOffset;
          SPindex += stackSize;

          stack[wbeginStack + SPindex] = xWidth;
          stack[wbeginStack + SPindex + 1] = yWidth;
          stack[wbeginStack + SPindex + 2] = roffsetX;
          stack[wbeginStack + SPindex + 3] = k;
          stack[wbeginStack + SPindex + 4] = j;
          stack[wbeginStack + SPindex + 5] = bvhElem.primOffset+1;
          SPindex += stackSize;
        }
        //interior nodes
        if ( roffsetX && !bvhElem.nPrimitives){
          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k;
          stack[wbeginStack + SPindex + 4] = 2*j;
          stack[wbeginStack + SPindex + 5] = bvhElem.primOffset;
          SPindex += stackSize;

          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k;
          stack[wbeginStack + SPindex + 4] = 2*j;
          stack[wbeginStack + SPindex + 5] = bvhElem.primOffset+1;
          SPindex += stackSize;

          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k + 1;
          stack[wbeginStack + SPindex + 4] = 2*j;
          stack[wbeginStack + SPindex + 5] = bvhElem.primOffset;
          SPindex += stackSize;

          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k + 1;
          stack[wbeginStack + SPindex + 4] = 2*j;
          stack[wbeginStack + SPindex + 5] = bvhElem.primOffset+1;
          SPindex += stackSize;

          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k + 1;
          stack[wbeginStack + SPindex + 4] = 2*j + 1;
          stack[wbeginStack + SPindex + 5] = bvhElem.primOffset;
          SPindex += stackSize;

          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k + 1;
          stack[wbeginStack + SPindex + 4] = 2*j + 1;
          stack[wbeginStack + SPindex + 5] = bvhElem.primOffset+1;
          SPindex += stackSize;

          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k;
          stack[wbeginStack + SPindex + 4] = 2*j + 1;
          stack[wbeginStack + SPindex + 5] = bvhElem.primOffset;
          SPindex += stackSize;

          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k;
          stack[wbeginStack + SPindex + 4] = 2*j + 1;
          stack[wbeginStack + SPindex + 5] = bvhElem.primOffset+1;
          SPindex += stackSize;
        }
        //rayhierarchy inner node and BVH leaf node
        if ( roffsetX && bvhElem.nPrimitives){
          //store the children to the stack
          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k;
          stack[wbeginStack + SPindex + 4] = 2*j;
          stack[wbeginStack + SPindex + 5] = iGID;
          SPindex += stackSize;

          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k + 1;
          stack[wbeginStack + SPindex + 4] = 2*j;
          stack[wbeginStack + SPindex + 5] = iGID;
          SPindex += stackSize;

          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k + 1;
          stack[wbeginStack + SPindex + 4] = 2*j + 1;
          stack[wbeginStack + SPindex + 5] = iGID;
          SPindex += stackSize;

          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k;
          stack[wbeginStack + SPindex + 4] = 2*j + 1;
          stack[wbeginStack + SPindex + 5] = iGID;
          SPindex += stackSize;
        }

      }


    }

}
