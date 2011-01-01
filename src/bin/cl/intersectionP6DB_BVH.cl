__kernel void IntersectionP (
  const __global float* vertex, __read_only image2d_t dir, __read_only image2d_t o,
  __read_only image2d_t nodes, __read_only image2d_t validity,
  __read_only image2d_t bounds, __global char* tHit,
  __global int* stack, __global GPUNode* bvh,
  int roffsetX, int xWidth, int yWidth,
  const int lwidth, const int lheight,
  unsigned int topLevelNodes
#ifdef STAT_PRAY_TRIANGLE
 , __global int* stat_rayTriangle
#endif
#ifdef STAT_ALL
 , __global unsigned int* stats
#endif
) {
    // find position in global and shared arrays
    unsigned int iGID = get_global_id(0);

    // bound check
    if (iGID >= topLevelNodes) return;
    unsigned int stackSize = 6*topLevelNodes;
    GPUNode bvhElem ;
    uint4 valid;

    // find geometry for the work-item
    float4 v1, v2, v3;

    //calculate bounding box
    float4 bmin, bmax;
    float4 omin, omax, dmin, dmax;

    unsigned int SPindex = 6*iGID;

    int k;
    int j;
    for ( j = 0; j < yWidth; j++){
      for ( k = 0; k < xWidth; k++){
        valid = read_imageui(validity, imageSampler, (int2)(roffsetX + k, j));
        if ( valid.x != 0) {
        stack[ SPindex] = xWidth ;
        stack[ SPindex + 1] = yWidth;
        stack[ SPindex + 2] = roffsetX;
        stack[ SPindex + 3] = k;
        stack[ SPindex + 4] = j;
        stack[ SPindex + 5] = iGID;
        SPindex += stackSize;
        }
      }
    }

    while ( SPindex >= stackSize) {
      SPindex -= stackSize;
      xWidth = stack[ SPindex];
      yWidth = stack[ SPindex + 1];
      roffsetX = stack[ SPindex + 2];
      k = stack[ SPindex + 3];
      j = stack[ SPindex + 4];
      valid = read_imageui(validity, imageSampler, (int2)(roffsetX + k, j));
      if ( valid.x != 0) {

      iGID = stack[ SPindex + 5];
      bvhElem = bvh[iGID];
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

      #ifdef STAT_ALL
        atom_add(stats+1,1);
      #endif

      if ( intersectsNode(omin, omax, dmin, dmax, bmin, bmax) )
      {
        //if it is a rayhierarchy leaf node and bvh leaf node
        if ( roffsetX == 0 && bvhElem.nPrimitives > 0){
            for ( unsigned int f = 0; f < bvhElem.nPrimitives; f++){
              v1 = vload4(0, vertex + 9*(bvhElem.primOffset+f));
              v2 = vload4(0, vertex + 9*(bvhElem.primOffset+f) + 3);
              v3 = vload4(0, vertex + 9*(bvhElem.primOffset+f) + 6);
              v1.w = 0; v2.w = 0; v3.w = 0;
          intersectAllLeavesP( dir, o, bounds, tHit, v1,v2,v3,v2 - v1,v3 - v1,
                xWidth*lwidth, lheight, lwidth, k*lwidth, j*lheight
                #ifdef STAT_PRAY_TRIANGLE
                , stat_rayTriangle
                #endif
                #ifdef STAT_ALL
                , stats
                #endif
                );
            }
        }

        //it is a rayhierarchy leaf node but BVH inner node - traverse BVH
        if ( roffsetX == 0 && bvhElem.nPrimitives == 0){
          stack[ SPindex] = xWidth;
          stack[ SPindex + 1] = yWidth;
          stack[ SPindex + 2] = roffsetX ;
          stack[ SPindex + 3] = k;
          stack[ SPindex + 4] = j;
          stack[ SPindex + 5] = bvhElem.primOffset;
          SPindex += stackSize;

          stack[ SPindex] = xWidth;
          stack[ SPindex + 1] = yWidth;
          stack[ SPindex + 2] = roffsetX;
          stack[ SPindex + 3] = k;
          stack[ SPindex + 4] = j;
          stack[ SPindex + 5] = bvhElem.primOffset+1;
          SPindex += stackSize;
        }
        //interior nodes
        if ( roffsetX > 0 && bvhElem.nPrimitives == 0){
          stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k;
          stack[ SPindex + 4] = 2*j;
          stack[ SPindex + 5] = bvhElem.primOffset;
          SPindex += stackSize;

          stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k;
          stack[ SPindex + 4] = 2*j;
          stack[ SPindex + 5] = bvhElem.primOffset+1;
          SPindex += stackSize;

          stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k + 1;
          stack[ SPindex + 4] = 2*j;
          stack[ SPindex + 5] = bvhElem.primOffset;
          SPindex += stackSize;

          stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k + 1;
          stack[ SPindex + 4] = 2*j;
          stack[ SPindex + 5] = bvhElem.primOffset+1;
          SPindex += stackSize;

          stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k + 1;
          stack[ SPindex + 4] = 2*j + 1;
          stack[ SPindex + 5] = bvhElem.primOffset;
          SPindex += stackSize;

          stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k + 1;
          stack[ SPindex + 4] = 2*j + 1;
          stack[ SPindex + 5] = bvhElem.primOffset+1;
          SPindex += stackSize;

          stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k;
          stack[ SPindex + 4] = 2*j + 1;
          stack[ SPindex + 5] = bvhElem.primOffset;
          SPindex += stackSize;

         stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k;
          stack[ SPindex + 4] = 2*j + 1;
          stack[ SPindex + 5] = bvhElem.primOffset+1;
          SPindex += stackSize;
        }
        //rayhierarchy inner node and BVH leaf node
        if ( roffsetX > 0 && bvhElem.nPrimitives > 0){
          //store the children to the stack
          stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k;
          stack[ SPindex + 4] = 2*j;
          stack[ SPindex + 5] = iGID;
          SPindex += stackSize;

          stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k + 1;
          stack[ SPindex + 4] = 2*j;
          stack[ SPindex + 5] = iGID;
          SPindex += stackSize;

          stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k + 1;
          stack[ SPindex + 4] = 2*j + 1;
          stack[ SPindex + 5] = iGID;
          SPindex += stackSize;

          stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k;
          stack[ SPindex + 4] = 2*j + 1;
          stack[ SPindex + 5] = iGID;
          SPindex += stackSize;
        }

      }
      }

    }
}
