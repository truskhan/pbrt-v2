__kernel void IntersectionR2 (
  const __global float* vertex, __read_only image2d_t dir, __read_only image2d_t o,
  __read_only image2d_t nodes, __read_only image2d_t validity, __read_only image2d_t bounds,
  __global float* tHit, __global int* index,  __global int* stack,
  int roffsetX, int xWidth, int yWidth,
  const int lwidth, const int lheight,
    int size, unsigned int offsetGID, int stackSize //, __write_only image2d_t kontrola
#ifdef STAT_RAY_TRIANGLE
 , __global int* stat_rayTriangle
#endif
) {
    // find position in global and shared arrays
    int iGID = get_global_id(0);
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

    int SPindex = 0;
    int wbeginStack = 5*iGID;
    uint4 valid;
    //3D bounding box of the origin
    float4 omin, omax, uv;
    //2D bounding box for uv
    float2 uvmin, uvmax;

    int j,k;
    for ( j = 0; j < yWidth; j++){
      for ( k = 0; k < xWidth; k++){
        valid = read_imageui(validity, imageSampler, (int2)(roffsetX + k, j));
        if ( valid.x == 0) continue;
        stack[wbeginStack + SPindex] = xWidth ;
        stack[wbeginStack + SPindex + 1] = yWidth;
        stack[wbeginStack + SPindex + 2] = roffsetX;
        stack[wbeginStack + SPindex + 3] = k;
        stack[wbeginStack + SPindex + 4] = j;
        SPindex += stackSize;
      }
    }

    while ( SPindex > 0) {
      SPindex -= stackSize;
      xWidth = stack[wbeginStack + SPindex];
      yWidth = stack[wbeginStack + SPindex + 1];
      roffsetX = stack[wbeginStack + SPindex + 2];
      k = stack[wbeginStack + SPindex + 3];
      j = stack[wbeginStack + SPindex + 4];

      valid = read_imageui(validity, imageSampler, (int2)(roffsetX + k, j));
      if ( valid.x == 0) continue;

      omax = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,             yWidth + j));
      uv = read_imagef(nodes, imageSampler, (int2)(roffsetX + xWidth + k, yWidth + j));
      omin = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,             j));
      uvmin.x = uv.x;
      uvmin.y = uv.z;
      uvmax.x = uv.y;
      uvmax.y = uv.w;

      if ( intersectsNode(omin, omax, uvmin, uvmax, bmin, bmax) )
      {
        //if it is a leaf node
        if ( roffsetX == 0) {
          intersectAllLeaves2( dir, o, bounds, index, tHit, v1,v2,v3,e1,e2,
                xWidth*lwidth, lheight, lwidth, k*lwidth, j*lheight , get_global_id(0)+offsetGID
                #ifdef STAT_RAY_TRIANGLE
                , stat_rayTriangle
                #endif
                );
        } else {
          //store the children to the stack
          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k;
          stack[wbeginStack + SPindex + 4] = 2*j;
          SPindex += stackSize;

          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k + 1;
          stack[wbeginStack + SPindex + 4] = 2*j;
          SPindex += stackSize;

          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k + 1;
          stack[wbeginStack + SPindex + 4] = 2*j + 1;
          SPindex += stackSize;

          stack[wbeginStack + SPindex] = xWidth*2;
          stack[wbeginStack + SPindex + 1] = yWidth*2;
          stack[wbeginStack + SPindex + 2] = roffsetX - xWidth*2;
          stack[wbeginStack + SPindex + 3] = 2*k;
          stack[wbeginStack + SPindex + 4] = 2*j + 1;
          SPindex += stackSize;
        }
      }


    }

}
