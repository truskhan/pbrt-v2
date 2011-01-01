#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
__kernel void IntersectionP (
  const __global float* vertex, __read_only image2d_t dir, __read_only image2d_t o,
  __read_only image2d_t nodes, __read_only image2d_t validity,
  __read_only image2d_t bounds, __global char* tHit,
  __global int* stack,
  int roffsetX, int xWidth, int yWidth,
  const int lwidth, const int lheight,
    int size
#ifdef STAT_PRAY_TRIANGLE
 , __global int* stat_rayTriangle
#endif
#ifdef STAT_ALL
 , __global unsigned int* stats
#endif
) {

    int GID = get_global_id(0);
    if ( GID >= size) return;
    int stackSize = 5*size;
    float4 e1, e2;
    float4 v1, v2, v3;
    float4 bmin, bmax;
    float4 omin, omax, dmin, dmax;
    uint4 valid;
    int SPindex = 5*GID;
    int j,k;

    // find geometry for the work-item
    v1 = vload4(0, vertex + 9*GID);
    v2 = vload4(0, vertex + 9*GID + 3);
    v3 = vload4(0, vertex + 9*GID + 6);
    v1.w = 0; v2.w = 0; v3.w = 0;
    e1 = v2 - v1;
    e2 = v3 - v1;

    //calculate bounding box
    bmin = min(v1,v2);
    bmin = min(bmin, v3);
    bmax = max(v1,v2);
    bmax = max(bmax, v3);

    for ( j = 0; j < yWidth; j++){
      for ( k = 0; k < xWidth; k++){
        valid = read_imageui(validity, imageSampler, (int2)(roffsetX + k, j));
        if ( valid.x == 0) continue;
        stack[ SPindex] = xWidth ;
        stack[ SPindex + 1] = yWidth;
        stack[ SPindex + 2] = roffsetX;
        stack[ SPindex + 3] = k;
        stack[ SPindex + 4] = j;
        SPindex += stackSize;
      }
    }

    while ( SPindex >= stackSize ) {
      SPindex -= stackSize;
      xWidth = stack[ SPindex];
      yWidth = stack[ SPindex + 1];
      roffsetX = stack[ SPindex + 2];
      k = stack[ SPindex + 3];
      j = stack[ SPindex + 4];

      valid = read_imageui(validity, imageSampler, (int2)(roffsetX + k, j));
      if ( valid.x == 0) continue;
      dmax = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,             yWidth + j));
      omin = read_imagef(nodes, imageSampler, (int2)(roffsetX + xWidth + k, yWidth + j));
      dmin = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,             j));
      omax.x = omin.w;
      omax.y = dmax.w;
      omax.z = dmin.w;

      #ifdef STAT_ALL
        atom_add(stats+1, 1);
      #endif

      if ( intersectsNode(omin, omax, dmin, dmax, bmin, bmax) )
      {
        //if it is a leaf node
        if ( roffsetX == 0) {
          intersectAllLeavesP( dir, o, bounds, tHit, v1,v2,v3,e1,e2,
                xWidth*lwidth, lheight, lwidth, k*lwidth, j*lheight
                #ifdef STAT_PRAY_TRIANGLE
                , stat_rayTriangle
                #endif
                #ifdef STAT_ALL
                , stats
                #endif
                );
        } else {
          //store the children to the stack
          stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k;
          stack[ SPindex + 4] = 2*j;
          SPindex += stackSize;

          stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k + 1;
          stack[ SPindex + 4] = 2*j;
          SPindex += stackSize;

          stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k + 1;
          stack[ SPindex + 4] = 2*j + 1;
          SPindex += stackSize;

          stack[ SPindex] = xWidth*2;
          stack[ SPindex + 1] = yWidth*2;
          stack[ SPindex + 2] = roffsetX - xWidth*2;
          stack[ SPindex + 3] = 2*k;
          stack[ SPindex + 4] = 2*j + 1;
          SPindex += stackSize;
        }
      }

    }

}
