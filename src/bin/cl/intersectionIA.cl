__kernel void IntersectionR (
  const __global float* vertex, __read_only image2d_t dir, __read_only image2d_t o,
  __read_only image2d_t nodes, __read_only image2d_t bounds, __global float* tHit,
  __global int* index,  __global int* stack,
  int roffsetX, int xWidth, int yWidth,
  const int lwidth, const int lheight,
    int size, unsigned int offsetGID//, __write_only image2d_t kontrola
#ifdef STAT_RAY_TRIANGLE
 , __global unsigned int* stat_rayTriangle
#endif
) {
    // find position in global and shared arrays
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= size) return;
    int stackSize = 5*size;

    // find geometry for the work-item
    float4 v1, v2, v3;
    v1 = vload4(0, vertex + 9*iGID);
    v2 = vload4(0, vertex + 9*iGID + 3);
    v3 = vload4(0, vertex + 9*iGID + 6);
    v1.w = 0; v2.w = 0; v3.w = 0;
    float4 bmin,bmax;
    bmin = min(v1,v2);
    bmin = min(bmin,v3);
    bmax = max(v1,v2);
    bmax = max(bmax,v3);

    float4 omin, omax, dmin, dmax;

    int SPindex = 5*iGID;

    int j,k;
    for ( j = 0; j < yWidth; j++){
      for ( k = 0; k < xWidth; k++){
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

            dmax = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,             yWidth + j));
            omin = read_imagef(nodes, imageSampler, (int2)(roffsetX + xWidth + k, yWidth + j));
            dmin = read_imagef(nodes, imageSampler, (int2)(roffsetX + k,             j));
            omax.x = omin.w;
            omax.y = dmax.w;
            omax.z = dmin.w;

            if ( intersectsNode(omin, omax, dmin, dmax, bmin,bmax) )
            {
              //if it is a leaf node
              if ( roffsetX == 0) {
                intersectAllLeaves( dir, o, bounds, index, tHit, v1,v2,v3,v2 - v1,v3 - v1,
                      xWidth*lwidth, lheight, lwidth, k*lwidth, j*lheight , get_global_id(0)+offsetGID
                      #ifdef STAT_RAY_TRIANGLE
                      , stat_rayTriangle
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
