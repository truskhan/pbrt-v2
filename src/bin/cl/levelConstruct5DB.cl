#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

__kernel void levelConstruct(__global float* cones, const int count,
  const int threadsCount, const int level){
  int iGID = get_global_id(0);

  int beginr = 0;
  int beginw = 0;
  int levelcount = threadsCount; //end of level0
  int temp;

  for ( int i = 0; i < level; i++){
      beginw += levelcount;
      temp = levelcount;
      levelcount = (levelcount+1)/2; //number of elements in level
  }
  beginr = beginw - temp;

  if ( iGID >= levelcount ) return;

  //3D bounding box of the origin
  float2 ox1, oy1, oz1, ox2, oy2, oz2;
  //2D bounding box for uv
  float2 u1, v1, u2, v2;

  ox1 = vload2(0, cones + 11*beginr + 22*iGID);
  oy1 = vload2(0, cones + 11*beginr + 22*iGID + 2);
  oz1 = vload2( 0, cones + 11*beginr + 22*iGID + 4);
  u1 = vload2( 0, cones + 11*beginr + 22*iGID + 6);
  v1 = vload2( 0, cones + 11*beginr + 22*iGID + 8);

  cones[11*beginw + 11*iGID + 10] = 1;
  if ( !( iGID == (levelcount - 1) && temp % 2 == 1 )) {
    //last thread only copies node1 to the output
    cones[11*beginw + 11*iGID + 10] = 2;

    ox2 = vload2(0, cones + 11*beginr + 22*iGID + 11);
    oy2 = vload2(0, cones + 11*beginr + 22*iGID + 13);
    oz2 = vload2( 0, cones + 11*beginr + 22*iGID + 15);
    u2 = vload2( 0, cones + 11*beginr + 22*iGID + 17);
    v2 = vload2( 0, cones + 11*beginr + 22*iGID + 19);

    //find 2D boundign box for uv
    u1.x = min(u1.x, u2.x);
    u1.y = max(u1.y, u2.y);
    v1.x = min(v1.x, v2.x);
    v1.y = max(v1.y, v2.y);

    //find 3D bounding box for ray origins
    ox1.x = min(ox1.x, ox2.x);
    ox2.y = max(ox1.y, ox2.y);
    oy1.x = min(oy1.x, oy2.x);
    oy2.y = max(oy1.y, oy2.y);
    oz1.x = min(oz1.x, oz2.x);
    oz2.y = max(oz1.y, oz2.y);

  }
  //store the result
  vstore2(ox1, 0, cones + 11*beginw + 11*iGID);
  vstore2(oy1, 0, cones + 11*beginw + 11*iGID + 2);
  vstore2(oz1, 0, cones + 11*beginw + 11*iGID + 4);
  vstore2(u1, 0, cones + 11*beginw + 11*iGID + 6);
  vstore2(v1, 0, cones + 11*beginw + 11*iGID + 8);

}
