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
  float4 omin1, omax1, omin2, omax2;
  //2D bounding box for uv
  float2 uvmin1, uvmax1, uvmin2, uvmax2;

  omin1 = vload4(0, cones + 11*beginr + 22*iGID);
  omax1 = vload4(0, cones + 11*beginr + 22*iGID + 3);
  uvmin1 = vload2( 0, cones + 11*beginr + 22*iGID + 6);
  uvmax1 = vload2( 0, cones + 11*beginr + 22*iGID + 8);

  cones[11*beginw + 11*iGID + 10] = 1;
  if ( !( iGID == (levelcount - 1) && temp % 2 == 1 )) {
    //last thread only copies node1 to the output
    cones[11*beginw + 11*iGID + 10] = 2;

    omin2 = vload4(0, cones + 11*beginr + 22*iGID + 11);
    omax2 = vload4(0, cones + 11*beginr + 22*iGID + 14);
    uvmin2 = vload2( 0, cones + 11*beginr + 22*iGID + 17);
    uvmax2 = vload2( 0, cones + 11*beginr + 22*iGID + 19);

    //find 2D boundign box for uv
    uvmin1 = min(uvmin1, uvmin2);
    uvmax1 = max(uvmax1, uvmax2);

    //find 3D bounding box for ray origins
    omin1 = min(omin1, omin2);
    omax1 = max(omax1, omax2);

  }
  //store the result
  vstore4(omin1, 0, cones + 11*beginw + 11*iGID);
  vstore4(omax1, 0, cones + 11*beginw + 11*iGID + 3);
  vstore2(uvmin1, 0, cones + 11*beginw + 11*iGID + 6);
  vstore2(uvmax1, 0, cones + 11*beginw + 11*iGID + 8);

}
