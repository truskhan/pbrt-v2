#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

__kernel void levelConstructP(__global float* cones, __global int* pointers, const int count,
  const int threadsCount, const int level){
  int iGID = get_global_id(0);

  int beginr = 0;
  int beginw = 0;
  int levelcount = threadsCount; //end of level0
  int temp, help;

  for ( int i = 0; i < level; i++){
      beginw += levelcount;
      temp = levelcount;
      levelcount = (levelcount+1)/2; //number of elements in level
  }
  beginr = beginw - temp;
  int2 child;

  if ( iGID >= levelcount ) return;

  //3D bounding box of the origin
  float4 omin1, omax1, omin2, omax2;
  //2D bounding box for uv
  float4 uvmin1, uvmax1, uvmin2, uvmax2;

    omin1 = vload4(0, cones + 13*beginr + 26*iGID);
    omax1 = vload4(0, cones + 13*beginr + 26*iGID + 3);
    uvmin1 = vload4( 0, cones + 13*beginr + 26*iGID + 6);
    uvmax1 = vload4( 0, cones + 13*beginr + 26*iGID + 9);
    child.x = 13*beginr + 26*iGID;

  child.y = -1;

  if ( !(iGID == (levelcount - 1) && temp % 2 == 1) ) {
    //last thread only copies node1 to the output
      omin2 = vload4(0, cones + 13*beginr + 26*iGID + 13);
      omax2 = vload4(0, cones + 13*beginr + 26*iGID + 16);
      uvmin2 = vload4( 0, cones + 13*beginr + 26*iGID + 19);
      uvmax2 = vload4( 0, cones + 13*beginr + 26*iGID + 22);
      child.y = 13*beginr + 26*iGID + 13;

    //find 2D boundign box for uv
    uvmin1 = min(uvmin1, uvmin2);
    uvmax1 = max(uvmax1, uvmax2);

    //find 3D bounding box for ray origins
    omin1 = min(omin1, omin2);
    omax1 = max(omax1, omax2);

  }
  //store the result
  vstore4(omin1, 0, cones + 13*beginw + 13*iGID);
  vstore4(omax1, 0, cones + 13*beginw + 13*iGID + 3);
  vstore4(uvmin1, 0, cones + 13*beginw + 13*iGID + 6);
  vstore4(uvmax1, 0, cones + 13*beginw + 13*iGID + 9);
   //store index into pointers
  cones[13*beginw + 13*iGID + 12] = 2*beginw + 2*iGID;
  //store pointers to children nodes
  vstore2(child, 0, pointers + 2*beginw + 2*iGID);

}
