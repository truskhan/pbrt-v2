#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

__kernel void levelConstructP(__global float* cones, __global int* pointers, const int count,
  const int threadsCount, const int level ){
  int iGID = get_global_id(0);

  int beginr = 0;
  int beginw = 0;
  int levelcount = threadsCount; //end of level0
  int temp;
  int help;

  for ( int i = 0; i < level; i++){
      beginw += levelcount;
      temp = levelcount;
      levelcount = (levelcount+1)/2; //number of elements in level
  }
  beginr = beginw - temp;

  if ( iGID >= levelcount ) return;

  float4 omin1, omax1, dmin1, dmax1;
  float4 omin2, omax2, dmin2, dmax2;
  int2 child;

  omin1 = vload4(0,cones + 13*beginr + 26*iGID);
  omax1 = vload4(0,cones + 13*beginr + 26*iGID + 3);
  dmin1 = vload4(0,cones + 13*beginr + 26*iGID + 6);
  dmax1 = vload4(0,cones + 13*beginr + 26*iGID + 9);
  child.x = 13*beginr + 26*iGID;

  omin1.w = omax1.w = dmin1.w = 0;
  child.y = -1;

  if ( !(iGID == (levelcount - 1) && temp % 2 == 1) ) {
    //posledni vlakno jen prekopiruje
    omin2 = vload4(0,cones + 13*beginr + 26*iGID + 13);
    omax2 = vload4(0,cones + 13*beginr + 26*iGID + 16);
    dmin2 = vload4(0,cones + 13*beginr + 26*iGID + 19);
    dmax2 = vload4(0,cones + 13*beginr + 26*iGID + 22);
    child.y = 13*beginr + 26*iGID + 13;

    omin2.w = omax2.w = dmin2.w = dmax2.w = 0;


    omin1 = min(omin1, omin2);
    omax1 = max(omax1, omax2);
    dmin1 = min(dmin1, dmin2);
    dmax1 = max(dmax1, dmax2);

  }

  dmax1.w = 2*beginw + 2*iGID;
  vstore4(omin1, 0, cones + 13*beginw + 13*iGID);
  vstore4(omax1, 0, cones + 13*beginw + 13*iGID + 3);
  vstore4(dmin1, 0, cones + 13*beginw + 13*iGID + 6);
  //store index into pointers -  dmax1.w
  vstore4(dmax1, 0, cones + 13*beginw + 13*iGID + 9);
  //store pointers to children nodes
  vstore2(child, 0, pointers + 2*beginw + 2*iGID);

}
