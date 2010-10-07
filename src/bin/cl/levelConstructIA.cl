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

  float4 omin1, omax1, dmin1, dmax1;
  float4 omin2, omax2, dmin2, dmax2;

  omin1 = vload4(0,cones + 13*beginr + 26*iGID);
  omax1 = vload4(0,cones + 13*beginr + 26*iGID + 3);
  dmin1 = vload4(0,cones + 13*beginr + 26*iGID + 6);
  dmax1 = vload4(0,cones + 13*beginr + 26*iGID + 9);
  omin1.w = omax1.w = dmin1.w = 0;
  dmax1.w = 1;

  if ( !( iGID == (levelcount - 1) && temp % 2 == 1 )) {
    //posledni vlakno jen prekopiruje
    omin2 = vload4(0,cones + 13*beginr + 26*iGID + 13);
    omax2 = vload4(0,cones + 13*beginr + 26*iGID + 16);
    dmin2 = vload4(0,cones + 13*beginr + 26*iGID + 19);
    dmax2 = vload4(0,cones + 13*beginr + 26*iGID + 22);
    omin2.w = omax2.w = dmin2.w = dmax2.w = 0;

    omin1 = min(omin1, omin2);
    omax1 = max(omax1, omax2);
    dmin1 = min(dmin1, dmin2);
    dmax1 = max(dmax1, dmax2);

    dmax1.w = 2;
  }

  vstore4(omin1, 0, cones + 13*beginw + 13*iGID);
  vstore4(omax1, 0, cones + 13*beginw + 13*iGID + 3);
  vstore4(dmin1, 0, cones + 13*beginw + 13*iGID + 6);
  vstore4(dmax1, 0, cones + 13*beginw + 13*iGID + 9);


}
