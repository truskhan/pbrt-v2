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
  dmax1.w = 1;

  if ( !( iGID == (levelcount - 1) && temp % 2 == 1 )) {
    //posledni vlakno jen prekopiruje
    dmax1.w = 2;
    omin2 = vload4(0,cones + 13*beginr + 26*iGID + 13);
    omax2 = vload4(0,cones + 13*beginr + 26*iGID + 16);
    dmin2 = vload4(0,cones + 13*beginr + 26*iGID + 19);
    dmax2 = vload4(0,cones + 13*beginr + 26*iGID + 22);

    omin1.x = (omin1.x < omin2.x)? omin1.x : omin2.x;
    omin1.y = (omin1.y < omin2.y)? omin1.y : omin2.y;
    omin1.z = (omin1.z < omin2.z)? omin1.z : omin2.z;

    omax1.x = (omax1.x > omax2.x)? omax1.x : omax2.x;
    omax1.y = (omax1.y > omax2.y)? omax1.y : omax2.y;
    omax1.z = (omax1.z > omax2.z)? omax1.z : omax2.z;

    dmin1.x = (dmin1.x < dmin2.x)? dmin1.x : dmin2.x;
    dmin1.y = (dmin1.y < dmin2.y)? dmin1.y : dmin2.y;
    dmin1.z = (dmin1.z < dmin2.z)? dmin1.z : dmin2.z;

    dmax1.x = (dmax1.x > dmax2.x)? dmax1.x : dmax2.x;
    dmax1.y = (dmax1.y > dmax2.y)? dmax1.y : dmax2.y;
    dmax1.z = (dmax1.z > dmax2.z)? dmax1.z : dmax2.z;
  }

  vstore4(omin1, 0, cones + 13*beginw + 13*iGID);
  vstore4(omax1, 0, cones + 13*beginw + 13*iGID + 3);
  vstore4(dmin1, 0, cones + 13*beginw + 13*iGID + 6);
  vstore4(dmax1, 0, cones + 13*beginw + 13*iGID + 9);


}
