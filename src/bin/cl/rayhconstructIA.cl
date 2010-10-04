#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

__kernel void rayhconstruct(const __global float* dir,const  __global float* o,
  const __global unsigned int* counts, __global float* cones, const int count){
  int iGID = get_global_id(0);
  if (iGID >= count ) return;

  float4 omin, omax, dmin, dmax;
  float4 dtemp, otemp;

  unsigned int index = 0;
  for ( int i = 0; i < iGID; i++)
    index += counts[i];

  //start with the first ray
  dmin = dmax = vload4(0, dir + 3*index);
  omin = omax = vload4(0, o + 3*index);

  for ( int i = 1; i < counts[iGID]; i++){
    //check if the direction of the ray lies within the solid angle
    dtemp = vload4(0,dir+3*(index+i));
    otemp = vload4(0,o+3*(index+i));

    dmin.x = (dmin.x < dtemp.x)? dmin.x : dtemp.x;
    dmax.x = (dmax.x > dtemp.x)? dmax.x : dtemp.x;
    dmin.y = (dmin.y < dtemp.y)? dmin.y : dtemp.y;
    dmax.y = (dmax.y > dtemp.y)? dmax.y : dtemp.y;
    dmin.z = (dmin.z < dtemp.z)? dmin.z : dtemp.z;
    dmax.z = (dmax.z > dtemp.z)? dmax.z : dtemp.z;

    omin.x = (omin.x < otemp.x)? omin.x : otemp.x;
    omax.x = (omax.x > otemp.x)? omax.x : otemp.x;
    omin.y = (omin.y < otemp.y)? omin.y : otemp.y;
    omax.y = (omax.y > otemp.y)? omax.y : otemp.y;
    omin.z = (omin.z < otemp.z)? omin.z : otemp.z;
    omax.z = (omax.z > otemp.z)? omax.z : otemp.z;

  }

  //store the result
  vstore4(omin, 0, cones + 13*iGID);
  vstore4(omax, 0 ,cones + 13*iGID + 3);
  vstore4(dmin, 0, cones + 13*iGID + 6);
  dmax.w = counts[iGID];
  vstore4(dmax, 0, cones + 13*iGID + 9);
}
