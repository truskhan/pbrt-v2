#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

__kernel void rayhconstructP(const __global float* dir,const  __global float* o,
  const __global unsigned int* counts, __global float* cones, __global int* pointers, const int count){
  int iGID = get_global_id(0);
  if (iGID >= count ) return;

  float4 omin, omax, dmin, dmax;
  float4 dtemp, otemp;
  int2 child;
  child.x = -2;

  unsigned int index = 0;
  for ( int i = 0; i < iGID; i++)
    index += counts[i];

  //start with the first ray
  dmin = dmax = vload4(0, dir + 3*index);
  omin = omax = vload4(0, o + 3*index);
  dmin.w = dmax.w = omin.w = omax.w = 0;
  child.y = counts[iGID];

  for ( int i = 1; i < counts[iGID]; i++){
    //check if the direction of the ray lies within the solid angle
    dtemp = vload4(0,dir+3*(index+i));
    otemp = vload4(0,o+3*(index+i));
    dtemp.w = otemp.w = 0;

    dmin = min(dmin, dtemp);
    dmax = max(dmax, dtemp);
    omin = min(omin, otemp);
    omax = max(omax, otemp);
  }

  //store the result
  vstore4(omin, 0, cones + 13*iGID);
  vstore4(omax, 0 ,cones + 13*iGID + 3);
  vstore4(dmin, 0, cones + 13*iGID + 6);
  dmax.w = 2*iGID; //counts[iGID];
  vstore4(dmax, 0, cones + 13*iGID + 9);
  //store the ray grouped count and indicate that it is a list (-2)
  vstore2(child, 0, pointers + 2*iGID);
}
