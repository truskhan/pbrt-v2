#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

__kernel void rayhconstructP(const __global float* dir,const  __global float* o,
  const __global unsigned int* counts, __global float* cones, __global int* pointers, const int count){
  int iGID = get_global_id(0);
  if (iGID >= count ) return;

  float4 x;
  float2 uv;
  //3D bounding box of the origin
  float4 omin, omax;
  //3D bounding box for ray directions
  float4 dmin, dmax;
  int2 child;
  child.x = -2;

  unsigned int index = 0;
  for ( int i = 0; i < iGID; i++)
    index += counts[i];

  //load ray dir
  x = vload4(0, dir + 3*index);
  dmin.x = dmax.x = x.x;
  dmin.y = dmax.y = x.y;
  dmin.z = dmax.z = x.z;
  dmin.w = dmax.w = 0;

  //load ray origin
  x = vload4(0, o + 3*index);
  omin.x = omax.x = x.x;
  omin.y = omax.y = x.y;
  omin.z = omax.z = x.z;
  omin.w = omax.w = 0;
  child.y = counts[iGID];

  for ( int i = 1; i < counts[iGID]; i++){
    x = vload4(0,dir+3*(index+i));
    x.w = 0;
    dmin = min(dmin, x);
    dmax = max(dmax, x);

    //find box for the origin of rays
    x = vload4(0,o+3*(index+i));
    x.w = 0;
    omin = min(omin, x);
    omax = max(omax, x);

  }

  //store the result
  vstore4(omin, 0, cones + 13*iGID);
  vstore4(omax, 0, cones + 13*iGID + 3);
  vstore4(dmin, 0, cones + 13*iGID + 6);
  vstore4(dmax, 0, cones + 13*iGID + 9);
  cones[13*iGID + 12 ] = 2*iGID;
  //store the ray grouped count and indicate that it is a list (-2)
  vstore2(child, 0, pointers + 2*iGID);
}
