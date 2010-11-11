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
  //2D bounding box for uv
  float2 uvmin, uvmax;
  int2 child;
  child.x = -2;

  unsigned int index = 0;
  for ( int i = 0; i < iGID; i++)
    index += counts[i];

  //load ray dir
  x = vload4(0, dir + 3*index);
  uvmin.x = uvmax.x = (x.x == 0) ? 0 : x.y/x.x;
  uvmin.y = uvmax.y = x.z;

  //load ray origin
  x = vload4(0, o + 3*index);
  omin.x = omax.x = x.x;
  omin.y = omax.y = x.y;
  omin.z = omax.z = x.z;
  omin.w = omax.w = 0;
  child.y = counts[iGID];

  for ( int i = 1; i < counts[iGID]; i++){
    x = vload4(0,dir+3*(index+i));
    uv.x = ( x.x == 0)? 0 : x.y/x.x;
    uv.y = (x.z);
    //find 2D boundign box for uv
    uvmin = min(uvmin, uv);
    uvmax = max(uvmax, uv);

    //find box for the origin of rays
    x = vload4(0,o+3*(index+i));
    x.w = 0;
    omin = min(omin, x);
    omax = max(omax, x);

  }

  //store the result
  vstore4(omin, 0, cones + 11*iGID);
  vstore4(omax, 0, cones + 11*iGID + 3);
  vstore2(uvmin, 0, cones + 11*iGID + 6);
  vstore2(uvmax, 0, cones + 11*iGID + 8);
  cones[11*iGID + 10 ] = 2*iGID;
  //store the ray grouped count and indicate that it is a list (-2)
  vstore2(child, 0, pointers + 2*iGID);
}
