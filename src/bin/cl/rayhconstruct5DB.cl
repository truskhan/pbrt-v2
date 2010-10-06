#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

__kernel void rayhconstruct(const __global float* dir,const  __global float* o,
  const __global unsigned int* counts, __global float* cones, const int count){
  int iGID = get_global_id(0);
  if (iGID >= count ) return;

  float4 x;
  float2 uv;
  //3D bounding box of the origin
  float2 ox, oy, oz;
  //2D bounding box for uv
  float2 u, v;

  unsigned int index = 0;
  for ( int i = 0; i < iGID; i++)
    index += counts[i];

  //load ray dir
  x = vload4(0, dir + 3*index);
  u.x = u.y = ( x.x == 0) ? 0:x.y / x.x;
  v.x = v.y = (x.z);

  //load ray origin
  x = vload4(0, o + 3*index);
  ox.x = ox.y = x.x;
  oy.x = oy.y = x.y;
  oz.x = oz.y = x.z;

  for ( int i = 1; i < counts[iGID]; i++){
    x = vload4(0,dir+3*(index+i));
    uv.x = ( x.x == 0)? 0 : x.y/x.x;
    uv.y = (x.z);
    //find 2D boundign box for uv
    u.x = min(u.x, uv.x);
    u.y = max(u.y, uv.x);
    v.x = min(v.x, uv.y);
    v.y = max(v.y, uv.y);

    //find box for the origin of rays
    x = vload4(0,o+3*(index+i));
    ox.x = min(ox.x, x.x);
    ox.y = max(ox.y, x.x);
    oy.x = min(oy.x, x.y);
    oy.y = max(oy.y, x.y);
    oz.x = min(oz.x, x.z);
    oz.y = max(oz.y, x.z);

  }

  //store the result
  vstore2(ox, 0, cones + 11*iGID);
  vstore2(oy, 0, cones + 11*iGID + 2);
  vstore2(oz, 0, cones + 11*iGID + 4);
  vstore2(u, 0, cones + 11*iGID + 6);
  vstore2(v, 0, cones + 11*iGID + 8);
  cones[11*iGID + 10 ] = counts[iGID];
}
