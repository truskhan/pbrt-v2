#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

__kernel void rayhconstruct(const __global float* dir,const  __global float* o,
  const __global unsigned int* counts, __global float* cones, const int count){
  int iGID = get_global_id(0);
  if (iGID >= count ) return;

  float4 x, a, r, orig;
  float2 uv;
  //center and radius of the sphere
  float4 center;
  float radius;
  //2D bounding box for uv
  float2 u, v;

  unsigned int index = 0;
  for ( int i = 0; i < iGID; i++)
    index += counts[i];

  //start with the zero angle enclosing cone of the first ray
  x = vload4(0, dir + 3*index);
  center = vload4(0, o + 3*index);
  x.w = 0; center.w = 0;
  radius = 0;
  u.x = u.y = atan2( x.y, x.z) ;
  v.x = v.y = acos(x.z);

  for ( int i = 1; i < counts[iGID]; i++){
    //check if the direction of the ray lies within the solid angle
    x = vload4(0,dir+3*(index+i));
    uv.x = atan2( x.y, x.z) ;
    uv.y = acos(x.z);
    //find 2D boundign box for uv
    u.x = (u.x < uv.x)? u.x : uv.x;
    u.y = (u.y > uv.x)? u.y : uv.x;
    v.x = (v.x < uv.y)? v.x : uv.y;
    v.y = (v.x > uv.y)? v.y : uv.y;

    //is ray origin inside the sphere?
    orig = vload4(0,o+3*(index+i));
    orig.w = 0;
    x.x = length(orig - center);
    //extend the radius
    radius = (x.x > (radius + EPS))? x.x: radius;

  }

  //store the result
  center.w = radius;
  vstore4(center, 0, cones + 9*iGID);
  vstore2(u, 0, cones + 9*iGID + 4);
  vstore2(v, 0, cones + 9*iGID + 6);
  cones[9*iGID + 8 ] = counts[iGID];
}
