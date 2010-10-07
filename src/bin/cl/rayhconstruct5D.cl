#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

__kernel void rayhconstruct(const __global float* dir,const  __global float* o,
  const __global unsigned int* counts, __global float* cones, const int count){
  int iGID = get_global_id(0);
  if (iGID >= count ) return;

  float4 x, a, orig;
  float2 uv;
  //center and radius of the sphere
  float4 center;
  float radius;
  //2D bounding box for uv
  float2 uvmin, uvmax;

  unsigned int index = 0;
  for ( int i = 0; i < iGID; i++)
    index += counts[i];

  //start with the zero angle enclosing cone of the first ray
  x = vload4(0, dir + 3*index);
  center = vload4(0, o + 3*index);
  x.w = 0; center.w = 0;
  radius = 0;
  uvmin.x = uvmax.x = (x.x == 0)? 0 :  atan(x.y / x.x) ;
  uvmin.y = uvmax.y = acos(x.z);

  for ( int i = 1; i < counts[iGID]; i++){
    //check if the direction of the ray lies within the solid angle
    x = vload4(0,dir+3*(index+i));
    uv.x = (x.x == 0)? 0 : atan(x.y/ x.x);
    uv.y = acos(x.z);
    //find 2D boundign box for uv
    uvmin = min(uvmin, uv);
    uvmax = max(uvmax, uv);

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
  vstore2(uvmin, 0, cones + 9*iGID + 4);
  vstore2(uvmax, 0, cones + 9*iGID + 6);
  cones[9*iGID + 8 ] = counts[iGID];
}
