#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void rayhconstruct(__read_only image2d_t dir, __read_only image2d_t o,
  __write_only image2d_t nodes, const int globalWidth, const int globalHeight,
  const int lheight, const int lwidth){

  int xGID = get_global_id(0);
  int yGID = get_global_id(1);
  if ( xGID > globalWidth || yGID > globalHeight ) return;

  float4 omin, omax, dmin, dmax;
  float4 dtemp, otemp;

  //int width = get_image_width(dir);

  //start with the first ray
  dmin = dmax = read_imagef(dir, imageSampler, (int2)(lwidth*xGID, lheight*yGID));
  omin = omax = read_imagef(o, imageSampler, (int2)(lwidth*xGID, lheight*yGID));

  //read the tile
  for ( int i = 0; i < lheight; i++){
    for ( int j = 0; j < lwidth; j++) {
      dtemp = read_imagef(dir, imageSampler, (int2)(lwidth*xGID + j, lheight*yGID + i));
      otemp = read_imagef(o, imageSampler, (int2)(lwidth*xGID + j, lheight*yGID + i));

      dmin = min(dmin, dtemp);
      dmax = max(dmax, dtemp);
      omin = min(omin, otemp);
      omax = max(omax, otemp);
    }
  }

  //store the result
  omin.w = omax.x;
  dmax.w = omax.y;
  dmin.w = omax.z;
  write_imagef(nodes, (int2)(xGID,                      yGID + get_global_size(1)),dmax);
  write_imagef(nodes, (int2)(xGID + get_global_size(0), yGID + get_global_size(1)),omin);
  write_imagef(nodes, (int2)(xGID,                      yGID),dmin);
}
