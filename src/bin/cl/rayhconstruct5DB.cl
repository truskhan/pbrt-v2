#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void rayhconstruct(__read_only image2d_t dir, __read_only image2d_t o,
  __write_only image2d_t nodes, const int globalWidth, const int globalHeight,
  const int lwidth, const int lheight){

  int xGID = get_global_id(0);
  int yGID = get_global_id(1);
  if ( xGID >= globalWidth || yGID >= globalHeight ) return;

  float4 omin, omax;
  float4 uv;
  float4 uvtemp, otemp;

  //int width = get_image_width(dir);

  //start with the first ray
  uvtemp = read_imagef(dir, imageSampler, (int2)(lwidth*xGID, lheight*yGID));
  omin = omax = read_imagef(o, imageSampler, (int2)(lwidth*xGID, lheight*yGID));
  uv.x = uv.y = (uvtemp.x == 0)? 0 : uvtemp.y/uvtemp.x;
  uv.z = uv.w = uvtemp.z;

  //read the tile
  for ( int i = 0; i < lheight; i++){
    for ( int j = 0; j < lwidth; j++) {
      uvtemp = read_imagef(dir, imageSampler, (int2)(lwidth*xGID + j, lheight*yGID + i));
      otemp = read_imagef(o, imageSampler, (int2)(lwidth*xGID + j, lheight*yGID + i));

      uv.x = min(uv.x, (uvtemp.x==0)?0:uvtemp.y/uvtemp.x);
      uv.y = max(uv.y, (uvtemp.x==0)?0:uvtemp.y/uvtemp.x);
      uv.z = min(uv.z, uvtemp.z);
      uv.w = max(uv.w, uvtemp.z);

      omin = min(omin, otemp);
      omax = max(omax, otemp);
    }
  }

  //store the result
  write_imagef(nodes, (int2)(xGID,                      yGID + globalHeight),omax);
  write_imagef(nodes, (int2)(xGID + globalWidth,        yGID + globalHeight),uv);
  write_imagef(nodes, (int2)(xGID,                      yGID),omin);
}
