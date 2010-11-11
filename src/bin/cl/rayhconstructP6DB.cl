#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void rayhconstructP(__read_only image2d_t dir, __read_only image2d_t o,
  __write_only image2d_t nodes, __write_only image2d_t validity,
  const int globalWidth, const int globalHeight,
  const int lwidth, const int lheight){

  int xGID = get_global_id(0);
  int yGID = get_global_id(1);
  if ( xGID >= globalWidth || yGID >= globalHeight ) return;

  float4 omin, omax, dmin, dmax;
  float4 dtemp, otemp;
  int valid;

  //start with the first ray
  dmin = dmax = read_imagef(dir, imageSampler, (int2)(lwidth*xGID, lheight*yGID));
  if ( dmax.w > 0 )
    valid = 1;
  else valid = 0;
  omin = omax = read_imagef(o, imageSampler, (int2)(lwidth*xGID, lheight*yGID));

  //read the tile
  for ( int i = 0; i < lheight; i++){
    for ( int j = 0; j < lwidth; j++) {
      dtemp = read_imagef(dir, imageSampler, (int2)(lwidth*xGID + j, lheight*yGID + i));
      otemp = read_imagef(o, imageSampler, (int2)(lwidth*xGID + j, lheight*yGID + i));

      if ( dtemp.w > 0 && valid == 0){
        dmin = dmax = dtemp;
        omin = omax = otemp;
        valid = 1;
      } else {
        if ( dtemp.w > 0){
          dmin = min(dmin, dtemp);
          dmax = max(dmax, dtemp);
          omin = min(omin, otemp);
          omax = max(omax, otemp);
       }
      }
    }
  }

  //store the result
  int4 wvalid = (int4)(valid,0,0,0);
  write_imagei (validity, (int2)(xGID, yGID),wvalid);
  if ( valid){
    omin.w = omax.x;
    dmax.w = omax.y;
    dmin.w = omax.z;
    write_imagef(nodes, (int2)(xGID,                      yGID + globalHeight),dmax);
    write_imagef(nodes, (int2)(xGID + globalWidth,        yGID + globalHeight),omin);
    write_imagef(nodes, (int2)(xGID,                      yGID),dmin);
  }
}
