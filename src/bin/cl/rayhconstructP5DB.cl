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

  float4 omin, omax;
  float4 uv;
  float4 uvtemp, otemp;
  int valid;

  //start with the first ray
  uvtemp = read_imagef(dir, imageSampler, (int2)(lwidth*xGID, lheight*yGID));
  omin = omax = read_imagef(o, imageSampler, (int2)(lwidth*xGID, lheight*yGID));
  if ( uvtemp.w > 0 ) {
    valid = 1;
    if ( uvtemp.x == 0){
      uv.x = -1; uv.y = 1;
    } else {
    uv.x = uv.y = uvtemp.y/uvtemp.x;
    }
    uv.z = uv.w = uvtemp.z;
  }else {
    valid = 0;
  }

  //read the tile
  for ( int i = 0; i < lheight; i++){
    for ( int j = 0; j < lwidth; j++) {
      uvtemp = read_imagef(dir, imageSampler, (int2)(lwidth*xGID + j, lheight*yGID + i));
      otemp = read_imagef(o, imageSampler, (int2)(lwidth*xGID + j, lheight*yGID + i));

      if ( uvtemp.w > 0 && valid == 0){
        omin = omax = otemp;
        if ( uvtemp.x == 0){
          uv.x = -1; uv.y = 1;
        } else {
        uv.x = uv.y = uvtemp.y/uvtemp.x;
        }
        uv.z = uv.w = uvtemp.z;
        valid = 1;
      } else {
        if ( uvtemp.w > 0){
          if ( uvtemp.x == 0){
            uv.x = -1; uv.y = 1;
          } else {
          uv.x = min(uv.x, uvtemp.y/uvtemp.x);
          uv.y = max(uv.y, uvtemp.y/uvtemp.x);
          }
          uv.z = min(uv.z, uvtemp.z);
          uv.w = max(uv.w, uvtemp.z);

          omin = min(omin, otemp);
          omax = max(omax, otemp);
        }
      }

    }
  }

  int4 wvalid = (int4)(valid,0,0,0);
  write_imagei (validity, (int2)(xGID, yGID),wvalid);
  if ( valid){
    //store the result
    write_imagef(nodes, (int2)(xGID,                      yGID + globalHeight),omax);
    write_imagef(nodes, (int2)(xGID + globalWidth,        yGID + globalHeight),uv);
    write_imagef(nodes, (int2)(xGID,                      yGID),omin);
  }
}
