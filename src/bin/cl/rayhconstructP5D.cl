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

  float4 center;
  float radius, dist;
  float4 uv;
  float4 uvtemp, otemp, tempCenter;
  int valid;

  //int width = get_image_width(dir);

  //start with the first ray
  uvtemp = read_imagef(dir, imageSampler, (int2)(lwidth*xGID, lheight*yGID));
  if ( uvtemp.w > 0 ) {
    valid = 1;
    center = read_imagef(o, imageSampler, (int2)(lwidth*xGID, lheight*yGID));
    uv.x = uv.y = (uvtemp.x == 0)? 0 : uvtemp.y/uvtemp.x;
    uv.z = uv.w = uvtemp.z;
  } else {
    valid = 0;
  }

  //read the tile
  for ( int i = 0; i < lheight; i++){
    for ( int j = 0; j < lwidth; j++) {
      uvtemp = read_imagef(dir, imageSampler, (int2)(lwidth*xGID + j, lheight*yGID + i));
      otemp = read_imagef(o, imageSampler, (int2)(lwidth*xGID + j, lheight*yGID + i));

      if ( uvtemp.w > 0 && valid == 0){
        center = otemp;
        uv.x = uv.y = (uvtemp.x == 0)? 0: uvtemp.y/uvtemp.x;
        uv.z = uv.w = uvtemp.z;
        valid = 1;
      } else {
        if ( uvtemp.w > 0){
          uv.x = min(uv.x, (uvtemp.x==0)?0:atan(uvtemp.y/uvtemp.x));
          uv.y = max(uv.y, (uvtemp.x==0)?0:atan(uvtemp.y/uvtemp.x));
          uv.z = min(uv.z, acos(uvtemp.z));
          uv.w = max(uv.w, acos(uvtemp.z));

          //is sphere inside the other one?
          otemp.w = 0;
          tempCenter = center - otemp;
          dist = length(tempCenter);
          if (dist  > ( radius + EPS)){
            //compute bounding sphere of the old ones
            center = center + (0.5f*(dist - radius)/dist)*tempCenter;
            //center1 = center1 + (0.5f*(radius2 + dist - radius1)/dist)*tempCenter;
            radius = ( radius + dist)/2;
            //radius1 = (radius1 + radius2 + dist)/2;
          }
        }
      }

    }
  }
  int4 wvalid = (int4)(valid,0,0,0);
  write_imagei (validity, (int2)(xGID, yGID),wvalid);
  if ( valid){
    center.w = radius;
    //store the result
    write_imagef(nodes, (int2)(xGID,                      yGID + globalHeight),uv);
    write_imagef(nodes, (int2)(xGID,                      yGID),center);
  }
}
