#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void rayhconstruct(__read_only image2d_t dir, __read_only image2d_t o,
  __write_only image2d_t nodes, const int globalWidth, const int globalHeight,
  const int lwidth, const int lheight){

  int xGID = get_global_id(0);
  int yGID = get_global_id(1);
  if ( xGID >= globalWidth || yGID >= globalHeight ) return;

  float4 center;
  float radius, dist;
  float4 uv;
  float4 uvtemp, otemp, tempCenter;

  //int width = get_image_width(dir);

  //start with the first ray
  uvtemp = read_imagef(dir, imageSampler, (int2)(lwidth*xGID, lheight*yGID));
  center = read_imagef(o, imageSampler, (int2)(lwidth*xGID, lheight*yGID));
  //radius
  center.w = 0;
  uv.x = uv.y = (uvtemp.x == 0)? 0 : atan(uvtemp.y/uvtemp.x);
  uv.z = uv.w = acos(uvtemp.z);

  //read the tile
  for ( int i = 0; i < lheight; i++){
    for ( int j = 0; j < lwidth; j++) {
      uvtemp = read_imagef(dir, imageSampler, (int2)(lwidth*xGID + j, lheight*yGID + i));
      otemp = read_imagef(o, imageSampler, (int2)(lwidth*xGID + j, lheight*yGID + i));

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

  center.w = radius;

  //store the result
  write_imagef(nodes, (int2)(xGID,                      yGID + globalHeight),uv);
  write_imagef(nodes, (int2)(xGID,                      yGID),center);
}
