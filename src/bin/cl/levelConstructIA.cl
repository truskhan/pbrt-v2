#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

//takes for nodes and merge them
//offsetX and offsetY says where to begin with storing
__kernel void levelConstruct(
   __write_only image2d_t wnodes, __read_only image2d_t nodes,
  int roffsetX, int woffsetX, int width, int height )
{

  int xGID = get_global_id(0);
  int yGID = get_global_id(1);

  if ( xGID*2 >= width || yGID*2 >= height) return;

  float4 omin1, omax1, dmin1, dmax1;
  float4 omin2, omax2, dmin2, dmax2;
  int2 child;

  int posX, posY, offsetX, offsetY;

  dmax1 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX,          2*yGID + height));
  omin1 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + width,  2*yGID + height));
  dmin1 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX,          2*yGID ));
  omax1.x = omin1.w;
  omax1.y = dmax1.w;
  omax1.z = dmin1.w;

  dmax2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX,          2*yGID + height + 1));
  omin2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + width,  2*yGID + height + 1));
  dmin2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX,          2*yGID + 1));
  omax2.x = omin2.w;
  omax2.y = dmax2.w;
  omax2.z = dmin2.w;

  omin1 = min(omin1, omin2);
  omax1 = max(omax1, omax2);
  dmin1 = min(dmin1, dmin2);
  dmax1 = max(dmax1, dmax2);

  dmax2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + 1,         2*yGID + height ));
  omin2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + width + 1, 2*yGID + height));
  dmin2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + 1,         2*yGID ));
  omax2.x = omin2.w;
  omax2.y = dmax2.w;
  omax2.z = dmin2.w;

  omin1 = min(omin1, omin2);
  omax1 = max(omax1, omax2);
  dmin1 = min(dmin1, dmin2);
  dmax1 = max(dmax1, dmax2);

  dmax2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + 1,          2*yGID + height + 1));
  omin2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + width + 1,  2*yGID + height + 1));
  dmin2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + 1,          2*yGID + 1));
  omax2.x = omin2.w;
  omax2.y = dmax2.w;
  omax2.z = dmin2.w;

  omin1 = min(omin1, omin2);
  omax1 = max(omax1, omax2);
  dmin1 = min(dmin1, dmin2);
  dmax1 = max(dmax1, dmax2);

  omin1.w = omax1.x;
  dmax1.w = omax1.y;
  dmin1.w = omax1.z;

  write_imagef(wnodes, (int2)(woffsetX + xGID,            yGID + height/2 ),dmax1);
  write_imagef(wnodes, (int2)(woffsetX + xGID + width/2,  yGID + height/2 ),omin1);
  write_imagef(wnodes, (int2)(woffsetX + xGID,            yGID),dmin1);

}
