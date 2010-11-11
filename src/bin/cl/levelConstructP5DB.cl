#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

//takes for nodes and merge them
//offsetX and offsetY says where to begin with storing
__kernel void levelConstructP(
   __write_only image2d_t wnodes, __write_only image2d_t wvalidity,
    __read_only image2d_t nodes, __read_only image2d_t validity,
  int roffsetX, int woffsetX, int width, int height )
{

  int xGID = get_global_id(0);
  int yGID = get_global_id(1);

  if ( xGID*2 >= width || yGID*2 >= height) return;

  float4 omin1, omax1, uv1;
  float4 omin2, omax2, uv2;
  int4 valid1, valid2;

  int posX, posY, offsetX, offsetY;

  valid1 = read_imagei(validity, imageSampler, (int2)(2*xGID + roffsetX,     2*yGID));
  if ( valid1.x == 1){
    omax1 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX,          2*yGID + height));
    uv1 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + width,  2*yGID + height));
    omin1 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX,          2*yGID ));
  }

  valid2 = read_imagei(validity, imageSampler, (int2)(2*xGID + roffsetX,     2*yGID + 1));
  if ( valid2.x == 1) {
    omax2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX,          2*yGID + height + 1));
    uv2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + width,  2*yGID + height + 1));
    omin2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX,          2*yGID + 1));
  }
  if ( valid1.x == 0 && valid2.x == 1){
    omax1 = omax2;
    uv1 = uv2;
    omin1 = omin2;
    valid1.x = 1; valid2.x = 0;
  }
  if ( valid1.x == 1 && valid2.x == 1){
    omin1 = min(omin1, omin2);
    omax1 = max(omax1, omax2);
    uv1.x = min(uv1.x, uv2.x);
    uv1.y = max(uv1.y, uv2.y);
    uv1.z = min(uv1.z, uv2.z);
    uv1.w = max(uv1.w, uv2.w);
  }

  valid2 = read_imagei(validity, imageSampler, (int2)(2*xGID + roffsetX + 1,     2*yGID ));
  if ( valid2.x == 1) {
    omax2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + 1,         2*yGID + height ));
    uv2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + width + 1, 2*yGID + height));
    omin2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + 1,         2*yGID ));
  }
  if ( valid1.x == 0 && valid2.x == 1){
    omax1 = omax2;
    uv1 = uv2;
    omin1 = omin2;
    valid1.x = 1; valid2.x = 0;
  }
  if ( valid1.x == 1 && valid2.x == 1){
    omin1 = min(omin1, omin2);
    omax1 = max(omax1, omax2);
    uv1.x = min(uv1.x, uv2.x);
    uv1.y = max(uv1.y, uv2.y);
    uv1.z = min(uv1.z, uv2.z);
    uv1.w = max(uv1.w, uv2.w);
  }

  valid2 = read_imagei(validity, imageSampler, (int2)(2*xGID + roffsetX + 1,     2*yGID + 1));
  if ( valid2.x == 1) {
    omax2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + 1,          2*yGID + height + 1));
    uv2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + width + 1,  2*yGID + height + 1));
    omin2 = read_imagef(nodes, imageSampler, (int2)(2*xGID + roffsetX + 1,          2*yGID + 1));
  }

  if ( valid1.x == 0 && valid2.x == 1){
    omax1 = omax2;
    uv1 = uv2;
    omin1 = omin2;
    valid1.x = 1; valid2.x = 0;
  }
  if ( valid1.x == 1 && valid2.x == 1){
    omin1 = min(omin1, omin2);
    omax1 = max(omax1, omax2);
    uv1.x = min(uv1.x, uv2.x);
    uv1.y = max(uv1.y, uv2.y);
    uv1.z = min(uv1.z, uv2.z);
    uv1.w = max(uv1.w, uv2.w);
  }

  write_imagei ( wvalidity, (int2)(woffsetX + xGID, yGID),valid1);
  if ( valid1.x == 1){
    write_imagef(wnodes, (int2)(woffsetX + xGID,            yGID + height/2 ),omax1);
    write_imagef(wnodes, (int2)(woffsetX + xGID + width/2,  yGID + height/2 ),uv1);
    write_imagef(wnodes, (int2)(woffsetX + xGID,            yGID),omin1);
  }

}
