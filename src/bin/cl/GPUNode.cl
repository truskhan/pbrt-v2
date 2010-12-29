typedef struct
{
  float ax, ay, az, bx, by, bz;
  int primOffset;
  int nPrimitives; //number of primitives, if it is interior node -> n = 0
} GPUNode;
