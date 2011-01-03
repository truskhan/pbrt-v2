/**
 * @file GOUNode.cl
 * @author: Hana Truskova hana.truskova@seznam.cz
 * class for holding BVH node
**/
typedef struct
{
  /** bounding box size */
  float ax, ay, az, bx, by, bz;
  /** index into primitives array **/
  unsigned int primOffset;
  /** number of primitives, if it is interior node -> n = 0 **/
  unsigned int nPrimitives;
} GPUNode;
