#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

void intersectPAllLeaves (const __global float* dir, const __global float* o, const __global float* bounds,
__global unsigned char* tHit, float4 v1, float4 v2, float4 v3, float4 e1, float4 e2, int chunk, int rindex
#ifdef STAT_PRAY_TRIANGLE
 ,__global int* stat_rayTriangle
#endif
){
    float4 s1, s2, d, rayd, rayo;
    float divisor, invDivisor, t, b1, b2;
    // process all rays in the cone
    for ( int i = 0; i < chunk; i++){
      #ifdef STAT_PRAY_TRIANGLE
       ++stat_rayTriangle[rindex + i];
      #endif

      rayd = vload4(0, dir + 3*rindex + 3*i);
      rayo = vload4(0, o + 3*rindex + 3*i);
      rayd.w = 0; rayo.w = 0;
      s1 = cross(rayd, e2);
      divisor = dot(s1, e1);
      if ( divisor == 0.0f) continue;
      invDivisor = 1.0f/ divisor;

      // compute first barycentric coordinate
      d = rayo - v1;
      b1 = dot(d, s1) * invDivisor;
      if ( b1 < -1e-3f  || b1 > 1+1e-3f) continue;

      // compute second barycentric coordinate
      s2 = cross(d, e1);
      b2 = dot(rayd, s2) * invDivisor;
      if ( b2 < -1e-3f || (b1 + b2) > 1+1e-3f) continue;

      // Compute _t_ to intersection point
      t = dot(e2, s2) * invDivisor;
      if (t < bounds[2*rindex + i*2]) continue;

      tHit[rindex+i] = '1';
    }
}
int computeChild (unsigned int threadsCount, int i){
  int index = 0;
  int levelcount = threadsCount;
  int temp;

  if ( i < 13*levelcount)
    return -1; // level 0, check rays

  while ( (index + 13*levelcount) <= i){
    temp = levelcount;
    index += 13*levelcount;
    levelcount = (levelcount+1)/2;
  }
  int offset = i - index;

  return (index - 13*temp) + 2*offset;
}

int computeRIndex (unsigned int j, const __global float* cones){
  int rindex = 0;
  for ( int i = 0; i < j; i += 13){
      rindex += cones[i + 12];
  }
  return rindex;
}

bool intersectsNode ( float4 bmin, float4 bmax, float4 omin, float4 omax, float4 dmin, float4 dmax){
  float2 tx, ty, tz;
  omin = bmin - omin;
  omax = bmax - omax;
  //if d interval includes 0
  if ( dmin.x <= 0 && dmax.x >= 0){
    dmin.x = -MAXFLOAT;
    dmax.x = MAXFLOAT;
  } else {
    dmin.x = 1/dmax.x;
    dmax.x = 1/dmin.x;
  }
  if ( dmin.y <= 0 && dmax.y >= 0){
    dmin.y = -MAXFLOAT;
    dmax.y = MAXFLOAT;
  } else {
    dmin.y = 1/dmax.y;
    dmax.y = 1/dmin.y;
  }
  if ( dmin.z <= 0 && dmax.z >= 0){
    dmin.z = -MAXFLOAT;
    dmax.z = MAXFLOAT;
  } else {
    dmin.z = 1/dmax.z;
    dmax.z = 1/dmin.z;
  }

  float2 temp;
  temp.x = omin.x * dmin.x;
  temp.y = omax.x * dmax.x;
  tx.x = (temp.x < temp.y)? temp.x : temp.y;
  tx.y = (temp.x > temp.y)? temp.x : temp.y;

  temp.x = omin.y * dmin.y;
  temp.y = omax.y * dmax.y;
  ty.x = (temp.x < temp.y)? temp.x : temp.y;
  ty.y = (temp.x > temp.y)? temp.x : temp.y;

  temp.x = omin.z * dmin.z;
  temp.y = omax.z * dmax.z;
  tz.x = (temp.x < temp.y)? temp.x : temp.y;
  tz.y = (temp.x > temp.y)? temp.x : temp.y;

  //find interval intersections
  tx.x = (tx.x > ty.x)? tx.x : ty.x;
  tx.x = (tx.x > tz.x)? tx.x : tz.x;
  tx.y = (tx.y < ty.y)? tx.y : ty.y;
  tx.y = (tx.y < tz.y)? tx.y : tz.y;

  return (tx.x < tx.y);
}

__kernel void IntersectionP (
const __global float* vertex, const __global float* dir, const __global float* o,
 const __global float* cones, const __global float* bounds,
__global unsigned char* tHit,
#ifdef STAT_PRAY_TRIANGLE
 __global int* stat_rayTriangle,
#endif
__local int* stack, int count, int size, int height,unsigned int threadsCount
)
{
    int iGID = get_global_id(0);
    int iLID = get_local_id(0);
    if (iGID >= size) return;

    // process all geometry
    float4 e1, e2;

    float4 v1, v2, v3;
    v1 = vload4(0, vertex + 9*iGID);
    v2 = vload4(0, vertex + 9*iGID + 3);
    v3 = vload4(0, vertex + 9*iGID + 6);
    v1.w = 0; v2.w = 0; v3.w = 0;
    e1 = v2 - v1;
    e2 = v3 - v1;

    //calculate bounding box
    float4 bmin, bmax;
    bmin.x = (v1.x < v2.x)? v1.x : v2.x;
    bmin.x = (bmin.x < v3.x)? bmin.x : v3.x;
    bmax.x = (v1.x > v2.x)? v1.x : v2.x;
    bmax.x = (bmax.x > v3.x)? bmax.x : v3.x;

    bmin.y = (v1.y < v2.y)? v1.y : v2.y;
    bmin.y = (bmin.y < v3.y)? bmin.y : v3.y;
    bmax.y = (v1.y > v2.y)? v1.y : v2.y;
    bmax.y = (bmax.y > v3.y)? bmax.y : v3.y;

    bmin.z = (v1.z < v2.z)? v1.z : v2.z;
    bmin.z = (bmin.z < v3.z)? bmin.z : v3.z;
    bmax.z = (v1.z > v2.z)? v1.z : v2.z;
    bmax.z = (bmax.z > v3.z)? bmax.z : v3.z;

    float4 omin, omax, dmin, dmax;

    //find number of elements in top level of the ray hieararchy
    uint levelcount = threadsCount; //end of level0
    uint num = 0;
    uint lastlevelnum = 0;

    for ( int i = 1; i < height; i++){
        lastlevelnum = levelcount;
        num += levelcount;
        levelcount = (levelcount+1)/2; //number of elements in level
    }

    int SPindex = 0;
    uint begin, rindex;
    int i = 0;
    int child;

    begin = 13*num;
    for ( int j = 0; j < levelcount; j++){
      // get IA description
      omin = vload4(0, cones + begin+13*j);
      omax = vload4(0, cones + begin+13*j + 3);
      dmin = vload4(0, cones + begin+13*j + 6);
      dmax = vload4(0, cones + begin+13*j + 9);

      // check if triangle intersects IA node
      if ( intersectsNode(bmin, bmax, omin, omax, dmin, dmax) )
      {
        //store child to the stack
        stack[iLID*(height) + SPindex++] = begin - 13*lastlevelnum + 26*j;
        while ( SPindex > 0 ){
          //take the cones from the stack and check them
          --SPindex;
          i = stack[iLID*(height) + SPindex];
          omin = vload4(0, cones + i);
          omax = vload4(0, cones + i + 3);
          dmin = vload4(0, cones + i + 6);
          dmax = vload4(0, cones + i + 9);

          if ( intersectsNode(bmin, bmax, omin, omax, dmin, dmax))
          {
            child = computeChild(threadsCount,i);
            //if the cones is at level 0 - check leaves
            if ( child < 0) {
              rindex = computeRIndex(i, cones);
              intersectPAllLeaves( dir, o, bounds, tHit, v1,v2,v3,e1,e2,cones[i+12], rindex
              #ifdef STAT_PRAY_TRIANGLE
                , stat_rayTriangle
              #endif
              );
            }
            else {
              //save the intersected cone to the stack
              stack[iLID*(height) + SPindex++] = child;
            }
          }
          omin = vload4(0, cones + i + 13);
          omax = vload4(0, cones + i + 16);
          dmin = vload4(0, cones + i + 19);
          dmax = vload4(0, cones + i + 22);

          if ( intersectsNode(bmin, bmax, omin, omax, dmin, dmax))
         {
            child = computeChild (threadsCount, i+13);
            //if the cone is at level 0 - check leaves
            if ( child < 0) {
              rindex = computeRIndex(i + 13, cones);
              intersectPAllLeaves( dir, o, bounds, tHit, v1,v2,v3,e1,e2,cones[i+25],rindex
              #ifdef STAT_PRAY_TRIANGLE
                , stat_rayTriangle
              #endif
              );
            }
            else {
              stack[iLID*(height) + SPindex++] = child;
            }
          }
        }
      }

    }


}
