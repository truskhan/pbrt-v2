#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

__kernel void rayhconstruct(const __global float* dir,const  __global float* o,
  const __global unsigned int* counts, __global float* cones, const int count){
  int iGID = get_global_id(0);
  if (iGID >= count ) return;

  float4 x, r, q, c, p , a, e, n , g;
  float cosfi, sinfi;
  float dotrx, dotcx, t ;

  unsigned int index = 0;
  for ( int i = 0; i < iGID; i++)
    index += counts[i];

  //start with the zero angle enclosing cone of the first ray
  x = vload4(0, dir + 3*index);
  a = vload4(0, o + 3*index);
  x.w = 0; a.w = 0;
  cosfi = 1;

  for ( int i = 1; i < counts[iGID]; i++){
    //check if the direction of the ray lies within the solid angle
    r = vload4(0,dir+3*(index+i));
    r.w = 0;
    p = vload4(0,o+3*(index+i));
    p.w = 0;
    dotrx = dot(r,x);
    if ( dotrx < cosfi ){
      //extend the cone
      q = normalize(dotrx*x - r);
      sinfi = (cosfi > (1-EPS))? 0:native_sin(acos(cosfi)); //precison problems
      e = normalize(x*cosfi + q*sinfi);
      x = normalize(e+r);

      cosfi = dot(x,r);
    }
    //check if the origin of the ray is within the wolume
    if ( length(p-a) > EPS){
      c = normalize(p - a);
      dotcx = dot(c,x);
      if ( dotcx < cosfi){
        q = (dotcx*x - c)/length(dotcx*x-x);
        sinfi = native_sin(acos(cosfi));
        e = x*cosfi + q*sinfi;
        n = x*cosfi - q*sinfi;
        g = c - dot(n,c)*n;
        t = (length(g)*length(g))/dot(e,g);
        a = a - t*e;
      }
    }
  }

  //store the result
  vstore4(a, 0, cones + 8*iGID);
  x.w = (cosfi > (1-EPS)) ? 0.003f: acos(cosfi); //precision problems
  vstore4(x, 0 ,cones + 8*iGID + 3);
  cones[8*iGID+7] = counts[iGID];

}
