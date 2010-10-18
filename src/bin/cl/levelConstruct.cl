#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

__kernel void levelConstruct(__global float* cones, __global int* pointers, const int count,
  const int threadsCount, const int level, const int xwidth){
  int iGID = get_global_id(0);

  int beginr = 0;
  int beginw = 0;
  int levelcount = threadsCount; //end of level0
  int temp, help;

  for ( int i = 0; i < level; i++){
      beginw += levelcount;
      temp = levelcount;
      levelcount = (levelcount+1)/2; //number of elements in level
  }
  beginr = beginw - temp;

  if ( iGID >= levelcount ) return;
  int2 child;

  float4 x, q, c, a, g, xb;
  float4 ab,e,n, ra, rb;
  float cosfi, sinfi, cosfib;
  float dotrx, dotcx, t ;
  float fi,fib;

  if ( level & 0x1 == 1){
    help = iGID / xwidth;
    a = vload4(0,cones + 8*beginr + 8*iGID + 8*help*xwidth);
    x = vload4(0,cones + 8*beginr + 8*iGID + 8*help*xwidth + 3);
    child.x = 8*beginr + 8*iGID + 8*help*xwidth;
  } else {
    a = vload4(0,cones + 8*beginr + 16*iGID);
    x = vload4(0,cones + 8*beginr + 16*iGID + 3);
    child.x = 8*beginr + 16*iGID;
  }
  child.y = -1;
  fi = x.w;
  a.w = 0; x.w = 0;
  cosfi = native_cos(fi);

  if ( !( iGID == (levelcount - 1) && temp % 2 == 1 )) {
    //posledni vlakno jen prekopiruje
    if ( level & 0x1 == 1){
      ++help;
      ab = vload4(0,cones + 8*beginr + 8*iGID + 8*help*xwidth);
      xb = vload4(0,cones + 8*beginr + 8*iGID + 8*help*xwidth + 3);
      child.y = 8*beginr + 8*iGID + 8*help*xwidth;
    }else {
      ab = vload4(0,cones + 8*beginr + 16*iGID + 8);
      xb = vload4(0,cones + 8*beginr + 16*iGID + 11);
      child.y = 8*beginr + 16*iGID + 8;
    }
    fib = xb.w;
    ab.w = 0; xb.w = 0;

    cosfib = native_cos(fib);

    dotcx = dot(xb,x);
    q = normalize(dotcx*xb - x);
    ra = normalize(xb*cosfib + q*native_sin(fib)); //e
    rb = normalize(xb*cosfib - q*native_sin(fib)); //n

    dotrx = dot(ra,x);
    if ( dotrx < cosfi){
      //extend the cone
      q = normalize(dotrx*x - ra);
      sinfi = (cosfi>(1-EPS))? 0:native_sin(fi);
      e = normalize(x*cosfi + q*sinfi);
      x = normalize(e+ra);
      cosfi = dot(x,ra);
      fi = acos(cosfi);
    }

    dotrx = dot(rb,x);
    if ( dotrx < cosfi){
      //extend the cone
      q = normalize(dotrx*x - rb);
      sinfi = (cosfi>(1-EPS))? 0:native_sin(fi);
      e = normalize(x*cosfi + q*sinfi);
      x = normalize(e+rb);
      cosfi = dot(x,rb);
      fi = acos(cosfi);
    }

    //move the apex
    c = ab - a;
    if ( length(c) > EPS){
      c = normalize(c);
      dotcx = dot(x,c);
      if ( dotcx < cosfi){
        q = (dotcx*x - c)/length(dotcx*x-c);
        sinfi = native_sin(fi);
        e = x*cosfi + q*sinfi;
        n = x*cosfi - q*sinfi;
        g = c - dot(n,c)*n;
        t = (length(g)*length(g))/dot(-e,normalize(g));
        a = a - t*e;
      }

    }

  }
    vstore4(a, 0, cones + 8*beginw + 8*iGID);
    x.w = fi;
    vstore4(x, 0, cones + 8*beginw + 8*iGID + 3);
    cones[8*beginw + 8*iGID + 7] = 2*beginw + 2*iGID;
    //store pointers to children nodes
    vstore2(child, 0, pointers + 2*beginw + 2*iGID);
}
