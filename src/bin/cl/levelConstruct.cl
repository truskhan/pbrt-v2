#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

__kernel void levelConstruct(__global float* cones, const int count,
  const int threadsCount, const int level){
  int iGID = get_global_id(0);

  int beginr = 0;
  int beginw = 0;
  int levelcount = threadsCount; //end of level0
  int temp;

  for ( int i = 0; i < level; i++){
      beginw += levelcount;
      temp = levelcount;
      levelcount = (levelcount+1)/2; //number of elements in level
  }
  beginr = beginw - temp;

  if ( iGID >= levelcount ) return;

  float4 x, q, c, a, g, xb;
  float4 ab,e,n, ra, rb;
  float cosfi, sinfi, cosfib;
  float dotrx, dotcx, t ;
  float fi,fib;

  x = (float4)(cones[8*beginr + 16*iGID+3],cones[8*beginr + 16*iGID+4],cones[8*beginr + 16*iGID+5],0);
  a = (float4)(cones[8*beginr + 16*iGID],cones[8*beginr + 16*iGID+1],cones[8*beginr + 16*iGID+2],0);
  fi = cones[8*beginr + 16*iGID+6];
  cosfi = native_cos(fi);
  cones[8*beginw + 8*iGID + 7] = 1;
  if ( !( iGID == (levelcount - 1) && temp % 2 == 1 )) {
    //posledni vlakno jen prekopiruje
    cones[8*beginw + 8*iGID + 7] = 2;
    ab = (float4)(cones[8*beginr + 16*iGID+8],cones[8*beginr + 16*iGID+9],cones[8*beginr + 16*iGID+10],0);
    xb = (float4)(cones[8*beginr + 16*iGID+11],cones[8*beginr + 16*iGID+12],cones[8*beginr + 16*iGID+13],0);
    fib = cones[8*beginr + 16*iGID+14];
    cosfib = native_cos(fib);

    dotcx = dot(xb,x);
    q = normalize(dotcx*xb - x);
    ra = xb*cosfib + q*native_sin(fib); //e
    rb = xb*cosfib - q*native_sin(fib); //n

    dotrx = dot(ra,x);
    if ( dotrx < cosfi){
      //extend the cone
      q = normalize(dotrx*x - ra);
      sinfi = (cosfi>(1-EPS))? 0:native_sin(acos(cosfi));
      e = normalize(x*cosfi + q*sinfi);
      x = normalize(e+ra);
      cosfi = dot(x,ra);
      fi = acos(cosfi);
    }

    dotrx = dot(rb,x);
    if ( dotrx < cosfi){
      //extend the cone
      q = normalize(dotrx*x - rb);
      sinfi = (cosfi>(1-EPS))? 0:native_sin(acos(cosfi));
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
        q = (dotcx*x - c)/length(dotcx*x-x);
        sinfi = native_sin(fi);
        e = x*cosfi + q*sinfi;
        n = x*cosfi - q*sinfi;
        g = c - dot(n,c)*n;
        t = (length(g)*length(g))/dot(e,g);
        a = a - t*e;
      }

    }

  }
    cones[8*beginw + 8*iGID]     = a.x;
    cones[8*beginw + 8*iGID + 1] = a.y;
    cones[8*beginw + 8*iGID + 2] = a.z;
    cones[8*beginw + 8*iGID + 3] = x.x;
    cones[8*beginw + 8*iGID + 4] = x.y;
    cones[8*beginw + 8*iGID + 5] = x.z;
    cones[8*beginw + 8*iGID + 6] = fi;
}
