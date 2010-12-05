#define EPS 0.002f
bool intersectsNode(float4 omin, float4 omax, float4 uvmin, float4 uvmax, float4 bmin, float4 bmax) {
 float4 ocenter = (float4)0;
 float4 ray;
 float4 tmin, tmax;
 float4 fmin, fmax;
 bool ret;

//Minkowski sum of the two boxes
 ray = (omax - omin)/2;
 bmin -= ray;
 bmax += ray;

 ocenter = ray + omin;
 ocenter.w = 0;

 ray = normalize((float4)(bmin.x, bmin.y, bmin.z,0) - ocenter);
 tmin = ray;
 tmax = ray;

 ray = normalize((float4)(bmin.x, bmin.y, bmax.z,0) - ocenter);
 tmin = min(tmin, ray);
 tmax = max(tmax, ray);

 ray = normalize((float4)(bmin.x, bmax.y, bmin.z,0) - ocenter);
 tmin = min(tmin, ray);
 tmax = max(tmax, ray);

 ray = normalize((float4)(bmin.x, bmax.y, bmax.z,0) - ocenter);
 tmin = min(tmin, ray);
 tmax = max(tmax, ray);

 ray = normalize((float4)(bmax.x, bmin.y, bmin.z,0) - ocenter);
 tmin = min(tmin, ray);
 tmax = max(tmax, ray);

 ray = normalize((float4)(bmax.x, bmax.y, bmin.z,0) - ocenter);
 tmin = min(tmin, ray);
 tmax = max(tmax, ray);

 ray = normalize((float4)(bmax.x, bmin.y, bmax.z,0) - ocenter);
 tmin = min(tmin, ray);
 tmax = max(tmax, ray);

 ray = normalize((float4)(bmax.x, bmax.y, bmax.z,0) - ocenter);
 tmin = min(tmin, ray);
 tmax = max(tmax, ray);

 fmin = max(tmin, uvmin);
 fmax = min(tmax, uvmax);

 if ( tmin.x < 0 && tmax.x > 0)
  ret = (fmin.x < fmax.x + EPS);
 if ( tmin.y < 0 && tmax.y > 0)
  ret |= (fmin.y < fmax.y + EPS);
 if ( tmin.z < 0 && tmax.z > 0)
  ret |= (fmin.z < fmax.z + EPS);

 return (ret || ( fmin.x < (fmax.x + EPS) && fmin.y < (fmax.y + EPS) && fmin.z < (fmax.z + EPS) ));
}
