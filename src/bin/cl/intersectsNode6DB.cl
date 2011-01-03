/**
 * @file intersectsNode6DB.cl
 * @author: Hana Truskova hana.truskova@seznam.cz
**/
/** method for deciding if 6DB node - triangle can intersects */
bool intersectsNode(float4 omin, float4 omax, float4 uvmin, float4 uvmax, float4 bmin, float4 bmax) {
 float4 ocenter;
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

bool intersectsTri(float4 omin, float4 omax, float4 uvmin, float4 uvmax, float4 v1, float4 v2, float4 v3) {
 float4 ocenter;
 float4 tmin, tmax;
 float4 fmin, fmax;
 bool ret;

//Minkowski sum of the two boxes
 ocenter = (omin+omax)/2;
 ocenter.w = 0;

 v1 = normalize((float4)(v1.x, v1.y, v1.z,0) - ocenter);
 tmin = v1;
 tmax = v1;

 v2 = normalize((float4)(v2.x, v2.y, v2.z,0) - ocenter);
 tmin = min(tmin, v2);
 tmax = max(tmax, v2);

 v3 = normalize((float4)(v3.x, v3.y, v3.z,0) - ocenter);
 tmin = min(tmin, v3);
 tmax = max(tmax, v3);

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
