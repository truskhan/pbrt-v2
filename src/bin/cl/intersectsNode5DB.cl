#define EPS 0.002f
bool intersectsNode(float4 omin, float4 omax, float2 uvmin, float2 uvmax, float4 bmin, float4 bmax) {
 float4 ocenter = (float4)0;
 float4 ray;
 float2 uv;
 float2 tmin, tmax;
 int range = 0;

//Minkowski sum of the two boxes (sum the widths/heights and position it at boxB_pos - boxA_pos).
 ray = (omax - omin)/2;
 ocenter = ray + omin;
 ocenter.w = 0;

 bmin -= ray;
 bmax += ray;

 ray = (float4)0;
 ray = normalize((float4)(bmin.x, bmin.y, bmin.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 tmin.x = tmax.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 tmin.y = tmax.y = ray.z;

 ray = normalize((float4)(bmax.x, bmax.y, bmax.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmin.x, bmax.y, bmin.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmin.x, bmax.y, bmax.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmin.x, bmin.y, bmax.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmax.x, bmin.y, bmin.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmax.x, bmax.y, bmin.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 ray = normalize((float4)(bmax.x, bmin.y, bmax.z,0) - ocenter);
 range |= (ray.x < 0) ? 0x1000 : 0x0100;
 range |= (ray.y < 0) ? 0x0010 : 0x0001;
 uv.x = ( ray.x == 0) ? 0 : ray.y/ray.x;
 uv.y = ray.z;
 tmin = min(tmin, uv);
 tmax = max(tmax, uv);

 if ( (range & 0x1100 == 0x1100) || (range & 0x0011 == 0x0011)) return true;

 return (( max(tmin.x, uvmin.x) < min(tmax.x, uvmax.x)) + EPS && (max(tmin.y, uvmin.y) < min(tmax.y, uvmax.y) + EPS));

}
