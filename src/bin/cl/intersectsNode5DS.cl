bool intersectsNode(float4 center, float2 uvmin, float2 uvmax, float4 o, float radius) {
  float2 uv;
  float4 ray = o - center;
  float len = length(ray);
  ray = ray/len;
  uv.x = (ray.x == 0)? 0: atan(ray.y/ray.x);
  uv.y = acos(ray.z);

  len = atan(radius/len);
  if ( uv.x - len < 0 && uv.x + len > 0 ) return true;

  if ( max(uv.x - len, uvmin.x) < min(uv.x + len, uvmax.x) + EPS &&
    max(uv.y - len, uvmin.y) < min(uv.y + len, uvmax.y) + EPS )
    return true;

  return false;
}
