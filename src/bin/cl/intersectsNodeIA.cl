/**
 * @file intersectsNodeIA.cl
 * @author: Hana Truskova hana.truskova@seznam.cz
**/
/** method for deciding if IA node - triangle can intersects */
bool intersectsNode ( float4 t_omin, float4 t_omax, float4 t_dmin, float4 t_dmax, float4 bmin, float4 bmax){
  //compute (Bx-Ox)*(1/Vx)
  float2 s,t,u;
  float4 temp;
  //compute (Bx-0x)
  float4 omin = bmin - t_omax;
  float4 omax = bmax - t_omin;
  float4 dmin;
  float4 dmax;
  //compute (1/Vx)
  if ( (t_dmin.x <= 0 && t_dmax.x >= 0) || (t_dmin.y <= 0 && t_dmax.y >= 0) || (t_dmin.z <= 0 && t_dmax.z >= 0) )
     return true;

  dmin.x = 1/t_dmax.x;
  dmax.x = 1/t_dmin.x;
  dmin.y = 1/t_dmax.y;
  dmax.y = 1/t_dmin.y;
  dmin.z = 1/t_dmax.z;
  dmax.z = 1/t_dmin.z;

  temp.x = omin.x*dmin.x;
  temp.y = omax.x*dmin.x;
  temp.z = omax.x*dmax.x;
  temp.w = omin.x*dmax.x;
  s.x = min(temp.x, temp.y);
  s.x = min(s.x, temp.z);
  s.x = min(s.x, temp.w);
  s.y = max(temp.x, temp.y);
  s.y = max(s.y, temp.z);
  s.y = max(s.y, temp.w);

  temp.x = omin.y*dmin.y;
  temp.y = omax.y*dmin.y;
  temp.z = omax.y*dmax.y;
  temp.w = omin.y*dmax.y;
  t.x = min(temp.x, temp.y);
  t.x = min(t.x, temp.z);
  t.x = min(t.x, temp.w);
  t.y = max(temp.x, temp.y);
  t.y = max(t.y, temp.z);
  t.y = max(t.y, temp.w);

  temp.x = omin.z*dmin.z;
  temp.y = omax.z*dmin.z;
  temp.z = omax.z*dmax.z;
  temp.w = omin.z*dmax.z;
  u.x = min(temp.x, temp.y);
  u.x = min(u.x, temp.z);
  u.x = min(u.x, temp.w);
  u.y = max(temp.x, temp.y);
  u.y = max(u.y, temp.z);
  u.y = max(u.y, temp.w);

  s.x = max(s.x, t.x);
  s.x = max(s.x, u.x);
  s.y = min(s.y, t.y);
  s.y = min(s.y, u.y);

  return (s.x < s.y + EPS);
}
