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

  float4 center1, center2;
  float2 u1, v1, u2, v2;
  float dist;
  float radius1, radius2;

  float4 tempCenter;
  float tempRadius;
  float2 tempu, tempv;

  center1 = vload4(0,cones + 9*beginr + 18*iGID);
  radius1 = center1.w;
  center1.w = 0;
  u1 = vload2(0,cones + 9*beginr + 18*iGID + 4);
  v1 = vload2(0,cones + 9*beginr + 18*iGID + 6);

  cones[9*beginw + 9*iGID + 8] = 1;
  if ( !( iGID == (levelcount - 1) && temp % 2 == 1 )) {
    //last thread only copies node1 to the output
    cones[9*beginw + 9*iGID + 8] = 2;
    center2 = vload4(0,cones + 9*beginr + 18*iGID + 9);
    radius2 = center2.w;
    center2.w = 0;
    u2 = vload2(0,cones + 9*beginr + 18*iGID + 13);
    v2 = vload2(0,cones + 9*beginr + 18*iGID + 15);

    //find 2D boundign box for uv
    u1.x = (u1.x < u2.x)? u1.x : u2.x;
    u1.y = (u1.y > u2.y)? u1.y : u2.y;
    v1.x = (v1.x < v2.x)? v1.x : v2.x;
    v1.y = (v1.y > v2.y)? v1.y : v2.y;

    //union the spheres according to http://answers.google.com/answers/threadview/id/342125.html
    //swap the spheres if sphere2 is bigger
    if ( radius2 > radius1 + EPS ){
      tempCenter = center1;
      tempu = u1;
      tempv = v1;
      tempRadius = radius1;

      center1 = center2;
      u1 = u2;
      v1 = v2;
      radius1 = radius2;

      center2 = tempCenter;
      u2 = tempu;
      v2 = tempv;
      radius2 = tempRadius;
    }
    //is sphere inside the other one?
    tempCenter = center2 - center1;
    dist = length(tempCenter);
    if (dist + radius2 > ( radius1 + EPS)){
      //compute bounding sphere of the old ones
      center1 = center1 + (0.5f*(radius2 + dist - radius1)/dist)*tempCenter;
      radius1 = (radius1 + radius2 + dist)/2;
    }


  }
  //store the result
  center1.w = radius1;
  vstore4(center1, 0, cones + 9*beginw + 9*iGID);
  vstore2(u1, 0, cones + 9*beginw + 9*iGID + 4);
  vstore2(v1, 0, cones + 9*beginw + 9*iGID + 6);
}
