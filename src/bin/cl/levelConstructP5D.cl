#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

__kernel void levelConstructP(__global float* cones, __global int* pointers, const int count,
  const int threadsCount, const int level){
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

  float4 center1, center2;
  float2 uvmin1, uvmax1, uvmin2, uvmax2;
  float dist;
  float radius1, radius2;

  float4 tempCenter;
  float tempRadius;
  float2 tempu, tempv;

    center1 = vload4(0,cones + 9*beginr + 18*iGID);
    uvmin1 = vload2(0,cones + 9*beginr + 18*iGID + 4);
    uvmax1 = vload2(0,cones + 9*beginr + 18*iGID + 6);
    child.x = 9*beginr + 18*iGID;

  radius1 = center1.w;
  center1.w = 0;
  child.y = -1;

  if ( !( iGID == (levelcount - 1) && temp % 2 == 1 )) {
    //last thread only copies node1 to the output
      center2 = vload4(0,cones + 9*beginr + 18*iGID + 9);
      uvmin2 = vload2(0,cones + 9*beginr + 18*iGID + 13);
      uvmax2 = vload2(0,cones + 9*beginr + 18*iGID + 15);
      child.y = 9*beginr + 18*iGID + 9;

    radius2 = center2.w;
    center2.w = 0;

    //find 2D boundign box for uv
    uvmin1 = min(uvmin1, uvmin2);
    uvmax1 = max(uvmax1, uvmax2);

    //union the spheres according to http://answers.google.com/answers/threadview/id/342125.html
    //swap the spheres if sphere2 is bigger
    if ( radius2 > radius1 + EPS ){
      tempCenter = center1;
      tempu = uvmin1;
      tempv = uvmax1;
      tempRadius = radius1;

      center1 = center2;
      uvmin1 = uvmin2;
      uvmax1 = uvmax2;
      radius1 = radius2;

      center2 = tempCenter;
      uvmin2 = tempu;
      uvmax2 = tempv;
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
  vstore2(uvmin1, 0, cones + 9*beginw + 9*iGID + 4);
  vstore2(uvmax1, 0, cones + 9*beginw + 9*iGID + 6);
  //store index into pointers
  cones[9*beginw + 9*iGID + 8] = 2*beginw + 2*iGID;
  //store pointers to children nodes
  vstore2(child, 0, pointers + 2*beginw + 2*iGID);
}
