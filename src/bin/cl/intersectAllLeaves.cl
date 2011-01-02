#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable
#define EPS 0.002

sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

void intersectAllLeaves (
  __read_only image2d_t dir, __read_only image2d_t o,
__read_only image2d_t bounds, __global int* index, __global float* tHit, float4 v1, float4 v2, float4 v3,
float4 e1, float4 e2, const int totalWidth, const int lheight, const int lwidth, const int x, const int y,
const unsigned int offsetGID
#ifdef STAT_RAY_TRIANGLE
, __global unsigned int* stat_rayTriangle
#endif
#ifdef STAT_ALL
, __global unsigned int* stats
#endif
 ){
  float4 s, rayd, rayo;
  float divisor, b1, b2;
  // process all rays in the cone
  #ifdef STAT_ALL
    atom_add(stats,lheight*lwidth);
  #endif

  //read the tile
  for ( int i = 0; i < lheight; i++){
    for ( int j = 0; j < lwidth; j++) {
      #ifdef STAT_RAY_TRIANGLE
        atom_add(stat_rayTriangle + totalWidth*(y + i) + x + j, 1);
      #endif
      rayd = read_imagef(dir, imageSampler, (int2)(x + j, y + i));
      rayo = read_imagef(o, imageSampler, (int2)(x + j, y + i));

      s = cross(rayd, e2);
      divisor = dot(s, e1);
      if ( divisor == 0.0f) continue; //degenarate triangle
      divisor = 1.0f/ divisor;

      // compute first barycentric coordinate
      b1 = dot(rayo-v1, s) * divisor;
      if ( b1 > 0.f  && b1 < 1.f) {

      // compute second barycentric coordinate
      s = cross(rayo - v1, e1);
      b2 = dot(rayd, s) * divisor;
      if ( b2 > 0.f && (b1 + b2) < 1.f) {

      // Compute _t_ to intersection point
      b1 = dot(e2, s) * divisor;

      s = read_imagef(bounds, imageSampler, (int2)(x + j, y + i));
      if (b1 > s.x && b1 < tHit[totalWidth*(y + i) + x + j]) {
        tHit[totalWidth*(y + i) + x + j] = b1;
        index[totalWidth*(y + i) + x + j] = offsetGID;
      } } }
    }
  }
}

void intersectAllLeaves2 (
  __read_only image2d_t dir, __read_only image2d_t o,
__read_only image2d_t bounds, __global int* index, __global float* tHit, float4 v1, float4 v2, float4 v3,
float4 e1, float4 e2, const int totalWidth, const int lheight, const int lwidth, const int x, const int y,
const unsigned int offsetGID
#ifdef STAT_RAY_TRIANGLE
, __global int* stat_rayTriangle
#endif
#ifdef STAT_ALL
, __global unsigned int* stats
#endif
 ){
  float4 s, rayd, rayo;
  float divisor, b1, b2;
  // process all rays in the cone
  #ifdef STAT_ALL
    unsigned int count = 0;
  #endif

  //read the tile
  for ( int i = 0; i < lheight; i++){
    for ( int j = 0; j < lwidth; j++) {
      #ifdef STAT_RAY_TRIANGLE
      atom_add(stat_rayTriangle + totalWidth*(y + i) + x + j, 1);
      #endif
      rayd = read_imagef(dir, imageSampler, (int2)(x + j, y + i));
      if ( rayd.w > 0 ) {
      #ifdef STAT_ALL
        ++count;
      #endif
      rayd.w = 0;
      rayo = read_imagef(o, imageSampler, (int2)(x + j, y + i));

      s = cross(rayd, e2);
      divisor = dot(s, e1);
      if ( divisor != 0.0f) { //degenarate triangle
      divisor = 1.0f/ divisor;

      // compute first barycentric coordinate
      b1 = dot(rayo-v1, s) * divisor;
      if ( b1 > 0.f  && b1 < 1.f) {

      // compute second barycentric coordinate
      s = cross(rayo-v1, e1);
      b2 = dot(rayd, s) * divisor;
      if ( b2 > 0.f && (b1 + b2) < 1.f) {

      // Compute _t_ to intersection point
      b1 = dot(e2, s) * divisor;

      s = read_imagef(bounds, imageSampler, (int2)(x + j, y + i));
      if (b1 > s.x && b1 < tHit[totalWidth*(y + i) + x + j]) {
        tHit[totalWidth*(y + i) + x + j] = b1;
        index[totalWidth*(y + i) + x + j] = offsetGID;
      }
      } } } }
    }
  }

  #ifdef STAT_ALL
  atom_add(stats,count);
  #endif
}

void intersectAllLeavesP (
  __read_only image2d_t dir, __read_only image2d_t o,
__read_only image2d_t bounds, __global char* tHit, float4 v1, float4 v2, float4 v3,
float4 e1, float4 e2, const int totalWidth, const int lheight, const int lwidth, const int x, const int y
#ifdef STAT_PRAY_TRIANGLE
, __global int* stat_rayTriangle
#endif
#ifdef STAT_ALL
, __global unsigned int* stats
#endif
 ){
  float4 s, rayd, rayo;
  float divisor, b1, b2;
  // process all rays in the cone
  #ifdef STAT_ALL
    unsigned int count = 0;
  #endif
  //read the tile
  for ( int i = 0; i < lheight; i++){
    for ( int j = 0; j < lwidth; j++) {
      rayd = read_imagef(dir, imageSampler, (int2)(x + j, y + i));
      if ( rayd.w > 0 ) { //not a valid ray
      rayd.w = 0;
      rayo = read_imagef(o, imageSampler, (int2)(x + j, y + i));

      #ifdef STAT_PRAY_TRIANGLE
      atom_add(stat_rayTriangle + totalWidth*(y + i) + x + j, 1);
      #endif
      #ifdef STAT_ALL
        ++count;
      #endif
      s = cross(rayd, e2);
      divisor = dot(s, e1);
      if ( divisor != 0.0f) { //degenarate triangle
      divisor = 1.0f/ divisor;

      // compute first barycentric coordinate
      b1 = dot(rayo-v1, s) * divisor;
      if ( b1 > 0.f  && b1 < 1.f) {

      // compute second barycentric coordinate
      s = cross(rayo-v1, e1);
      b2 = dot(rayd, s) * divisor;
      if ( b2 > 0.f && (b1 + b2) < 1.f) {

      // Compute _t_ to intersection point
      b1 = dot(e2, s) * divisor;
      s = read_imagef(bounds, imageSampler, (int2)(x + j, y + i));
      if ( b1 > s.x && b1 < s.y){
        tHit[totalWidth*(y + i) + x + j] = '1';
      } } } } }
    }
  }
  #ifdef STAT_ALL
  atom_add(stats,count);
  #endif
}

void yetAnotherIntersectAllLeaves (
  __read_only image2d_t dir, __read_only image2d_t o,
__read_only image2d_t bounds, __global int* index, __global float* tHit, __global int* changed,
float4 v1, float4 v2, float4 v3, float4 e1, float4 e2, const int totalWidth, const int lheight,
const int lwidth, const int x, const int y, const unsigned int GID, const unsigned int offsetGID
#ifdef STAT_RAY_TRIANGLE
, __global int* stat_rayTriangle
#endif
 ){
  float4 s, rayd, rayo;
  float divisor, b1, b2;
  // process all rays in the cone

  //read the tile
  for ( int i = 0; i < lheight; i++){
    for ( int j = 0; j < lwidth; j++) {
      #ifdef STAT_RAY_TRIANGLE
      atom_add(stat_rayTriangle + totalWidth*(y + i) + x + j, 1);
      #endif
      rayd = read_imagef(dir, imageSampler, (int2)(x + j, y + i));
      rayo = read_imagef(o, imageSampler, (int2)(x + j, y + i));

      s = cross(rayd, e2);
      divisor = dot(s, e1);
      if ( divisor == 0.0f) continue; //degenarate triangle
      divisor = 1.0f/ divisor;

      // compute first barycentric coordinate
      b1 = dot(rayo-v1, s) * divisor;
      if ( b1 < EPS  || b1 > 1+EPS) continue;

      // compute second barycentric coordinate
      s = cross(rayo-v1, e1);
      b2 = dot(rayd, s) * divisor;
      if ( b2 < EPS || (b1 + b2) > 1 - EPS) continue;

      // Compute _t_ to intersection point
      b1 = dot(e2, s) * divisor;

      s = read_imagef(bounds, imageSampler, (int2)(x + j, y + i));
      if (b1 < s.x ) continue;

      if ( b1 > tHit[totalWidth*(y + i) + x + j] - EPS
        || index[totalWidth*(y + i) + x + j] == GID) continue;

        tHit[totalWidth*(y + i) + x + j] = b1;
        index[totalWidth*(y + i) + x + j] = GID;
        changed[offsetGID] = totalWidth*(y + i) + x + j;
    }
  }
}
