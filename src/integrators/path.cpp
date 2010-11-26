
/*
    pbrt source code Copyright(c) 1998-2010 Matt Pharr and Greg Humphreys.

    This file is part of pbrt.

    pbrt is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.  Note that the text contents of
    the book "Physically Based Rendering" are *not* licensed under the
    GNU GPL.

    pbrt is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

 */


// integrators/path.cpp*
#include "stdafx.h"
#include "integrators/path.h"
#include "scene.h"
#include "intersection.h"
#include "paramset.h"

// PathIntegrator Method Definitions
void PathIntegrator::RequestSamples(Sampler *sampler, Sample *sample,
                                    const Scene *scene) {
    for (int i = 0; i < SAMPLE_DEPTH; ++i) {
        lightSampleOffsets[i] = LightSampleOffsets(1, sample);
        lightNumOffset[i] = sample->Add1D(1);
        bsdfSampleOffsets[i] = BSDFSampleOffsets(1, sample);
        pathSampleOffsets[i] = BSDFSampleOffsets(1, sample);
    }
}

void PathIntegrator::Li(const Scene *scene, const Renderer *renderer,
        const RayDifferential *ray, const Intersection *isect,
        const Sample *sample, RNG &rng, MemoryArena &arena,
        float* rayWeight, Spectrum* L, bool *hit, const size_t &count  ) const
{
  Spectrum *L_temp = new Spectrum[count];
  Spectrum *f = new Spectrum[count];
  Spectrum* pathThroughput = new Spectrum[count];
  bool* specularBounce = new bool[count];
  BSDF **bsdf = new BSDF*[count];
  Point *p = new Point[count];
  Normal *n = new Normal[count];
  Vector *wo = new Vector[count];
  BSDFSample *outgoingBSDFSample = new BSDFSample[count];
  Vector *wi = new Vector[count];
  float *pdf = new float[count];
  BxDFType *flags = new BxDFType[count];
  RayDifferential *r = new RayDifferential[count];
  Intersection *localIsect = new Intersection[count];

  for ( int i = 0; i < count; i++){
    specularBounce[i] = false;
    if ( !hit[i]) continue;
    localIsect[i] = isect[i];
    r[i].Copy(ray[i]);
    pathThroughput[i] = Spectrum(1.0);
    L[i] = 0.;
  }
  for ( int bounces = 0; ; ++bounces){
    for ( int i  = 0; i < count; i++){
      if ( !hit[i]) continue;
      // Possibly add emitted light at path vertex
      if ( bounces == 0 || specularBounce[i])
        L[i] += pathThroughput[i]*localIsect[i].Le(-r[i].d);

      // Sample illumination from lights to find path contribution
      bsdf[i] = localIsect[i].GetBSDF(r[i], arena);
      p[i] = bsdf[i]->dgShading.p;
      n[i] = bsdf[i]->dgShading.nn;
      wo[i] = -ray[i].d;

    }
    //calls to IntersectP
    if (bounces < SAMPLE_DEPTH)
         // L += pathThroughput *
               UniformSampleOneLight(scene, renderer, arena, p, n, wo,
                   localIsect, r, bsdf, sample, rng,
                   lightNumOffset[bounces], &lightSampleOffsets[bounces],
                   &bsdfSampleOffsets[bounces], rayWeight,L_temp,hit, count);
      else
         // L += pathThroughput *
               UniformSampleOneLight(scene, renderer, arena, p, n, wo,
                   localIsect, r, bsdf, sample, rng, -1, NULL, NULL, rayWeight, L_temp, hit, count);
    for ( int i = 0; i < count; i++){
      if ( !hit[i] ) continue;
      L[i] += pathThroughput[i]*L_temp[i];
    }
    for ( int i = 0; i < count; i++){
        if ( !hit[i]) continue;
        // Get _outgoingBSDFSample_ for sampling new path direction
        if (bounces < SAMPLE_DEPTH)
            outgoingBSDFSample[i] = BSDFSample(&sample[i], pathSampleOffsets[bounces],
                                            0);
        else
            outgoingBSDFSample[i] = BSDFSample(rng);
        f[i] = bsdf[i]->Sample_f(wo[i], &wi[i], outgoingBSDFSample[i], &pdf[i],
                                    BSDF_ALL, &flags[i]);
        if (f[i].IsBlack() || pdf[i] == 0.)
            continue;
        specularBounce[i] = (flags[i] & BSDF_SPECULAR) != 0;
        pathThroughput[i] *= f[i] * AbsDot(wi[i], n[i]) / pdf[i];
        r[i] = RayDifferential(p[i], wi[i], r[i], localIsect[i].rayEpsilon);

        // Possibly terminate the path
        if (bounces > 3) {
            float continueProbability = min(.5f, pathThroughput[i].y());
            if (rng.RandomFloat() > continueProbability){
              hit[i] = false;
              continue;
            }
            pathThroughput[i] /= continueProbability;
        }
    }
    if (bounces == maxDepth){
      break;
    }

    scene->Intersect(r, localIsect, hit, count, 4, bounces);

    // Find next vertex of path
    for ( int i = 0; i < count; i++){
      if ( !hit[i] && specularBounce[i]){
        for (uint32_t i = 0; i < scene->lights.size(); ++i)
          L[i] += pathThroughput[i] * scene->lights[i]->Le(ray[i]);
        continue;
      }

    }
  }

  delete [] L_temp;
  delete [] f;
  delete [] pathThroughput;
  delete [] specularBounce;
  delete [] bsdf;
  delete [] p;
  delete [] n;
  delete [] wo;
  delete [] outgoingBSDFSample;
  delete [] wi;
  delete [] pdf;
  delete [] flags;
  delete [] r;
  delete [] localIsect;
}

Spectrum PathIntegrator::Li(const Scene *scene, const Renderer *renderer,
        const RayDifferential &r, const Intersection &isect,
        const Sample *sample, RNG &rng, MemoryArena &arena) const {
    // Declare common path integration variables
    Spectrum pathThroughput = 1., L = 0.;
    RayDifferential ray(r);
    bool specularBounce = false;
    Intersection localIsect;
    const Intersection *isectp = &isect;
    for (int bounces = 0; ; ++bounces) {
        // Possibly add emitted light at path vertex
        if (bounces == 0 || specularBounce)
            L += pathThroughput * isectp->Le(-ray.d);

        // Sample illumination from lights to find path contribution
        BSDF *bsdf = isectp->GetBSDF(ray, arena);
        const Point &p = bsdf->dgShading.p;
        const Normal &n = bsdf->dgShading.nn;
        Vector wo = -ray.d;
        if (bounces < SAMPLE_DEPTH)
            L += pathThroughput *
                 UniformSampleOneLight(scene, renderer, arena, p, n, wo,
                     isectp->rayEpsilon, ray.time, bsdf, sample, rng,
                     lightNumOffset[bounces], &lightSampleOffsets[bounces],
                     &bsdfSampleOffsets[bounces]);
        else
            L += pathThroughput *
                 UniformSampleOneLight(scene, renderer, arena, p, n, wo,
                     isectp->rayEpsilon, ray.time, bsdf, sample, rng);

        // Sample BSDF to get new path direction

        // Get _outgoingBSDFSample_ for sampling new path direction
        BSDFSample outgoingBSDFSample;
        if (bounces < SAMPLE_DEPTH)
            outgoingBSDFSample = BSDFSample(sample, pathSampleOffsets[bounces],
                                            0);
        else
            outgoingBSDFSample = BSDFSample(rng);
        Vector wi;
        float pdf;
        BxDFType flags;
        Spectrum f = bsdf->Sample_f(wo, &wi, outgoingBSDFSample, &pdf,
                                    BSDF_ALL, &flags);
        if (f.IsBlack() || pdf == 0.)
            break;
        specularBounce = (flags & BSDF_SPECULAR) != 0;
        pathThroughput *= f * AbsDot(wi, n) / pdf;
        ray = RayDifferential(p, wi, ray, isectp->rayEpsilon);

        // Possibly terminate the path
        if (bounces > 3) {
            float continueProbability = min(.5f, pathThroughput.y());
            if (rng.RandomFloat() > continueProbability)
                break;
            pathThroughput /= continueProbability;
        }
        if (bounces == maxDepth)
            break;

        // Find next vertex of path
        if (!scene->Intersect(ray, &localIsect)) {
            if (specularBounce)
                for (uint32_t i = 0; i < scene->lights.size(); ++i)
                   L += pathThroughput * scene->lights[i]->Le(ray);
            break;
        }
        pathThroughput *= renderer->Transmittance(scene, ray, NULL, rng, arena);
        isectp = &localIsect;
    }
    return L;
}


PathIntegrator *CreatePathSurfaceIntegrator(const ParamSet &params) {
    int maxDepth = params.FindOneInt("maxdepth", 5);
    return new PathIntegrator(maxDepth);
}


