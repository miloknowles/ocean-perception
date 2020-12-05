#pragma once

#include <random>

#include "core/eigen_types.hpp"

namespace bm {
namespace core {


// NOTE(milo): All of the random functions rely on this generator to create noise.
static std::default_random_engine _G;


// Return a random float in the range [a, b).
inline float RandomUniform(float a, float b)
{
  std::uniform_real_distribution<float> d(a, b);
  return d(_G);
}


inline float RandomNormal(float mu, float sigma)
{
  std::normal_distribution<float> d(mu, sigma);
  return d(_G);
}


// Returns a unit vector that is uniformly distributed across a unit sphere.
// https://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space
inline Vector3f RandomUnit3f()
{
  Vector3f r(RandomNormal(0, 1), RandomNormal(0, 1), RandomNormal(0, 1));
  return r.normalized();
}

}
}
