#pragma once

#include <random>

#include "core/eigen_types.hpp"

namespace bm {
namespace core {


// NOTE(milo): All of the random functions rely on this generator to create noise.
static std::default_random_engine _G;


// Return a random double in the range [a, b).
template<typename Scalar>
inline double RandomUniform(Scalar a, Scalar b)
{
  std::uniform_real_distribution<Scalar> d(a, b);
  return d(_G);
}

template <typename Scalar>
inline double RandomNormal(Scalar mu, Scalar sigma)
{
  std::normal_distribution<Scalar> d(mu, sigma);
  return d(_G);
}


// Returns a unit vector that is uniformly distributed across a unit sphere.
// https://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space
inline Vector3f RandomUnit3f()
{
  Vector3f r(RandomNormal<float>(0, 1), RandomNormal<float>(0, 1), RandomNormal<float>(0, 1));
  return r.normalized();
}

inline Vector3d RandomUnit3d()
{
  Vector3d r(RandomNormal<double>(0, 1), RandomNormal<double>(0, 1), RandomNormal<double>(0, 1));
  return r.normalized();
}

inline Vector3d RandomNormal3d(double mu, double sigma)
{
  return Vector3d(RandomNormal<double>(mu, sigma),
                  RandomNormal<double>(mu, sigma),
                  RandomNormal<double>(mu, sigma));
}

inline Vector2d RandomNormal2d(double mu, double sigma)
{
  return Vector2d(RandomNormal<double>(mu, sigma),
                  RandomNormal<double>(mu, sigma));
}

}
}
