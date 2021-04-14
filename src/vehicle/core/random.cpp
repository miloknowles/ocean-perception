#include <random>

#include "core/random.hpp"

namespace bm {
namespace core {


// NOTE(milo): All of the random functions rely on this generator to create noise.
static std::default_random_engine _G;

// Return a random float in the range [a, b).
float RandomUniformf(float a, float b)
{
  std::uniform_real_distribution<float> d(a, b);
  return d(_G);
}


float RandomNormalf(float mu, float sigma)
{
  std::normal_distribution<float> d(mu, sigma);
  return d(_G);
}


// Return a random double in the range [a, b).
double RandomUniformd(double a, double b)
{
  std::uniform_real_distribution<double> d(a, b);
  return d(_G);
}


double RandomNormald(double mu, double sigma)
{
  std::normal_distribution<double> d(mu, sigma);
  return d(_G);
}


// Returns a unit vector that is uniformly distributed across a unit sphere.
// https://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space
Vector3f RandomUnit3f()
{
  Vector3f r(RandomNormalf(0, 1), RandomNormalf(0, 1), RandomNormalf(0, 1));
  return r.normalized();
}


Vector3d RandomUnit3d()
{
  Vector3d r(RandomNormald(0, 1), RandomNormald(0, 1), RandomNormald(0, 1));
  return r.normalized();
}


Vector3d RandomNormal3d(double mu, double sigma)
{
  return Vector3d(RandomNormald(mu, sigma),
                  RandomNormald(mu, sigma),
                  RandomNormald(mu, sigma));
}


Vector2d RandomNormal2d(double mu, double sigma)
{
  return Vector2d(RandomNormald(mu, sigma),
                  RandomNormald(mu, sigma));
}

}
}
