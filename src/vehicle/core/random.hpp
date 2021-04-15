#pragma once

#include "core/eigen_types.hpp"

namespace bm {
namespace core {


// Return a random float in the range [a, b).
float RandomUniformf(float a, float b);
float RandomNormalf(float mu, float sigma);

// Return a random double in the range [a, b).
double RandomUniformd(double a, double b);
double RandomNormald(double mu, double sigma);

// Returns a unit vector that is uniformly distributed across a unit sphere.
// https://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space
Vector3f RandomUnit3f();
Vector3d RandomUnit3d();

Vector3d RandomNormal3d(double mu, double sigma);
Vector2d RandomNormal2d(double mu, double sigma);


}
}
