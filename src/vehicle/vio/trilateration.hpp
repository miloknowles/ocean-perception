#pragma once

#include "core/eigen_types.hpp"
#include "core/range_measurement.hpp"

namespace bm {
namespace vio {

using namespace core;


// Use trilateration (3 or more range measurements) to estimate the robot's position.
// Internally uses a Levenberg-Marquardt least squares optimization, starting with world_t_body
// as an initial guess. The "sigmas" vector should contain measurement noise for each of the ranges.
double TrilateratePosition(const MultiRange& ranges,
                          const std::vector<double>& sigmas,
                          Vector3d& world_t_body,
                          Matrix3d& solution_cov,
                          int max_iters,
                          double min_error = 1e-3);

}
}
