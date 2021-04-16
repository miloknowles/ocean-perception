#pragma once

#include "core/eigen_types.hpp"
#include "core/range_measurement.hpp"

namespace bm {
namespace vio {

using namespace core;


double EstimatePosition(const MultiRange& ranges,
                        const std::vector<double>& sigmas,
                        Vector3d& world_t_body,
                        Matrix3d& solution_cov,
                        int max_iters,
                        double min_error = 1e-3);

}
}
