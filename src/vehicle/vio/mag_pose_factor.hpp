#pragma once

#include <iostream>

#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam {

/**
 * Factor to estimate rotation of a Pose2 or Pose3 given a magnetometer reading.
 * This version uses the measurement model bM = scale * bRn * direction + bias,
 * and assumes scale, direction, and the bias are known.
 *
 * Adapted from: https://github.com/borglab/gtsam/blob/develop/gtsam/navigation/MagFactor.h
 */
template <class POSE>
class MagPoseFactor: public NoiseModelFactor1<POSE> {
 private:
  typedef MagPoseFactor<POSE> This;
  typedef NoiseModelFactor1<POSE> Base;
  typedef typename POSE::Translation Point;   // Could be a Vector2 or Vector3 depending on POSE.
  typedef typename POSE::Rotation Rot;

  const Point measured_;   ///< The measured magnetometer data.
  const Point nM_;         ///< Local magnetic field (in mag output units).
  const Point bias_;       ///< The bias vector (in mag output units).

  static const int MeasDim = Point::RowsAtCompileTime;
  static const int PoseDim = traits<POSE>::dimension;
  static const int RotDim = traits<Rot>::dimension;

  boost::optional<POSE> body_P_sensor_; ///< The pose of the sensor in the body frame.

 public:
  /**
   * @param pose_key of the unknown pose nav_P_body in the factor graph.
   * @param measurement magnetometer reading, a 2D or 3D vector
   * @param scale by which a unit vector is scaled to yield a magnetometer reading
   * @param direction of the local magnetic field, see e.g. http://www.ngdc.noaa.gov/geomag-web/#igrfwmm
   * @param bias of the magnetometer, modeled as purely additive (after scaling)
   * @param model of the additive Gaussian noise that is assumed
   * @param body_P_sensor an optional transform of the magnetometer in the body frame
   */
  MagPoseFactor(Key pose_key,
                const Point& measured,
                double scale,
                const Point& direction,
                const Point& bias,
                const SharedNoiseModel& model,
                const boost::optional<POSE>& body_P_sensor)
      : Base(model, pose_key),
        measured_(measured),
        nM_(scale * direction.normalized()),
        bias_(bias),
        body_P_sensor_(body_P_sensor) {}

  /// @return a deep copy of this factor.
  NonlinearFactor::shared_ptr clone() const override
  {
    return boost::static_pointer_cast<NonlinearFactor>(
        NonlinearFactor::shared_ptr(new This(*this)));
  }

  // Return the factor's error h(x) - z, and the optional Jacobian.
  Vector evaluateError(const POSE& nPb, boost::optional<Matrix&> H = boost::none) const override
  {
    // Get rotation of the nav frame in the sensor frame.
    const Rot nRs = body_P_sensor_ ? nPb.rotation() * body_P_sensor_->rotation() : nPb.rotation();

    // Predict the measured magnetic field h(x) in the sensor frame.
    Matrix H_rot = Matrix::Zero(MeasDim, RotDim);
    const Point hx = nRs.unrotate(nM_, H_rot, boost::none) + bias_;

    if (H) {
      *H = Matrix::Zero(MeasDim, PoseDim);
      const size_t rot0 = nPb.rotationInterval().first;
      (*H).block(0, rot0, MeasDim, RotDim) = H_rot;
    }

    return (hx - measured_);
  }
};

}
