#ifndef INC_GUARD_LOOP_HPP
#define INC_GUARD_LOOP_HPP

#include "ceres/autodiff_cost_function.h"
#include <Eigen/Geometry>
#include <ceres/cost_function.h>

namespace sfm
{
class PoseGraph3dErrorTerm
{
public:
  PoseGraph3dErrorTerm(
    Eigen::Matrix4f T_ab_measured
  )
  : T_ab_measured{T_ab_measured},
    q_ab_measured{T_ab_measured.block<3, 3>(0, 0)}
  {}

  /// @brief The residual function
  /// @param p_a_ptr the pointer to the x, y, z pose of the first node
  /// @param q_a_ptr the pointer to the quaternion orientation of the first node
  /// @param p_b_ptr the pointer to the x, y, z pose of the second node
  /// @param q_b_ptr the pointer to the quaternion orientation of the second node
  /// @param residuals_ptr the pointer to the residuals
  template<typename T>
  bool operator()(
    const T * const p_a_ptr, const T * q_a_ptr,
    const T * const p_b_ptr, const T * q_b_ptr,
    T * residuals_ptr) const
  {
    // map quaternions to eigen types
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(p_a_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> q_a(q_a_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(p_b_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> q_b(q_b_ptr);

    // Compute the relative transformation between the two frames
    Eigen::Quaternion<T> q_a_inv = q_a.conjugate();
    Eigen::Quaternion<T> q_ab_est = q_a_inv * q_b;

    // Represent the displacement in the A frame
    Eigen::Matrix<T, 3, 1> p_ab_est = q_a_inv * (p_b - p_a);

    // compute the error between the two orientation estimates
    Eigen::Quaternion<T> delta_q = q_ab_measured.template cast<T>() * q_ab_est.conjugate();

    // Compute the residuals
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) = p_ab_est - T_ab_measured.block<3, 1>(
      3, 0).template cast<T>();

    return true;
  }

  static ceres::CostFunction * Create(
    const Eigen::Matrix4f T_ab_measured)
  {
    return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(
      new PoseGraph3dErrorTerm(T_ab_measured));
  }

private:
  Eigen::Matrix4f T_ab_measured;
  Eigen::Quaternionf q_ab_measured;
};
}

#endif // INC_GUARD_LOOP_HPP
