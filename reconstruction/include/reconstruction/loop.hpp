#ifndef INC_GUARD_LOOP_HPP
#define INC_GUARD_LOOP_HPP

#include "ceres/autodiff_cost_function.h"
#include <Eigen/Geometry>
#include <Eigen/src/Geometry/Quaternion.h>
#include <ceres/cost_function.h>
#include <ceres/problem.h>

namespace sfm
{
class PoseGraph3dErrorTerm
{
public:
  PoseGraph3dErrorTerm(
    Eigen::Matrix4d T_ab_measured,
    double scaler = 1.0
  )
  : T_ab_measured{T_ab_measured},
    q_ab_measured{T_ab_measured.block<3, 3>(0, 0)},
    p_ab_measured{T_ab_measured.block<3, 1>(0, 3)},
    scaler{scaler}
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

    Eigen::Matrix<T, 4, 4> T_aw = Eigen::Matrix<T, 4, 4>::Identity();
    T_aw.template block<3, 3>(0, 0) = q_a.toRotationMatrix();
    T_aw.template block<3, 1>(0, 3) = p_a;
    Eigen::Matrix<T, 4, 4> T_bw = Eigen::Matrix<T, 4, 4>::Identity();
    T_bw.template block<3, 3>(0, 0) = q_b.toRotationMatrix();
    T_bw.template block<3, 1>(0, 3) = p_b;

    Eigen::Matrix<T, 4, 4> T_ab_est = T_aw * T_bw.inverse();
    Eigen::Matrix<T, 3, 3> R_ab_est = T_ab_est.template block<3, 3>(0, 0);

    Eigen::Matrix<T, 3, 1> p_ab_est = p_ab_measured.template cast<T>() - T_ab_est.template block<3,
        1>(
      0,
      3);

    Eigen::Quaternion<T> q_ab_est{R_ab_est};
    Eigen::Quaternion<T> delta_q = q_ab_measured.template cast<T>() * q_ab_est.conjugate();

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) = p_ab_est * scaler;
    residuals.template block<3, 1>(3, 0) = delta_q.vec() * scaler;

    return true;
  }

  static ceres::CostFunction * Create(
    const Eigen::Matrix4d T_ab_measured, double scaler = 1.0)
  {
    return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(
      new PoseGraph3dErrorTerm(T_ab_measured));
  }

private:
  Eigen::Matrix4d T_ab_measured;
  Eigen::Quaterniond q_ab_measured;
  Eigen::Vector3d p_ab_measured;
  double scaler;
};
}

#endif // INC_GUARD_LOOP_HPP
