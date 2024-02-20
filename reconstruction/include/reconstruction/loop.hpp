#ifndef INC_GUARD_LOOP_HPP
#define INC_GUARD_LOOP_HPP

#include "ceres/autodiff_cost_function.h"
#include <Eigen/Geometry>
#include <Eigen/src/Geometry/Quaternion.h>
#include <ceres/cost_function.h>
#include <ceres/problem.h>

#include <sophus/se3.hpp>

namespace sfm
{
class PoseGraph3dErrorTerm
{
public:
  PoseGraph3dErrorTerm(
    Eigen::Matrix4d T_ab_measured
  )
  : T_ab_measured{T_ab_measured}
  {}

  /// @brief The residual function
  /// @param p_a_ptr the pointer to the x, y, z pose of the first node
  /// @param q_a_ptr the pointer to the quaternion orientation of the first node
  /// @param p_b_ptr the pointer to the x, y, z pose of the second node
  /// @param q_b_ptr the pointer to the quaternion orientation of the second node
  /// @param residuals_ptr the pointer to the residuals
  template<typename T>
  bool operator()(
    const T * const a_se3_vec,
    const T * const b_se3_vec,
    T * residuals_ptr) const
  {
    Eigen::Map<const Eigen::Matrix<T, 6, 1>> a_se3(a_se3_vec);
    Eigen::Map<const Eigen::Matrix<T, 6, 1>> b_se3(b_se3_vec);

    Eigen::Matrix<T, 4, 4> T_a = Sophus::SE3<T>::exp(a_se3).matrix();
    Eigen::Matrix<T, 4, 4> T_b = Sophus::SE3<T>::exp(b_se3).matrix();

    Eigen::Matrix<T, 4, 4> T_id = T_ab_measured.template cast<T>() * T_b * T_a.inverse();
    Eigen::Vector<T, 6> e_ab = Sophus::SE3<T>{T_id}.log();

    T cost = e_ab.transpose() * Eigen::Matrix<T, 6, 6>::Identity() * e_ab;

    // map residuals
    residuals_ptr[0] = cost;

    return true;
  }

  static ceres::CostFunction * Create(
    const Eigen::Matrix4d T_ab_measured)
  {
    return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 1, 6, 6>(
      new PoseGraph3dErrorTerm(T_ab_measured));
  }

private:
  Eigen::Matrix4d T_ab_measured;
};
}

#endif // INC_GUARD_LOOP_HPP
