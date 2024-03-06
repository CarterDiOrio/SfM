#ifndef BUNDLE_ADJUST_HPP_GUARD_
#define BUNDLE_ADJUST_HPP_GUARD_

#include "ceres/ceres.h"
#include "reconstruction/pinhole.hpp"
#include <Eigen/Dense>
#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>
#include <sophus/se3.hpp>


namespace sfm
{

/// camera model for use with ceres. The camera is parameterized using 3 parameters.
/// 3 for rotation, 3 for translation.
struct ReprojectionError
{
  ReprojectionError(
    double observed_x,
    double observed_y,
    const PinholeModel model)
  : observed_x(observed_x), observed_y(observed_y), model(model) {}

  template<typename T>
  bool operator()(
    const T * const camera_params,
    const T * const point, T * residuals) const
  {
    Eigen::Vector<T, 6> se3{
      camera_params[0], camera_params[1], camera_params[2],
      camera_params[3], camera_params[4], camera_params[5]
    };
    Eigen::Vector<T, 3> p{
      point[0], point[1], point[2]
    };

    Sophus::SE3<T> T_kw = Sophus::SE3<T>::exp(se3);
    Eigen::Vector<T, 3> point_k = (T_kw * p.homogeneous()).template head<3>();

    T x = point_k(0);
    T y = point_k(1);
    T z = point_k(2);

    const auto projected_x = (x / z) * model.fx + model.cx;
    const auto projected_y = (y / z) * model.fy + model.cy;

    residuals[0] = projected_x - T{observed_x};
    residuals[1] = projected_y - T{observed_y};

    return true;
  }

  static ceres::CostFunction * Create(
    double observed_x, double observed_y, PinholeModel model
  )
  {
    return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
      new ReprojectionError(observed_x, observed_y, model)
    );
  }

  const double observed_x;
  const double observed_y;
  const PinholeModel model;
};
}

#endif
