#ifndef BUNDLE_ADJUST_HPP_GUARD_
#define BUNDLE_ADJUST_HPP_GUARD_

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "reconstruction/pinhole.hpp"
#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>


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
    T p[3];
    ceres::AngleAxisRotatePoint(camera_params, point, p);
    p[0] += camera_params[3];
    p[1] += camera_params[4];
    p[2] += camera_params[5];

    const auto projected_x = (p[0] / p[2]) * model.fx + model.cx;
    const auto projected_y = (p[1] / p[2]) * model.fy + model.cy;

    residuals[0] = observed_x - projected_x;
    residuals[1] = observed_y - projected_y;

    // const auto mag = sqrt(residuals[0] * residuals[0] + residuals[1] * residuals[1]);


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
