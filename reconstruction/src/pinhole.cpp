#include "reconstruction/pinhole.hpp"

namespace sfm
{
Eigen::Vector3d deproject_pixel_to_point(const PinholeModel & model, int px, int py, double depth)
{
  /// convert pixel to canonical camera plane, can be projected to depth by
  /// multiplication from here
  auto x = (px - model.cx) / model.fx;
  auto y = (py - model.cy) / model.fy;
  return {
    x * depth,
    y * depth,
    depth
  };
}
}
