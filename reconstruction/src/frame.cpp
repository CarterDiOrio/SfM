#include "reconstruction/frame.hpp"

namespace sfm
{
Frame::Frame(
  const model::PinholeModel & model,
  cv::Mat color,
  cv::Mat depth)
: model{model},
  color{color},
  depth{depth}
{}

}
