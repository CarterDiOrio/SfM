#include "reconstruction/keyframe.hpp"

namespace sfm
{
KeyFrame::KeyFrame(
  cv::Matx33d K, cv::Matx44d T_kw, std::vector<cv::KeyPoint> keypoints,
  std::vector<cv::Mat> descriptions)
: K{K}, T_kw{T_kw}, keypoints{keypoints}, descriptions{descriptions} {}
}
