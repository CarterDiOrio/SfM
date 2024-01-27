#include "reconstruction/keyframe.hpp"
#include "reconstruction/pinhole.hpp"
#include <algorithm>

namespace sfm
{
KeyFrame::KeyFrame(
  PinholeModel K, cv::Matx44d T_kw, std::vector<cv::KeyPoint> keypoints,
  std::vector<cv::Mat> descriptions)
: K{K}, T_kw{T_kw}, keypoints{keypoints}, descriptions{descriptions} {}

std::vector<cv::Matx31d> deproject_keypoints(
  const std::vector<cv::KeyPoint> & keypoints, const cv::Mat & depth, const PinholeModel & model)
{
  std::vector<cv::Matx31d> points(keypoints.size());
  std::transform(
    keypoints.begin(), keypoints.end(), points.begin(),
    [&depth, &model](const cv::KeyPoint & kp) {
      auto pixel_depth = depth.at<uint16_t>(
        (int)std::round(kp.pt.x),
        (int)std::round(kp.pt.y));
      return deproject_pixel_to_point(model, kp.pt.x, kp.pt.y, pixel_depth);
    }
  );
  return points;
}

}
