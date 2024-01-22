#include "reconstruction/verification.hpp"
#include <opencv2/calib3d.hpp>

namespace sfm
{
std::optional<VerifiedMatch> verify(const Match & match, size_t inlier_threshold)
{
  const auto & [E, inlier_mask] = computeEssential(match);

  // count number of inliers
  size_t num_inliers = std::count_if(
    inlier_mask.begin(), inlier_mask.end(),
    [](const int & i) {return i > 0;});

  // didn't meet our verification threshold reject
  if (num_inliers < inlier_threshold) {
    return {};
  }

  // met inlier threshold
  return VerifiedMatch{
    match,
    E,
    num_inliers,
    inlier_mask
  };
}

std::pair<cv::Mat, std::vector<char>> computeEssential(const Match & match)
{
  const auto & [pts1, pts2] = match.getMatchingPoints();
  std::vector<char> inlier_mask;
  auto E = cv::findEssentialMat(
    pts1, pts2, match.view1.view->K, cv::USAC_DEFAULT, 0.99, 1.0,
    inlier_mask);

  return {E, inlier_mask};
}
}
