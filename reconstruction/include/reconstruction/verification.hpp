#ifndef INC_GUARD_VERIFICATION_HPP
#define INC_GUARD_VERIFICATION_HPP

#include <optional>

#include "matching.hpp"

namespace sfm
{
struct VerifiedMatch
{
  const Match match;
  const cv::Mat model;
  const unsigned long num_inliers;
  const std::vector<char> inlier_mask;
};

std::optional<VerifiedMatch> verify(const Match & match, size_t inlier_threshold);

std::pair<cv::Mat, std::vector<char>> computeEssential(const Match & match);
}

#endif
