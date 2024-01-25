#ifndef INC_GUARD_MATCHING_HPP
#define INC_GUARD_MATCHING_HPP

#include "features.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>

namespace sfm
{
class Match
{
public:
  Match();
  Match(FeatureView view1, FeatureView view2, std::vector<cv::DMatch> matches);

  FeatureView view1;
  FeatureView view2;
  std::vector<cv::DMatch> matches;

  /// @brief Extracts the corresponding points from each image.
  /// @return a pair of two vectors where the corresponding indicies are the corresponding points
  std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> getMatchingPoints() const;

private:
  std::vector<cv::Point2d> pts1, pts2;
};

class Matcher
{
public:
  /// @brief Creates a matcher object
  /// @param matcher the OpenCV matcher to use
  explicit Matcher(std::shared_ptr<cv::DescriptorMatcher> matcher);

  /// @brief Matches features between the query and training views
  /// @param query the query view
  /// @param training the training view
  /// @return A match object describing the match
  Match match(const FeatureView & query, const FeatureView & training) const;

private:
  std::shared_ptr<cv::DescriptorMatcher> matcher;
};

/// @brief Naively brute force matches by comparing each view to every other view
/// @param views the views to compare
/// @returns the a vector of matches for each view pair
std::vector<Match> naiveMatching(std::vector<FeatureView> views, const Matcher & matcher);

}

#endif
