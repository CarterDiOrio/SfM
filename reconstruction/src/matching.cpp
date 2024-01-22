#include "reconstruction/matching.hpp"
#include <execution>
#include <algorithm>
#include <vector>

namespace sfm
{
Match::Match() {}

Match::Match(
  FeatureView view1, FeatureView view2, std::vector<cv::DMatch> matches
)
: view1{view1}, view2{view2}, matches{matches}
{
  for (const auto & dmatch: matches) {
    pts1.emplace_back(view1.features[dmatch.queryIdx].pt);
    pts2.emplace_back(view2.features[dmatch.trainIdx].pt);
  }
}

std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> Match::getMatchingPoints() const
{
  return {pts1, pts2};
}

Matcher::Matcher(std::shared_ptr<cv::DescriptorMatcher> matcher)
: matcher{matcher}
{}

Match Matcher::match(const FeatureView & query, const FeatureView & training) const
{
  std::vector<cv::DMatch> matches;
  matcher->match(query.feature_descriptors, training.feature_descriptors, matches);
  return {
    query,
    training,
    matches
  };
}

std::vector<Match> naiveMatching(std::vector<FeatureView> views, const Matcher & matcher)
{
  //We want combinations not permutations. We don't need to match in both directions.
  //These indicies follow the upper half of the triangular matrix
  std::vector<std::pair<size_t, size_t>> pairs;
  for (size_t i = 1; i < views.size(); i++) {
    for (size_t j = 0; j < i; j++) {
      // std::cout << i << " " << j << "\n";
      pairs.push_back({i, j});
      // matches.emplace_back(matcher.match(views[i], views[j]));
    }
  }

  std::vector<Match> matches(pairs.size());
  std::transform(
    std::execution::par_unseq,
    pairs.begin(), pairs.end(), matches.begin(),
    [&views, &matcher](const auto & pair) {
      return matcher.match(views[pair.first], views[pair.second]);
    }
  );

  return matches;
}

}
