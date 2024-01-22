#include "reconstruction/features.hpp"

namespace sfm
{
FeatureDetector::FeatureDetector(std::shared_ptr<cv::Feature2D> detector)
: fdetector{detector}
{}

FeatureView FeatureDetector::detectFeaturesInImage(const std::shared_ptr<View> view) const
{
  FeatureView feature_view;
  fdetector->detectAndCompute(
    view->image,
    cv::noArray(), feature_view.features, feature_view.feature_descriptors);
  feature_view.view = view;
  return feature_view;
}

std::vector<FeatureView> FeatureDetector::detectFeaturesInImages(
  const std::vector<std::shared_ptr<View>> & views) const
{
  std::vector<FeatureView> feature_views;
  int image = 0;
  for (const auto & view: views) {
    feature_views.emplace_back(detectFeaturesInImage(view));
  }
  return feature_views;
}
}
