#ifndef INC_GUARD_FEATURES_HPP
#define INC_GUARD_FEATURES_HPP

#include "view.hpp"
#include <opencv2/imgproc.hpp>

namespace sfm
{

// template<class T>
// using shared_vector = std::vector<std::shared_ptr<T>>;

struct FeatureView
{
  std::shared_ptr<View> view;
  std::vector<cv::KeyPoint> features;
  cv::Mat feature_descriptors;
};

class FeatureDetector
{
public:
  explicit FeatureDetector(std::shared_ptr<cv::Feature2D> detector);

  /// @brief Finds features in a single view
  /// @param view The view to find features in
  /// @return A FeatureView containing the feature information
  FeatureView detectFeaturesInImage(const std::shared_ptr<View> view) const;

  /// @brief Finds featues in multiple views
  /// @param views the views to find the features in
  /// @return a vector of feature views (see detectFeaturesInImage for details)
  std::vector<FeatureView> detectFeaturesInImages(const std::vector<std::shared_ptr<View>> & views)
  const;

private:
  std::shared_ptr<cv::Feature2D> fdetector;
};
}

#endif
