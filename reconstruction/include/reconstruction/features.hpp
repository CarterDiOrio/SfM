#ifndef INC_GUARD_FEATURES_HPP
#define INC_GUARD_FEATURES_HPP

#include <algorithm>
#include <concepts>
#include <memory>
#include <type_traits>
#include <vector>
#include <ranges>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include "reconstruction/frame.hpp"

namespace sfm
{

template<typename T>
concept FeatureFrame = requires(T f) {
  {T(Frame{}, std::vector<cv::KeyPoint>{}, cv::Mat{})}->std::same_as<T>;
  {f.num_features()}->std::same_as<size_t>;
  {f.get_key_point(size_t{})}->std::same_as<cv::KeyPoint>;
  {f.get_descriptor(size_t{})}->std::same_as<cv::Mat>;
} && IsFrame<T>;

template<typename T>
class Features : public T
{
public:
  Features(
    const T & t,
    const std::vector<cv::KeyPoint> & key_points,
    const cv::Mat & descriptors)
  : T{t}, key_points{key_points}, descriptors{descriptors}
  {}

  inline size_t num_features() const {return key_points.size();}
  inline cv::KeyPoint get_key_point(size_t i) const {return key_points[i];}
  inline cv::Mat get_descriptor(size_t i) const {return descriptors.row(i);}

private:
  const std::vector<cv::KeyPoint> key_points;
  const cv::Mat descriptors;
};
static_assert(FeatureFrame<Features<Frame>>);

template<IsFrame T, unsigned int grid_size>
class GridFeatures : public Features<T>
{
public:
  GridFeatures(
    const T & t,
    const std::vector<cv::KeyPoint> & key_points,
    const cv::Mat & descriptors)
  : Features<T>{t, key_points, descriptors}
  {
    // split up the key points into grid buckets
    const size_t num_rows = t.get_height() / grid_size;
    const size_t num_cols = t.get_width() / grid_size;
    grid_buckets.resize(num_rows * num_cols, {});

    for (const auto & [idx, kp] : key_points | std::views::enumerate) {
      const auto r = static_cast<size_t>(kp.pt.y) / grid_size;
      const auto c = static_cast<size_t>(kp.pt.x) / grid_size;
      grid_buckets[r * num_cols + c].push_back(idx);
    }
  }

private:
  /// \brief A 2D grid of key points stored as a 1d vector. Each element vector
  /// contains the indices of the key points in that grid cell
  std::vector<std::vector<size_t>> grid_buckets;
};
static_assert(FeatureFrame<GridFeatures<Frame, 10>>);

struct FeatureDetection
{
  std::vector<cv::KeyPoint> key_points;
  cv::Mat descriptors;
};

/// \brief A concept modeling a basic feature detector interface
template<typename T>
concept FeatureDetector = requires(T c) {
  {c.detect(cv::Mat{})}->std::same_as<FeatureDetection>;
};

class ORBDetector
{
public:
  ORBDetector(std::unique_ptr<cv::ORB> orb_detector);
  FeatureDetection detect(const cv::Mat & image) const;

private:
  std::unique_ptr<cv::ORB> orb_detector;
};
static_assert(FeatureDetector<ORBDetector>);

template<
  FeatureDetector Detector,
  FeatureFrame FeatureFrame,
  int grid_size>
FeatureFrame uniform_feature_extraction(const Frame & frame, Detector detector)
{
  // split up the image in each frame into a grid to more uniformly detect features
  std::vector<cv::KeyPoint> key_points;
  cv::Mat descriptors;

  cv::Mat gray;
  cv::cvtColor(frame.get_color(), gray, cv::COLOR_BGR2GRAY);

  for (int r = 0; r < gray.rows; r += grid_size) {
    for (int c = 0; c < gray.cols; c += grid_size) {
      const cv::Rect rect{
        c, r,
        std::max(grid_size, gray.cols - c), std::max(grid_size, gray.rows - r)
      };
      const auto region = gray(rect);

      // detect features
      const auto features = detector.detect(region);

      std::ranges::for_each(
        features.key_points,
        [r, c, &key_points](cv::KeyPoint & kp) {
          kp.pt.x += c;
          kp.pt.y += r;
          key_points.push_back(kp);
        }
      );
      descriptors.push_back(features.descriptors);
    }
  }

  return FeatureFrame{frame, key_points, descriptors};
}
}


#endif
