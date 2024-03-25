#include "reconstruction/features.hpp"

#include <algorithm>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>


namespace sfm
{
ORBDetector::ORBDetector(std::unique_ptr<cv::ORB> orb_detector)
: orb_detector{std::move(orb_detector)}
{}

FeatureDetection ORBDetector::detect(const cv::Mat & image) const
{
  FeatureDetection result;
  orb_detector->detectAndCompute(image, cv::noArray(), result.key_points, result.descriptors);
  return result;
}

Features<Frame> uniform_feature_extraction(const Frame & frame)
{

  // split up the image in each frame into a grid to more uniformly detect features
  const int grid_size = 50;

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
    }
  }


  return Features<Frame>{frame, {}, {}};
}


}
