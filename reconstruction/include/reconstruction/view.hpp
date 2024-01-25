#ifndef INC_GUARD_VIEW_HPP
#define INC_GUARD_VIEW_HPP

#include <memory>

#include <opencv2/opencv.hpp>


namespace sfm
{
struct View
{
  View(cv::Mat image, cv::Mat K, size_t id)
  : image{image}, K{K}, image_id{id} {}

  /// @brief The image of the view
  cv::Mat image;

  /// @brief The camera matrix
  cv::Mat K;

  size_t image_id;
};
}

#endif
