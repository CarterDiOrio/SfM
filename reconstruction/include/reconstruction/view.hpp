#ifndef INC_GUARD_VIEW_HPP
#define INC_GUARD_VIEW_HPP

#include <memory>

#include <opencv2/opencv.hpp>


namespace sfm
{
class View
{
  /// @brief The image of the view
  cv::Mat image;

  /// @brief The camera matrix
  cv::Mat K;
};
}

#endif
