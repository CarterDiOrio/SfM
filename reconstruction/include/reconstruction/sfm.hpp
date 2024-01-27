#ifndef INC_GUARD_SFM_HPP
#define INC_GUARD_SFM_HPP

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "reconstruction/pinhole.hpp"
#include "reconstruction/map.hpp"
#include "reconstruction/features.hpp"

namespace sfm
{
/// @brief Class that handles incremental 3D reconstruction
class Reconstruction
{
public:
  /// @brief Initializes a reconstruction
  /// @param model the camera model to use.
  Reconstruction(PinholeModel model);

  /// @brief Adds a sequential frame to the reconstruction. It is expected
  /// that the frame comes after the previously added frame
  /// @param frame the frame to add
  /// @param depth the corresponding depth image of the frame
  void add_frame_ordered(const cv::Mat & frame, const cv::Mat & depth);

private:
  const std::shared_ptr<cv::BFMatcher> matcher;
  const std::shared_ptr<cv::ORB> detector;
  const PinholeModel model;

  Map map;


  /// @brief Initializes the 3D reconstruction
  /// @param frame the frame to initialize the reconstruction with
  /// @param depth the corresponding depth image
  void initialize_reconstruction(const cv::Mat & frame, const cv::Mat & depth);
};
}

#endif
