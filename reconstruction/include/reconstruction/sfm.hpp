#ifndef INC_GUARD_SFM_HPP
#define INC_GUARD_SFM_HPP

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "reconstruction/pinhole.hpp"
#include "reconstruction/map.hpp"
#include "reconstruction/keyframe.hpp"

namespace sfm
{

/// @brief Options for the reconstruction
struct ReconstructionOptions
{
  /// @brief The camera model
  PinholeModel model;

  /// @brief The maximum valid depth for a feature (in meters)
  double max_depth;
};


/// @brief Class that handles incremental 3D reconstruction
class Reconstruction
{
public:
  /// @brief Initializes a reconstruction
  /// @param model the camera model to use.
  Reconstruction(ReconstructionOptions options);

  /// @brief Adds a sequential frame to the reconstruction. It is expected
  /// that the frame comes after the previously added frame
  /// @param frame the frame to add
  /// @param depth the corresponding depth image of the frame
  void add_frame_ordered(const cv::Mat & frame, const cv::Mat & depth);

  /// @brief Gets a const reference to the internal map
  /// @return the map
  inline const Map & get_map() const
  {
    return map;
  }

private:
  const std::shared_ptr<cv::BFMatcher> matcher;
  const std::shared_ptr<cv::ORB> detector;
  const PinholeModel model;
  const ReconstructionOptions options;

  std::weak_ptr<KeyFrame> previous_keyframe;

  Map map;

  /// @brief Initializes the 3D reconstruction
  /// @param frame the frame to initialize the reconstruction with
  /// @param depth the corresponding depth image
  void initialize_reconstruction(const cv::Mat & frame, const cv::Mat & depth);

  /// @brief Tracks the features from the previous keyframe and adds the keyframe
  /// @param frame the current frame
  /// @param depth the current depth image
  void track_previous_frame(const cv::Mat & frame, const cv::Mat & depth);

  /// @brief tracks and establishes a key frames link to its local map
  /// @param key_frame the key frame
  void track_local_map(std::shared_ptr<KeyFrame> key_frame);

  /// @brief Performs Perspective-n-Point between the image points and the world points
  /// @param image_points the 2D points in the image
  /// @param world_points the corresponding 3D world peoints
  /// @return the 4x4 transformation matrix of the camera to the world
  Eigen::Matrix4d pnp(
    const std::vector<cv::Point2d> & image_points,
    const std::vector<Eigen::Vector3d> & world_points);
};
}

#endif
