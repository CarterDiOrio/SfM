#ifndef INC_GUARD_MAPPOINT_HPP
#define INC_GUARD_MAPPOINT_HPP

#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "reconstruction/keyframe.fwd.hpp"
#include "reconstruction/mappoint.fwd.hpp"

namespace sfm
{
/// @brief Models a 3D point in the map
class MapPoint
{
public:
  MapPoint(
    cv::Mat descriptor,
    Eigen::Vector3d position,
    std::weak_ptr<KeyFrame> keyframe,
    Eigen::Vector3i color);

  /// @brief Gets the position of the map point in the world frame
  /// @return the position of the map point in world frame
  Eigen::Vector3d position() const;

  /// @brief Gets the ORB binary description of the map point
  /// @return The ORB binary description of the map point
  cv::Mat description() const;


  /// @brief Gets the color of the map point
  /// @return the color
  Eigen::Vector3i get_color() const;

  /// @brief Adds another keyframe where this point is visible from
  /// @param k_id the id of the keyframe
  void add_keyframe(std::weak_ptr<KeyFrame> keyframe);

private:
  /// @brief The orb descriptor that best matches the
  cv::Mat desc;

  /// @brief The 3D position of the point in world space
  Eigen::Vector3d pos;

  /// @brief The RGB color of the world point
  Eigen::Vector3i color;

  /// @brief The ids of the keyframes this point is visible in
  std::vector<std::weak_ptr<KeyFrame>> keyframes;

  /// @brief the minimum distance the point can be observed at according to
  /// orb scale and invariant constraints
  double min{0.0};

  /// @brief The maximum distance the point can be observed at according
  /// to orb scale and invariant constraints
  double max{10.0};
};
}


#endif
