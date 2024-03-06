#ifndef INC_GUARD_MAPPOINT_HPP
#define INC_GUARD_MAPPOINT_HPP

#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "reconstruction/keyframe.fwd.hpp"

namespace sfm
{

/// @brief Models a 3D point in the map
class MapPoint : public std::enable_shared_from_this<MapPoint>
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

  /// @brief Removes a keyframe from the list of keyframes where this point is visible
  /// @param k_id the id of the keyframe
  void remove_keyframe(std::weak_ptr<KeyFrame> keyframe);

  /// @brief updates the position to be the best observed position in the world
  void update_position();

  /// @brief Sets the position of the map point
  /// @param pos the new position of the map point
  void set_position(Eigen::Vector3d & pos);

  /// @brief Checks if the map point is invalid
  bool is_invalid() const;

  /// @brief Gets the keyframes where this point is visible
  std::vector<std::weak_ptr<KeyFrame>> get_keyframes() const;

  using key_frames_t = std::vector<std::weak_ptr<KeyFrame>>;
  key_frames_t::iterator begin();
  key_frames_t::iterator end();

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

using MapPointPtr = std::shared_ptr<MapPoint>;

}


#endif
