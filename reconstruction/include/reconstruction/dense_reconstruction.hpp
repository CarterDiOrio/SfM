#ifndef _INC_GUARD_DENSE_RECONSTRUCTION_HPP
#define _INC_GUARD_DENSE_RECONSTRUCTION_HPP

#include "reconstruction/keyframe.hpp"
#include "reconstruction/map.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <iosfwd>
#include <optional>
#include <unordered_map>
#include <Eigen/Dense>
#include <set>
#include <unordered_set>

namespace sfm
{

struct PointInfo
{
  Eigen::Vector3d point;
  Eigen::Vector3i color;
  cv::Point2i pixel;
  KeyFramePtr frame;
  double w;
};

class DenseReconstruction
{
public:
  DenseReconstruction(sfm::Map & map);

  void reconstruct(std::ostream & os);

private:
  /// @brief Contains additional reconstruction information associated with each KeyFrame
  struct ReconstructionInfo
  {
    /// @brief the mask of already reconstructed points
    cv::Mat visibility_mask;
  };


  sfm::Map & map;

  static constexpr double base_line = 50.0 / 1000.0;

  static constexpr double T_cov = 0.2;

  static constexpr double radius_removal_radius = 0.1;

  static constexpr size_t radius_removal_count = 10;

  static constexpr double voxel_size = 1.0;

  static constexpr double T_dist = 5.0;

  static constexpr double min_dist_filter = 0.0;

  Eigen::Matrix3d S;

  /// @brief A map of keyframes to their reconstruction information
  std::unordered_map<KeyFramePtr, ReconstructionInfo> info_index;

  std::vector<PointInfo> process_frame(KeyFramePtr frame);

  std::optional<PointInfo> process_pixel(
    KeyFramePtr frame, int u, int v,
    const std::vector<KeyFramePtr> local_frames);

  std::optional<std::vector<PointInfo>> geometric_check(
    Eigen::Vector3d p_w,
    const std::vector<KeyFramePtr> local_frames);

  // void radius_removal(std::vector<PointInfo> & points);

  std::vector<PointInfo> voxel_filter(const std::vector<PointInfo> & points);

  Eigen::Matrix3d calculate_jacobian(int u, int v, double d, double focal_length);

  Eigen::Matrix3d calculate_covariance(int u, int v, double d, double focal_length);
};
}


#endif
