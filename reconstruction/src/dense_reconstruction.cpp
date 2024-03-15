#include "reconstruction/dense_reconstruction.hpp"
#include "reconstruction/features.hpp"
#include "reconstruction/keyframe.hpp"
#include "reconstruction/pinhole.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <cstddef>
#include <opencv2/core/types.hpp>
#include <ranges>
#include <iostream>
#include <execution>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace sfm
{
DenseReconstruction::DenseReconstruction(sfm::Map & map)
: map{map}
{
  S << 0.05, 0, 0,
    0, 0.05, 0,
    0, 0, 10.0;
}


void DenseReconstruction::reconstruct(std::ostream & os)
{
  // initialize Reconstruction info
  std::set<KeyFramePtr> keyframes;
  for (const auto [pair, edge]: map.covisibility_edge) {
    if (edge.shared >= 100) {
      keyframes.insert(pair.first);
      keyframes.insert(pair.second);
    }
  }

  for (const auto & kf: keyframes) {
    info_index.insert(
      {kf, ReconstructionInfo{
          cv::Mat::zeros(kf->img.rows, kf->img.cols, CV_8U)
        }});
  }

  std::vector<PointInfo> points;
  for (const auto & [idx, kf]: std::views::enumerate(keyframes)) {
    std::cout << "Processing keyframe " << idx << "/" << keyframes.size() << std::endl;
    std::vector<PointInfo> frame_points = process_frame(kf);
    points.insert(points.end(), frame_points.begin(), frame_points.end());
  }

  std::cout << "processed all map frames" << std::endl;

  std::cout << "Points before voxel removal: " << points.size() << std::endl;
  std::vector<PointInfo> filtered_points = voxel_filter(points);
  std::cout << "Points after voxel removal: " << filtered_points.size() << std::endl;

  // write points to strema
  for (const auto & p: filtered_points) {
    os << p.point.x() << ' ' << p.point.y() << ' ' << p.point.z() << ' '
       << p.color.z() << ' ' << p.color.y() << ' ' << p.color.x() << std::endl;
  }
}

std::vector<PointInfo> DenseReconstruction::process_frame(KeyFramePtr frame)
{
  std::vector<PointInfo> points;
  auto local_frames = map.get_neighbors(frame, 100);
  // remove local frames that are too close to current frame
  std::vector<KeyFramePtr> local_frames_filtered;
  for (const auto & lf: local_frames) {
    if ((lf->transform().block<3, 1>(
        0,
        3) - frame->transform().block<3, 1>(0, 3)).norm() > min_dist_filter)
    {
      local_frames_filtered.push_back(lf);
    } else {
      Eigen::Vector3d u1, u2, d1, d2;
      u1 << 0, 0, 1;
      u2 << 0, 0, 1;
      d1 = frame->transform().block<3, 3>(0, 0) * u1;
      d2 = lf->transform().block<3, 3>(0, 0) * u2;
      if (d1.dot(d2) < 0.30) { // frames are pointed different directions
        local_frames_filtered.push_back(lf);
      }
    }
  }

  const auto indicies = std::views::iota(0, frame->img.cols * frame->img.rows);
  std::vector<int> indicies_vec(indicies.begin(), indicies.end());
  std::vector<PointInfo> local_points(indicies_vec.size());

  std::for_each(
    std::execution::par_unseq, indicies_vec.begin(), indicies_vec.end(),
    [&frame, this, &local_frames_filtered, &indicies_vec, &local_points](int i) {
      int u = i % frame->img.cols;
      int v = i / frame->img.cols;
      if (info_index.find(frame)->second.visibility_mask.at<uint8_t>(v, u) <= 0) {
        if (frame->get_depth_image().at<uint16_t>(v, u) > 0) {
          std::optional<PointInfo> point = process_pixel(frame, u, v, local_frames_filtered);
          if (point.has_value()) {
            local_points[i] = point.value();
            indicies_vec[i] = -1;
          }
        }
      }
    }
  );

  for (const auto & [idx, p]: std::views::enumerate(local_points)) {
    if (indicies_vec[idx] < 0) {
      points.push_back(p);
    }
  }

  std::cout << "Points before voxel filter: " << points.size() << std::endl;
  std::vector<PointInfo> filtered_points = voxel_filter(points);
  std::cout << "Points after voxel filter: " << filtered_points.size() << std::endl;

  std::cout << "\n";
  return filtered_points;
}

std::optional<PointInfo> DenseReconstruction::process_pixel(
  KeyFramePtr frame, int u, int v,
  const std::vector<KeyFramePtr> local_frames)
{
  const auto depth = frame->get_depth_image().at<uint16_t>(v, u);
  // check for valid depth
  if (depth <= 0) {
    return {};
  }

  Eigen::Matrix3d P = calculate_covariance(u, v, depth, frame->camera_calibration().fx);

  if (P.trace() > T_cov) {
    return {};
  }

  // transform pixel to world
  Eigen::Vector3d p_c = deproject_pixel_to_point(frame->camera_calibration(), u, v, depth);
  Eigen::Vector3d p_w = (frame->transform() * p_c.homogeneous()).head<3>();

  const auto local_ps = geometric_check(p_w, local_frames);
  if (!local_ps.has_value()) {
    return {};
  }

  // set the points as done in the covisiblity mask
  for (const auto & point_info: local_ps.value()) {
    info_index.find(point_info.frame)->second.visibility_mask.at<uint8_t>(point_info.pixel) = 255;
  }

  // combine the found points
  if (local_ps.has_value()) {
    Eigen::Vector3d weighted_point = Eigen::Vector3d::Zero();
    double total = 0;
    for (const auto & point_info: local_ps.value()) {
      weighted_point += point_info.point * (1.0 / point_info.w);
      total += 1.0 / point_info.w;
    }
    p_w = weighted_point / total;
  }


  return PointInfo{
    .point = p_w,
    .color = extract_color(frame->img, cv::Point2i{u, v}),
    .pixel = cv::Point2i{u, v},
    .frame = frame,
    .w = P.trace()
  };
}

std::optional<std::vector<PointInfo>> DenseReconstruction::geometric_check(
  Eigen::Vector3d p_w, const std::vector<KeyFramePtr> local_frames)
{
  if (local_frames.empty()) {
    return {};
  }


  std::vector<Eigen::Vector3d> local_points;
  std::vector<PointInfo> local_points_2d;
  for (const auto & lf: local_frames) {
    auto po = lf->point_in_frame(p_w);
    if (po.has_value()) {
      auto pf = po.value();

      // get depth
      auto depth = lf->get_depth_image().at<uint16_t>(pf);

      // get world point in local frame
      Eigen::Vector3d p_c = deproject_pixel_to_point(lf->camera_calibration(), pf.x, pf.y, depth);

      // get the point in the world frame
      Eigen::Vector3d local_p_w = (lf->transform() * p_c.homogeneous()).head<3>();

      // get covariance
      Eigen::Matrix3d P = calculate_covariance(pf.x, pf.y, depth, lf->camera_calibration().fx);

      if (P.trace() < T_cov) {
        local_points.push_back(local_p_w);
        local_points_2d.push_back(
          {
            .point = local_p_w,
            .color = extract_color(lf->img, pf),
            .pixel = pf,
            .frame = lf,
            .w = P.trace()
          });
      }
    }
  }

  if (local_points.size() < 1) {
    return {};
  }

  size_t close_count = 0;
  for (const auto & p: local_points) {
    if ((p - p_w).norm() < T_dist) {
      close_count++;
    }
  }

  if (close_count < local_points.size() * 0.50) {
    return {};
  }

  return local_points_2d;
}

// void DenseReconstruction::radius_removal(std::vector<PointInfo> & points)
// {
//   const auto radius_filter =
//     [&points, rc = radius_removal_count, rr = radius_removal_radius](const PointInfo & p) {
//       return std::ranges::count_if(
//         points, [&p, rr](const PointInfo & q) {
//           return (p.first - q.first).norm() < rr;
//         }) > rc;
//     };

//   points.erase(
//     std::remove_if(points.begin(), points.end(), radius_filter), points.end());
// }


// https://stackoverflow.com/questions/16792751/hashmap-for-2d3d-coordinates-i-e-vector-of-doubles
struct hashFunc
{
  size_t operator()(const std::tuple<size_t, size_t, size_t> & t) const
  {
    size_t h1 = std::hash<size_t>{}(std::get<0>(t));
    size_t h2 = std::hash<size_t>{}(std::get<1>(t));
    size_t h3 = std::hash<size_t>{}(std::get<2>(t));
    return (h1 ^ (h2 << 1)) ^ (h3 << 2);
  }
};

std::vector<PointInfo> DenseReconstruction::voxel_filter(const std::vector<PointInfo> & points)
{
  std::vector<PointInfo> filtered_points;
  std::unordered_map<std::tuple<size_t, size_t, size_t>, std::vector<PointInfo>,
    hashFunc> voxel_map;

  for (const auto & p: points) {
    std::tuple<size_t, size_t, size_t> voxel = std::make_tuple(
      static_cast<size_t>(p.point.x() / voxel_size),
      static_cast<size_t>(p.point.y() / voxel_size),
      static_cast<size_t>(p.point.z() / voxel_size)
    );

    if (voxel_map.find(voxel) == voxel_map.end()) {
      voxel_map.insert({voxel, std::vector<PointInfo>{p}});
    } else {
      voxel_map.find(voxel)->second.push_back(p);
    }
  }

  for (const auto & [_, v]: voxel_map) {
    if (!v.empty()) {
      filtered_points.push_back(v[0]);
    }
  }

  return filtered_points;
}

Eigen::Matrix3d DenseReconstruction::calculate_jacobian(int u, int v, double d, double focal_length)
{
  Eigen::Matrix3d J;
  J << base_line / d, 0, -u * base_line / (d * d),
    0, base_line / d, -v * base_line / (d * d),
    0, 0, -focal_length * base_line / (d * d);
  return J;
}

Eigen::Matrix3d DenseReconstruction::calculate_covariance(
  int u, int v, double d,
  double focal_length)
{

  // depth = (baseline * focal_length) / disparity
  // disparity * depth = baseline * focal_length
  // disparity = (baseline * focal_length) / depth
  double disparity = (focal_length * base_line) / (d / 1000);

  Eigen::Matrix3d J = calculate_jacobian(u, v, disparity, focal_length);
  return J * S * J.transpose();
}
}
