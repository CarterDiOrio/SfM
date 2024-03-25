#ifndef INC_GUARD_FRAME_HPP
#define INC_GUARD_FRAME_HPP

#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include "reconstruction/coordinate_frame.hpp"
#include "reconstruction/pinhole.hpp"


namespace sfm
{
class Frame
{
public:
  Frame() = default;
  Frame(
    const model::PinholeModel & model,
    cv::Mat color,
    cv::Mat depth);

  inline model::PinholeModel get_model() const {return model;}
  inline cv::Mat get_color() const {return color;}
  inline cv::Mat get_depth() const {return depth;}
  inline int get_width() const {return color.cols;}
  inline int get_height() const {return color.rows;}

private:
  model::PinholeModel model;
  cv::Mat color;
  cv::Mat depth;
};

template<typename T>
concept IsFrame = std::is_base_of<Frame, T>::value;


using CameraFrame = CoordinateFrame<Frame>;

inline auto project_to(const CameraFrame & frame, const Eigen::Vector3d & point)
{
  return model::model_to_eigen(frame.get_model()) * frame.point_to_frame(point);
}

inline uint16_t get_depth(const CameraFrame & frame, const cv::Point2d & point)
{
  return frame.get_depth().at<uint16_t>(point);
}

inline uint16_t get_depth(const CameraFrame & frame, const Eigen::VectorXd & point)
{
  return get_depth(frame, project_to(frame, point.head<3>()));
}

}
#endif
