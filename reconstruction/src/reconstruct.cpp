#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>

#include "reconstruction/recording.hpp"

int main()
{
  std::cout << "Hello, World!" << std::endl;
  cv::Mat K =
    (cv::Mat_<double>(3, 3) << 913.848, 0.0, 642.941, 0.0, 913.602, 371.196, 0.0, 0.0, 1.0);


  sfm::utils::RecordingReader reader{"../recording"};

  while (true) {
    auto frames = reader.read_frames();
    if (!frames.has_value()) {
      std::cout << "no more frames" << "\n";
      break;
    }

    const auto & [color_image, depth_image] = frames.value();

    cv::imshow("color", color_image);
    cv::imshow("depth", depth_image);

    cv::waitKey(33);
  }

  return 0;
}
