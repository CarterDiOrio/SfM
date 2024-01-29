#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>

#include "reconstruction/recording.hpp"
#include "reconstruction/sfm.hpp"
#include "reconstruction/pinhole.hpp"

int main()
{
  std::cout << "Hello, World!" << std::endl;
  cv::Mat K =
    (cv::Mat_<double>(3, 3) << 913.848, 0.0, 642.941, 0.0, 913.602, 371.196, 0.0, 0.0, 1.0);
  sfm::PinholeModel model{913.848, 913.602, 642.941, 371.196};

  sfm::utils::RecordingReader reader{"../recording"};

  sfm::Reconstruction reconstruction{model};

  // for (size_t i = 0; i < 5; i++) {
  //   auto f = reader.read_frames();
  // }

  for (size_t i = 0; i < 280; i++) {
    auto f1 = reader.read_frames();

    const auto & [c, d] = f1.value();
    reconstruction.add_frame_ordered(c, d);
    if (cv::waitKey(100) == 'q') {
      break;
    }
  }

  // auto f1 = reader.read_frames();
  // const auto & [c, d] = f1.value();
  // reconstruction.add_frame_ordered(c, d);

  // for (int i = 0; i < 100; i++) {
  //   reconstruction.add_frame_ordered(c, d);

  // }

  // std::cout << reconstruction.get_map();

  std::ofstream file;
  file.open("./cloud_2.txt");
  file << reconstruction.get_map();

  // while (true) {
  //   auto frames = reader.read_frames();
  //   if (!frames.has_value()) {
  //     std::cout << "no more frames" << "\n";
  //     break;
  //   }

  //   const auto & [color_image, depth_image] = frames.value();

  //   cv::imshow("color", color_image);
  //   cv::imshow("depth", depth_image);

  //   cv::waitKey(33);
  // }

  return 0;
}
