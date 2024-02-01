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

  sfm::ReconstructionOptions options{
    model,
    6000.0
  };
  sfm::Reconstruction reconstruction{options};

  for (size_t i = 0; i < 280; i++) {
    auto f1 = reader.read_frames();

    const auto & [c, d] = f1.value();
    reconstruction.add_frame_ordered(c, d);
    if (cv::waitKey(100) == 'q') {
      break;
    }
  }

  std::ofstream file;
  file.open("./cloud_2.txt");
  file << reconstruction.get_map();

  return 0;
}
