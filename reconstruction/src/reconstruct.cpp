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
  // sfm::PinholeModel model{913.848, 913.602, 642.941, 371.196};
  sfm::PinholeModel model{718.856, 718.856, 607.1928, 185.2157};
  sfm::utils::RecordingReader reader{"../kitti"};

  sfm::ReconstructionOptions options{
    model,
    50.0
  };
  sfm::Reconstruction reconstruction{options};

  for (size_t i = 0; i < 4500; i++) {
    auto f1 = reader.read_frames();

    if (i % 1 == 0) {
      const auto & [c, d] = f1.value();
      reconstruction.add_frame_ordered(c, d);
      if (cv::waitKey(1) == 'q') {
        break;
      }
    }
  }

  std::ofstream file;
  file.open("./kitti.txt");
  file << reconstruction.get_map();

  return 0;
}
