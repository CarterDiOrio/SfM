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
  sfm::utils::RecordingReader reader{"./stuff/kitti"};

  sfm::ReconstructionOptions options{
    model,
    50.0,
    "/home/cdiorio/ws/winter/stuff/small_voc"
  };
  sfm::Reconstruction reconstruction{options};

  auto frames = reader.read_frames();
  for (size_t i = 0; frames.has_value(); i++) {
    const auto & [c, d] = frames.value();
    if (i % 3 == 0) {
      reconstruction.add_frame_ordered(c, d);
      if (cv::waitKey(1) == 'q') {
        break;
      }
    }

    frames = reader.read_frames();
  }

  std::ofstream file;
  file.open("./kitti.txt");
  file << reconstruction.get_map();

  return 0;
}
