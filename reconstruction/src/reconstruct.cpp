#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>

#include "reconstruction/dense_reconstruction.hpp"
#include "reconstruction/recording.hpp"
#include "reconstruction/sfm.hpp"
#include "reconstruction/pinhole.hpp"

int main()
{
  sfm::PinholeModel model{913.848, 913.602, 642.941, 371.196};
  sfm::utils::RecordingReader reader{"./stuff/ish", 20};
  // sfm::PinholeModel model{718.856, 718.856, 607.1928, 185.2157};
  // sfm::utils::RecordingReader reader{"./stuff/nader"};

  sfm::ReconstructionOptions options{
    model,
    6000.0,
    "/home/cdiorio/ws/winter/stuff/voc_lab",
  };
  sfm::Reconstruction reconstruction{options};

  auto frames = reader.read_frames();
  for (size_t i = 0; frames.has_value(); i++) {
    const auto & [c, d] = frames.value();
    if (i % 5 == 0) {
      reconstruction.add_frame_ordered(c, d);
      if (cv::waitKey(1) == 'q') {
        break;
      }
    }

    frames = reader.read_frames();
  }

  reconstruction.get_map().local_bundle_adjustment(model);

  // for (const auto & kf: reconstruction.get_map().keyframes) {
  //   const auto tf = kf->transform();
  //   std::cout << tf.row(0) << ' ' << tf.row(1) << ' ' << tf.row(2) << std::endl;
  // }


  std::ofstream file;
  file.open("./kitti.txt");
  file << reconstruction.get_map();

  std::ofstream dense_file;
  dense_file.open("./dense.txt");
  auto map = reconstruction.get_map();
  sfm::DenseReconstruction dense_reconstruction{map};
  dense_reconstruction.reconstruct(dense_file);

  return 0;
}
