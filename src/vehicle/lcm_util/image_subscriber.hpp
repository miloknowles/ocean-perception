#pragma once

#include <vector>
#include <fstream>
#include <iostream>

#include <opencv2/core/mat.hpp>

#include <lcm/lcm-cpp.hpp>

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include "core/timestamp.hpp"
#include "vision_core/stereo_image.hpp"

#include "vehicle/stereo_image_t.hpp"
#include "vehicle/mmf_stereo_image_t.hpp"

namespace bm {

namespace ipc = boost::interprocess;


typedef std::function<void(const core::StereoImage1b&)> StereoImage1bCallback;
typedef std::function<void(const core::StereoImage3b&)> StereoImage3bCallback;


class ImageSubscriber final {
 public:
  // Create an image subscriber that listens on "channel". If expect_shm is true, this subscriber
  // will expect to receive a memory-mapped image (mmf_stereo_image_t).
  // NOTE(milo): An LCM handle must be passed in! Messages are only received if lcm.Spin() is
  // constantly called, which should happen in whatever process owns this ImageSubscriber.
  ImageSubscriber(lcm::LCM& lcm, const std::string& channel, bool expect_shm = true);

  // Register a callback function that will be called for each decoded image.
  void RegisterCallback(StereoImage1bCallback f) { callbacks_1b_.emplace_back(f); }

 private:
  void HandleMmf(const lcm::ReceiveBuffer*,
                const std::string&,
                const vehicle::mmf_stereo_image_t* msg);

  void Handle(const lcm::ReceiveBuffer*,
              const std::string&,
              const vehicle::stereo_image_t* msg);

  // Validates the image metadata to make sure it can be decoded.
  bool IsSupported(const std::string& encoding,
                   const std::string& format,
                   int height,
                   int width);

 private:
  std::string channel_;

  cv::Mat left_;
  cv::Mat right_;

  ipc::file_mapping mapped_file_;
  ipc::mapped_region mapped_region_;

  // https://stackoverflow.com/questions/604431/c-reading-unsigned-char-from-file-stream
  std::basic_filebuf<char> fbuf_;
  std::vector<char> lbuf_, rbuf_;

  std::vector<StereoImage1bCallback> callbacks_1b_;
};


}
