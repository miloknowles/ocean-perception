#include <glog/logging.h>

#include "lcm_util/image_subscriber.hpp"
#include "lcm_util/decode_image.hpp"

#include "vision_core/image_util.hpp"

namespace bm {


ImageSubscriber::ImageSubscriber(lcm::LCM& lcm, const std::string& channel, bool expect_shm)
    : channel_(channel)
{
  if (!lcm.good()) {
    LOG(WARNING) << "Failed to initialize LCM" << std::endl;
    return;
  }

  if (expect_shm) {
    lcm.subscribe(channel, &ImageSubscriber::HandleMmf, this);
  } else {
    lcm.subscribe(channel, &ImageSubscriber::Handle, this);
  }

  LOG(INFO) << "ImageSubscriber listening on: " << channel << std::endl;
}


void ImageSubscriber::HandleMmf(const lcm::ReceiveBuffer*,
                                const std::string&,
                                const vehicle::mmf_stereo_image_t* msg)
{
  const bool ok = IsSupported(msg->img_left.encoding, msg->img_left.format, msg->img_left.height, msg->img_left.width)
               && IsSupported(msg->img_left.encoding, msg->img_left.format, msg->img_left.height, msg->img_left.width);
  if (!ok) { return; }

  const std::string mm_filename = msg->img_left.mm_filename;

  // Open the memory-mapped file if not already open.
  if (mapped_file_.get_name() != mm_filename) {
    LOG(INFO) << "First message, opening MMF: " << mm_filename << std::endl;
    mapped_file_ = ipc::file_mapping(mm_filename.c_str(), ipc::read_only);
    mapped_region_ = ipc::mapped_region(mapped_file_, ipc::read_only);
    fbuf_.open(mm_filename, std::ios_base::in | std::ios_base::binary);
  }

  CHECK_EQ(mm_filename, mapped_file_.get_name())
      << "Message mm_filename doesn't match previous. Did the publisher switch?" << std::endl;
  CHECK(fbuf_.is_open()) << "File buffer not open, cannot read" << std::endl;

  // Read raw char buffers for both images.
  const int offl = msg->img_left.offset;
  const int szl = msg->img_left.size;

  if (offl < 0 || szl <= 0) {
    LOG(WARNING) << "Got a data buffer for the left image with negative offset or zero size" << std::endl;
    return;
  }

  lbuf_.resize(szl);
  fbuf_.pubseekpos(offl);
  fbuf_.sgetn(&lbuf_[0], std::streamsize(szl));

  const int offr = msg->img_right.offset;
  const int szr = msg->img_right.size;

  if (offr < 0 || szr <= 0) {
    LOG(WARNING) << "Got a data buffer for the right image with negative offset or zero size" << std::endl;
    return;
  }

  rbuf_.resize(szr);
  fbuf_.pubseekpos(offr);
  fbuf_.sgetn(&rbuf_[0], std::streamsize(szr));

  // https://stackoverflow.com/questions/4254615/how-to-cast-vectorunsigned-char-to-char
  bm::DecodeJPG(msg->img_left, reinterpret_cast<uint8_t*>(lbuf_.data()), left_);
  bm::DecodeJPG(msg->img_right, reinterpret_cast<uint8_t*>(rbuf_.data()), right_);

  core::StereoImage1b out(
      msg->header.timestamp,
      msg->header.seq,
      core::MaybeConvertToGray(std::move(left_)),
      core::MaybeConvertToGray(std::move(right_)));

  for (const StereoImage1bCallback& f : callbacks_1b_) {
    f(out);
  }
}


void ImageSubscriber::Handle(const lcm::ReceiveBuffer*,
                             const std::string&,
                             const vehicle::stereo_image_t* msg)
{
  const bool ok = IsSupported(msg->img_left.encoding, msg->img_left.format, msg->img_left.height, msg->img_left.width)
               && IsSupported(msg->img_left.encoding, msg->img_left.format, msg->img_left.height, msg->img_left.width);
  if (!ok) { return; }

  bm::DecodeJPG(msg->img_left, left_);
  bm::DecodeJPG(msg->img_right, right_);

  core::StereoImage1b out(
      msg->header.timestamp,
      msg->header.seq,
      core::MaybeConvertToGray(std::move(left_)),
      core::MaybeConvertToGray(std::move(right_)));

  for (const StereoImage1bCallback& f : callbacks_1b_) {
    f(out);
  }
}


bool ImageSubscriber::IsSupported(const std::string& encoding,
                                  const std::string& format,
                                  int height, int width)
{
  if (encoding != "jpg") {
    LOG(WARNING)
        << "Unsupported encoding:\n  " << encoding
        << "\nchannel:\n  " << channel_ << std::endl;
    return false;
  }

  if (format != "bgr8" && format != "rgb8" && format != "mono8") {
    LOG(WARNING)
        << "Unsupported format:\n  " << format
        << "\nchannel:\n  " << channel_ << std::endl;
    return false;
  }

  if (width <= 0 || height <= 0) {
    LOG(WARNING)
        << "Zero image dimension!\n  w=" << width << " h=" << height
        << "\nchannel:\n  " << channel_ << std::endl;
  }

  return true;
}

}
