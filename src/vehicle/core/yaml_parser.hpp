#pragma once

#include <string>

#include <glog/logging.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/persistence.hpp>

#include "core/eigen_types.hpp"
#include "core/pinhole_camera.hpp"
#include "core/stereo_camera.hpp"

namespace bm {
namespace core {


// Returns whether an id is requesting a "shared" parameter (prefixed by /shared/).
// If so, returns the suffix of the id after /shared/.
inline bool CheckIfSharedId(const std::string& id, std::string& suffix)
{
  const bool is_shared = id.substr(0, 8) == "/shared/";

  if (is_shared) {
    suffix = id.substr(8, std::string::npos);
  } else {
    suffix = "";
  }

  return is_shared;
}


// Class for parsing a YAML file, using OpenCV's FileStorage module.
class YamlParser {
 public:
  YamlParser() = default;

  // Construct with a path to a .yaml file. Optionally provide a shared_filepath, which points to
  // a shared_params.yaml file.
  YamlParser(const std::string& filepath,
             const std::string& shared_filepath = "")
  {
    CHECK(!filepath.empty()) << "Empty filepath given to YamlParser!" << std::endl;
    fs_.open(filepath, cv::FileStorage::READ);
    CHECK(fs_.isOpened())
        << "Cannot open file in YamlParser: " << filepath
        << " (remember that the first line should be: %YAML:1.0)";
    root_node_ = fs_.root();

    if (shared_filepath.size() > 0) {
      fs_shared_.open(shared_filepath, cv::FileStorage::READ);
      CHECK(fs_shared_.isOpened())
          << "Cannot open file in YamlParser: " << shared_filepath
          << " (remember that the first line should be: %YAML:1.0)";
      shared_node_ = fs_shared_.root();
    }
  }

  // Close OpenCV Filestorage IO on destruct.
  ~YamlParser()
  {
    fs_.release();
    fs_shared_.release();
  }

  // Construct from a YAML node.
  YamlParser(const cv::FileNode& root_node,
             const cv::FileNode& shared_node)
      : root_node_(root_node),
        shared_node_(shared_node) {}

  // Retrieve a param from the YAML hierarchy and pass it to output.
  template <class ParamType>
  void GetYamlParam(const std::string& id, ParamType* output) const
  {
    CHECK_NOTNULL(output);

    // Any id request prefixed with /shared/ is directed to the shared params.
    std::string maybe_suffix;
    if (CheckIfSharedId(id, maybe_suffix)) {
      CHECK(!shared_node_.empty())
          << "GetYamlParam: shared_node_ is empty. Was the parser constructed with a shared node?"
          << "\n  id: " << id << std::endl;
      const cv::FileNode& node = GetYamlNodeHelper(shared_node_, maybe_suffix);
      node >> *output;

    // Anything else is directed to the root params.
    } else {
      CHECK(!root_node_.empty())
          << "GetYamlParam: root_node_ is empty. Was the parser constructed?"
          << "\n  id: " << id << std::endl;
      const cv::FileNode& node = GetYamlNodeHelper(root_node_, id);
      node >> *output;
    }
  }

  // Get a YAML node relative to the root. This is used for constructing params that are a subtree.
  cv::FileNode GetYamlNode(const std::string& id) const
  {
    std::string maybe_suffix;
    if (CheckIfSharedId(id, maybe_suffix)) {
      CHECK(!shared_node_.empty())
          << "GetYamlParam: shared_node_ is empty. Was the parser constructed with a shared node?"
          << "\n  id: " << id << std::endl;
      return GetYamlNodeHelper(shared_node_, maybe_suffix);
    } else {
      CHECK(!root_node_.empty())
          << "GetYamlParam: root_node_ is empty. Was the parser constructed?"
          << "\n  id: " << id << std::endl;
      return GetYamlNodeHelper(root_node_, id);
    }
  }

  YamlParser Subtree(const std::string& id) const
  {
    return YamlParser(GetYamlNode(id), shared_node_);
  }

 private:
  // Recursively finds a node with "id", starting from the "root_node".
  cv::FileNode GetYamlNodeHelper(const cv::FileNode& root_node, const std::string& id) const
  {
    CHECK(!id.empty()) << "GetYamlParam: empty id given" << std::endl;
    CHECK_NE(id[0], '/') << "Don't use leading slash!" << std::endl;

    const size_t slash_idx = id.find_first_of("/");

    // CASE CASE: id is a leaf in the param tree.
    if (slash_idx == std::string::npos) {
      const cv::FileNode& file_handle = root_node[id];
      CHECK_NE(file_handle.type(), cv::FileNode::NONE) << "GetYamlParam: Missing id: " << id.c_str() << std::endl;
      return file_handle;

    // RECURSIVE CASE: id is a map (subtree) with params nested.
    } else {
      CHECK_GE(slash_idx, 1) << "GetYamlParam: should have nonzero substr before /"
          << "id: " << id.c_str() << " slash_idx: " << slash_idx << std::endl;
      const std::string& subtree_root_str = id.substr(0, slash_idx);
      const cv::FileNode& subtree_root = root_node[subtree_root_str];
      CHECK_NE(subtree_root.type(), cv::FileNode::NONE)
          << "GetYamlParam: Missing (subtree) id: " << subtree_root_str.c_str() << std::endl;
      const std::string& subtree_relative_id = id.substr(slash_idx + 1, std::string::npos);
      CHECK(!subtree_relative_id.empty())
          << "GetYamlParam: no recursive id within subtree: " << subtree_root_str
          << "Make sure id doesn't have a trailing slash." << std::endl;
      return GetYamlNodeHelper(subtree_root, subtree_relative_id);
    }
  }

 private:
  cv::FileStorage fs_, fs_shared_;
  cv::FileNode root_node_;
  cv::FileNode shared_node_;
};


// Convert a YAML list to an Eigen vector type.
template <typename VectorType>
void YamlToVector(const cv::FileNode& node, VectorType& vout)
{
  CHECK(node.isSeq()) << "Trying to parse a Vector from a YAML non-sequence" << std::endl;
  CHECK((int)node.size() == vout.rows());
  for (int i = 0; i < vout.rows(); ++i) {
    vout(i) = node[i];
  }
}


template <typename MatrixType>
void YamlToMatrix(const cv::FileNode& node, MatrixType& mout)
{
  const cv::FileNode& rows_node = node["rows"];
  const cv::FileNode& cols_node = node["cols"];
  CHECK(rows_node.type() != cv::FileNode::NONE && cols_node.type() != cv::FileNode::NONE)
      << "YamlToMatrix: required 'rows' or 'cols' attribute not found" << std::endl;
  int rows, cols;
  rows_node >> rows;
  cols_node >> cols;

  CHECK(rows == mout.rows() && cols == mout.cols())
      << "YamlToMatrix: Output matrix did not match YAML rows/cols" << std::endl;

  const cv::FileNode& data_node = node["data"];
  CHECK(data_node.type() != cv::FileNode::NONE)
      << "YamlToMatrix: 'data' node not found" << std::endl;

  CHECK(data_node.isSeq()) << "YamlToMatrix: 'data' node must contain a sequence" << std::endl;
  CHECK((int)data_node.size() == (rows * cols)) << "YamlToMatrix: wrong data size" << std::endl;

  // NOTE(milo): Data should be stored in ROW-MAJOR order as a vector.
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      mout(r, c) = data_node[r*cols + c];
    }
  }
}


inline std::string YamlToString(const cv::FileNode& node)
{
  CHECK(node.type() != cv::FileNode::NONE);
  cv::String cvstr;
  node >> cvstr;
  return std::string(cvstr.c_str());
}


template <typename EnumT>
inline EnumT YamlToEnum(const cv::FileNode& node)
{
  CHECK(node.type() != cv::FileNode::NONE);
  int val;
  node >> val;
  return static_cast<EnumT>(val);
}


inline void YamlToCameraModel(const cv::FileNode& node, PinholeCamera& cam)
{
  const cv::FileNode& h_node = node["image_height"];
  const cv::FileNode& w_node = node["image_width"];
  const cv::FileNode& camera_model_node = node["camera_model"];

  int h, w;
  h_node >> h;
  w_node >> w;
  CHECK_GT(h, 0) << "Height must be > 0" << std::endl;
  CHECK_GT(w, 0) << "Width must be > 0" << std::endl;

  CHECK(camera_model_node.type() != cv::FileNode::NONE &&
        YamlToString(camera_model_node) == "pinhole")
        << "Must contain a field camera_model: pinhole" << std::endl;

  const cv::FileNode& intrinsics_node = node["intrinsics"];
  CHECK(intrinsics_node.isSeq() && intrinsics_node.size() == 4)
      << "intrinsics must contain (4) values: fx, fy, cx, cy" << std::endl;

  const double fx = intrinsics_node[0];
  const double fy = intrinsics_node[1];
  const double cx = intrinsics_node[2];
  const double cy = intrinsics_node[3];

  const cv::FileNode& distort_node = node["distortion_coefficients"];
  CHECK(distort_node.isSeq() && distort_node.size() > 0)
      << "Expected distortion coefficients" << std::endl;

  LOG_IF(WARNING, (double)distort_node[0] > 0) << "WARNING: distortion_coefficients are nonzero, but we don't handle undistortion yet" << std::endl;

  cam = PinholeCamera(fx, fy, cx, cy, h, w);
}


inline void YamlToStereoRig(const cv::FileNode& node,
                            StereoCamera& stereo_rig,
                            Matrix4d& body_T_left,
                            Matrix4d& body_T_right)
{
  const cv::FileNode& cam_left_node = node["camera_left"];
  const cv::FileNode& cam_right_node = node["camera_right"];
  CHECK(cam_left_node.type() != cv::FileNode::NONE) << "camera_left node not found" << std::endl;
  CHECK(cam_right_node.type() != cv::FileNode::NONE) << "camera_right node not found" << std::endl;

  PinholeCamera cam_left, cam_right;
  YamlToCameraModel(cam_left_node, cam_left);
  YamlToCameraModel(cam_right_node, cam_right);

  YamlToMatrix<Matrix4d>(cam_left_node["body_T_cam"], body_T_left);
  YamlToMatrix<Matrix4d>(cam_right_node["body_T_cam"], body_T_right);
  CHECK(body_T_left(3, 3) == 1.0) << "body_T_left is invalid" << std::endl;
  CHECK(body_T_right(3, 3) == 1.0) << "body_T_right is invalid" << std::endl;

  const Matrix4d left_T_right = body_T_left.inverse() * body_T_right;
  stereo_rig = StereoCamera(cam_left, cam_right, Transform3d(left_T_right));
}


}
}
