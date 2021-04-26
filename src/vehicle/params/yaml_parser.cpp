#include "params/yaml_parser.hpp"

namespace bm {
namespace core {


// Construct with a path to a .yaml file. Optionally provide a shared_filepath, which points to
// a shared_params.yaml file.
YamlParser::YamlParser(const std::string& filepath,
                       const std::string& shared_filepath)
    : filepath_(filepath),
      shared_filepath_(shared_filepath)
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
YamlParser::~YamlParser()
{
  fs_.release();
  fs_shared_.release();
}


// Construct from a YAML node.
YamlParser::YamlParser(const cv::FileNode& root_node,
                       const cv::FileNode& shared_node,
                       const std::string& filepath,
                       const std::string& shared_filepath)
    : root_node_(root_node),
      shared_node_(shared_node),
      filepath_(filepath),
      shared_filepath_(shared_filepath) {}


// Get a YAML node relative to the root. This is used for constructing params that are a subtree.
cv::FileNode YamlParser::GetNode(const std::string& id) const
{
  std::string maybe_suffix;
  if (CheckIfSharedId(id, maybe_suffix)) {
    CHECK(!shared_node_.empty()) << HelpfulError(id) << " Was the parser constructed with a shared node?";
    return GetNodeHelper(shared_node_, maybe_suffix);
  } else {
    CHECK(!root_node_.empty()) << HelpfulError(id) << " GetParam: root_node_ is empty. Was the parser constructed?";
    return GetNodeHelper(root_node_, id);
  }
}


YamlParser YamlParser::Subtree(const std::string& id) const
{
  // Pass in the filepath and shared_filepath for debugging purposes.
  return YamlParser(GetNode(id), shared_node_, filepath_, shared_filepath_);
}


// Recursively finds a node with "id", starting from the "root_node".
cv::FileNode YamlParser::GetNodeHelper(const cv::FileNode& root_node, const std::string& id) const
{
  CHECK(!id.empty()) << HelpfulError(id) << " GetParam: empty id given" << std::endl;
  CHECK_NE(id[0], '/') << HelpfulError(id) << " Don't use leading slash!" << std::endl;

  const size_t slash_idx = id.find_first_of("/");

  // CASE CASE: id is a leaf in the param tree.
  if (slash_idx == std::string::npos) {
    const cv::FileNode& file_handle = root_node[id];
    CHECK_NE(file_handle.type(), cv::FileNode::NONE) << HelpfulError(id) << " GetParam: Missing id: " << id << std::endl;
    return file_handle;

  // RECURSIVE CASE: id is a map (subtree) with params nested.
  } else {
    CHECK_GE(slash_idx, 1) << HelpfulError(id) << " GetParam: should have nonzero substr before /"
        << "id: " << id.c_str() << " slash_idx: " << slash_idx << std::endl;
    const std::string& subtree_root_str = id.substr(0, slash_idx);
    const cv::FileNode& subtree_root = root_node[subtree_root_str];
    CHECK_NE(subtree_root.type(), cv::FileNode::NONE) << HelpfulError(id)
        << " GetParam: Missing (subtree) id: " << subtree_root_str.c_str() << std::endl;
    const std::string& subtree_relative_id = id.substr(slash_idx + 1, std::string::npos);
    CHECK(!subtree_relative_id.empty()) << HelpfulError(id)
        << " GetParam: no recursive id within subtree: " << subtree_root_str
        << " Make sure id doesn't have a trailing slash." << std::endl;
    return GetNodeHelper(subtree_root, subtree_relative_id);
  }
}


std::string YamlParser::HelpfulError(const std::string& id) const
{
  std::stringstream ss;
  ss << "\n" << "** YAML PARSING ERROR **" << std::endl;
  ss << " Error while trying to parse id: " << id << std::endl;
  ss << " Params file:    " << filepath_ << std::endl;
  ss << " Shared file:    " << shared_filepath_ << std::endl;
  ss << " Root node:      " << root_node_.name() << std::endl;
  ss << " Shared node:    " << shared_node_.name() << std::endl;
  return ss.str();
}


std::string YamlToString(const cv::FileNode& node)
{
  CHECK(node.type() != cv::FileNode::NONE);
  cv::String cvstr;
  node >> cvstr;
  return std::string(cvstr.c_str());
}


void YamlToCameraModel(const cv::FileNode& node, PinholeCamera& cam)
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


void YamlToStereoRig(const cv::FileNode& node,
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

  body_T_left = YamlToTransform(cam_left_node["body_T_cam"]);
  body_T_right = YamlToTransform(cam_right_node["body_T_cam"]);

  const Matrix4d left_T_right = body_T_left.inverse() * body_T_right;
  stereo_rig = StereoCamera(cam_left, cam_right, Transform3d(left_T_right));
}


Matrix4d YamlToTransform(const cv::FileNode& node)
{
  Matrix4d T;
  YamlToMatrix(node, T);

  CHECK_EQ(1.0, T(3, 3)) << "Transform matrix should have 1.0 as lower right entry" << std::endl;

  const Matrix3d& R = T.block<3, 3>(0, 0);
  CHECK(R.isApprox(R.transpose())) << "Rotation matrix should be symmetric" << std::endl;

  return T;
}


}
}
