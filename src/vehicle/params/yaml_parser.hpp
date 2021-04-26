#pragma once

#include <string>

#include <glog/logging.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/persistence.hpp>

#include "core/eigen_types.hpp"
#include "vision_core/pinhole_camera.hpp"
#include "vision_core/stereo_camera.hpp"

namespace bm {
namespace core {


// Returns whether an id is requesting a "shared" parameter (prefixed by /shared/).
// If so, returns the suffix of the id after /shared/.
static inline bool CheckIfSharedId(const std::string& id, std::string& suffix)
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
             const std::string& shared_filepath = "");

  // Close OpenCV Filestorage IO on destruct.
  ~YamlParser();

  // Construct from a YAML node.
  YamlParser(const cv::FileNode& root_node,
             const cv::FileNode& shared_node,
             const std::string& filepath = "",
             const std::string& shared_filepath = "");

  // Retrieve a param from the YAML hierarchy and pass it to output parameter.
  template <class ParamType>
  void GetParam(const std::string& id, ParamType* output) const
  {
    CHECK_NOTNULL(output);

    // Any id request prefixed with /shared/ is directed to the shared params.
    std::string maybe_suffix;
    if (CheckIfSharedId(id, maybe_suffix)) {
      CHECK(!shared_node_.empty())
          << "GetParam: shared_node_ is empty. Was the parser constructed with a shared node?"
          << "\n  id: " << id << std::endl;
      const cv::FileNode& node = GetNodeHelper(shared_node_, maybe_suffix);
      node >> *output;

    // Anything else is directed to the root params.
    } else {
      CHECK(!root_node_.empty())
          << "GetParam: root_node_ is empty. Was the parser constructed?"
          << "\n  id: " << id << std::endl;
      const cv::FileNode& node = GetNodeHelper(root_node_, id);
      node >> *output;
    }
  }

  // Retrieve a YAML param and return it.
  template <class ParamType>
  ParamType GetParam(const std::string& id) const
  {
    ParamType output;
    GetParam<ParamType>(id, &output);
    return output;
  }

  // Get a YAML node relative to the root. This is used for constructing params that are a subtree.
  cv::FileNode GetNode(const std::string& id) const;

  YamlParser Subtree(const std::string& id) const;

 private:
  // Recursively finds a node with "id", starting from the "root_node".
  cv::FileNode GetNodeHelper(const cv::FileNode& root_node, const std::string& id) const;

  // Returns a string with information about the YAML filepaths, node names, etc. to debug parsing errors.
  std::string HelpfulError(const std::string& id) const;

 private:
  cv::FileStorage fs_, fs_shared_;
  cv::FileNode root_node_;
  cv::FileNode shared_node_;
  std::string filepath_, shared_filepath_;
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

// Convert a YAML list to an Eigen vector type. Returns the vector instead of using output param.
template <typename VectorType>
VectorType YamlToVector(const cv::FileNode& node)
{
  VectorType out;
  YamlToVector<VectorType>(node, out);
  return out;
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

// Parse and return a 4x4 transformation matrix.
Matrix4d YamlToTransform(const cv::FileNode& node);


// Parse and return a string.
std::string YamlToString(const cv::FileNode& node);


// Parse and return an enum (cast from an int to enum type).
template <typename EnumT>
inline EnumT YamlToEnum(const cv::FileNode& node)
{
  CHECK(node.type() != cv::FileNode::NONE);
  int val;
  node >> val;
  return static_cast<EnumT>(val);
}


// Parse and return a PinholeCamera as an output param.
void YamlToCameraModel(const cv::FileNode& node, PinholeCamera& cam);


// Parse and return a StereoCamera as an output param.
void YamlToStereoRig(const cv::FileNode& node,
                    StereoCamera& stereo_rig,
                    Matrix4d& body_T_left,
                    Matrix4d& body_T_right);

}
}
