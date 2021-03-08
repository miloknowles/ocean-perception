#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include <glog/logging.h>

#include <opencv2/core/core.hpp>

namespace bm {
namespace core {


class YamlParser {
 public:
  YamlParser() = default;

  // Construct with a path to a .yaml file.
  YamlParser(const std::string& filepath) : filepath_(filepath)
  {
    CHECK(!filepath_.empty()) << "Empty filepath given to YamlParser!" << std::endl;
    fs_.open(filepath_, cv::FileStorage::READ);
    CHECK(fs_.isOpened())
        << "Cannot open file in YamlParser: " << filepath_
        << " (remember that the first line should be: %YAML:1.0)";
    root_node_ = fs_.root();
  }

  // Construct from a YAML node.
  YamlParser(const cv::FileNode& root_node) : root_node_(root_node) {}

  // Retrieve a param from the YAML hierarchy and pass it to output.
  template <class ParamType>
  void GetYamlParam(const std::string& id, ParamType* output) const
  {
    CHECK(!root_node_.empty()) << "GetYamlParam: root_node_ is empty, default constructor probably used" << std::endl;
    GetYamlParamHelper<ParamType>(root_node_, id, output);
  }

  // Get a YAML node relative to the root. This is used for constructing params that are a subtree.
  cv::FileNode GetYamlNode(const std::string& id) const
  {
    CHECK(!root_node_.empty()) << "GetYamlParam: root_node_ is empty, default constructor probably used" << std::endl;
    CHECK(!id.empty()) << "GetYamlParam: empty id given" << std::endl;
    CHECK_NE(id[0], '/') << "Don't use leading slash!" << std::endl;
    const cv::FileNode& file_handle = root_node_[id];
    CHECK_NE(file_handle.type(), cv::FileNode::NONE) << "GetYamlParam: Missing id: " << id.c_str() << std::endl;
    return file_handle;
  }

 private:
  // Helper function that allows recursive param-getting from an given root.
  template <class ParamType>
  void GetYamlParamHelper(const cv::FileNode& root_node, const std::string& id, ParamType* output) const
  {
    CHECK_NOTNULL(output);
    CHECK(!id.empty()) << "GetYamlParam: empty id given" << std::endl;
    CHECK_NE(id[0], '/') << "Don't use leading slash!" << std::endl;

    const size_t slash_idx = id.find_first_of("/");

    // CASE CASE: id is a leaf in the param tree.
    if (slash_idx == std::string::npos) {
      const cv::FileNode& file_handle = root_node[id];
      CHECK_NE(file_handle.type(), cv::FileNode::NONE) << "GetYamlParam: Missing id: " << id.c_str() << std::endl;
      file_handle >> *output;

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
      GetYamlParamHelper<ParamType>(subtree_root, subtree_relative_id, output);
    }
  }

 private:
  std::string filepath_;
  cv::FileStorage fs_;
  cv::FileNode root_node_;
};


}
}
