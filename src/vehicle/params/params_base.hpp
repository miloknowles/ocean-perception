#pragma once

#include <string>
#include <opencv2/core/persistence.hpp>

#include "params/yaml_parser.hpp"

namespace bm {
namespace core {


class ParamsBase {
 public:
  // Construct with default params (declared in ParamsDerived).
  ParamsBase() = default;

  // Construct from root YAML node(s).
  void Parse(const cv::FileNode& root_node,
             const cv::FileNode& shared_node = cv::FileNode());

  // Construct from a path to YAML file(s).
  void Parse(const std::string& filepath,
             const std::string& shared_filepath = "");

  // Construct from an existing parser.
  void Parse(const YamlParser& parser);

 protected:
  // This should be implemented in the derived params struct! It gets used when we call Parse()
  // on the derived params struct.
  // https://stackoverflow.com/questions/8951884/undefined-reference-to-typeinfo-for-class
  virtual void LoadParams(const YamlParser& parser) = 0;
};

}
}
