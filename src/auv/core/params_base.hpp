#pragma once

#include "core/yaml_parser.hpp"

#include "opencv2/core/core.hpp"

namespace bm {
namespace core {


class ParamsBase {
 public:
  // Construct with default params (declared in ParamsDerived).
  ParamsBase() = default;

  // Construct from a root YAML node.
  void Parse(const cv::FileNode& root_node)
  {
    LoadParams(YamlParser(root_node));
  }

  // Construct from a path to a YAML file.
  void Parse(const std::string& filepath)
  {
    LoadParams(YamlParser(filepath));
  }

 protected:
  // This should be implemented in the derived params struct! It gets used when we call Parse()
  // on the derived params struct.
  // https://stackoverflow.com/questions/8951884/undefined-reference-to-typeinfo-for-class
  virtual void LoadParams(const YamlParser& parser) = 0;
};

}
}
