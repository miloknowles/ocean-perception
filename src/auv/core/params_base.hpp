#pragma once

#include "core/yaml_parser.hpp"

#include "opencv2/core/core.hpp"

namespace bm {
namespace core {


class ParamsBase {
 public:
  // Construct with default params (declared in ParamsDerived).
  ParamsBase() = default;

  // Construct AND parse in one line.
  ParamsBase(const cv::FileNode& root_node) { Parse(root_node); }
  ParamsBase(const std::string& filepath) { Parse(filepath); }

  // Construct from a root YAML node.
  void Parse(const cv::FileNode& root_node)
  {
    LoadFromYamlNode(YamlParser(root_node));
  }

  // Construct from a path to a YAML file.
  void Parse(const std::string& filepath)
  {
    LoadFromYamlNode(YamlParser(filepath));
  }

 protected:
  // This should be implemented in the derived params struct! It gets used when we call Parse()
  // on the derived params struct.
  // https://stackoverflow.com/questions/8951884/undefined-reference-to-typeinfo-for-class
  virtual void LoadFromYamlNode(const YamlParser& parser) = 0;
};

}
}
