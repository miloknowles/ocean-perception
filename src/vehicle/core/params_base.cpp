#include "core/params_base.hpp"

namespace bm {
namespace core {


// Construct from root YAML node(s).
void ParamsBase::Parse(const cv::FileNode& root_node,
                       const cv::FileNode& shared_node)
{
  LoadParams(YamlParser(root_node, shared_node));
}


// Construct from a path to YAML file(s).
void ParamsBase::Parse(const std::string& filepath,
                       const std::string& shared_filepath)
{
  LoadParams(YamlParser(filepath, shared_filepath));
}


// Construct from an existing parser.
void ParamsBase::Parse(const YamlParser& parser)
{
  LoadParams(parser);
}


}
}
