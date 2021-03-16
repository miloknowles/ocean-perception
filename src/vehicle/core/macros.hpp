#pragma once

#include <memory>


#define MACRO_SHARED_POINTER_TYPEDEFS(TypeName) \
  typedef std::shared_ptr<TypeName> Ptr; \
  typedef std::shared_ptr<const TypeName> ConstPtr;


#define MACRO_DELETE_DEFAULT_CONSTRUCTOR(TypeName) \
  TypeName() = delete;


#define MACRO_DELETE_COPY_CONSTRUCTORS(TypeName) \
  TypeName(const TypeName&) = delete;             \
  void operator=(const TypeName&) = delete;


#define MACRO_PARAMS_STRUCT_CONSTRUCTORS(ClassName) \
  ClassName() : ParamsBase() {} \
  ClassName(const cv::FileNode& root_node, const cv::FileNode& shared_node = cv::FileNode()) : ParamsBase() { Parse(root_node, shared_node); } \
  ClassName(const std::string& filepath, const std::string& shared_filepath = "") : ParamsBase() { Parse(filepath, shared_filepath); } \
  ClassName(const YamlParser& parser) : ParamsBase() { Parse(parser); }
