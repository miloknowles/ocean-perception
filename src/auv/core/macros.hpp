#pragma once

#include <memory>

// Adapted from Kimera-VIO
#define MACRO_POINTER_TYPEDEFS(TypeName)                 \
  typedef std::shared_ptr<TypeName> Ptr;                  \
  typedef std::shared_ptr<const TypeName> ConstPtr;       \
  typedef std::unique_ptr<TypeName> UniquePtr;            \
  typedef std::unique_ptr<const TypeName> ConstUniquePtr; \
  typedef std::weak_ptr<TypeName> WeakPtr;                \
  typedef std::weak_ptr<const TypeName> WeakConstPtr;     \
  void definePointerTypedefs##__FILE__##__LINE__(void)


// Adapted from Kimera-VIO.
#define MACRO_DELETE_COPY_CONSTRUCTORS(TypeName) \
  TypeName(const TypeName&) = delete;             \
  void operator=(const TypeName&) = delete


#define MACRO_PARAMS_BASE_CONSTRUCTORS(ClassName) \
  ClassName(const cv::FileNode& root_node) : ParamsBase(root_node) {} \
  ClassName(const std::string& filepath) : ParamsBase(filepath) {}
