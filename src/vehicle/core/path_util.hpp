#pragma once

#include <cstdlib>

#include "core/file_utils.hpp"

namespace bm {
namespace core {


inline std::string vehicle_path(const std::string& subdir = "")
{
  const char* path = std::getenv("BM_VEHICLE_DIR");

  if (path == nullptr) {
    throw std::runtime_error("Environment does not contain $BM_VEHICLE_DIR. Did you source setup.bash?");
  }

  return Join(std::string(path), subdir);
}


inline std::string config_path(const std::string& subdir)
{
  return vehicle_path(Join("config", subdir));
}


inline std::string src_path(const std::string& subdir)
{
  return vehicle_path(Join("src", subdir));
}


inline std::string sandbox_path(const std::string& subdir)
{
  return src_path(Join("sandbox", subdir));
}


inline std::string tools_path(const std::string& subdir)
{
  return src_path(Join("tools", subdir));
}


}
}
