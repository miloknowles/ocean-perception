#pragma once

#include <string>
#include <vector>
#include <boost/filesystem.hpp>

namespace bm {
namespace core {

namespace fs = boost::filesystem;


// Reads all filenames in a directory and returns the number found.
inline int FilenamesInDirectory(const std::string& dir, std::vector<std::string>& out, bool sort = false)
{
  for (auto item = fs::directory_iterator(fs::path(dir)); item != fs::directory_iterator(); ++item) {
    // Skip directories.
    if (fs::is_directory(item->path())) { continue; }
    out.emplace_back(item->path().string());
  }

  if (sort) {
    std::sort(out.begin(), out.end());
  }

  return static_cast<int>(out.size());
}

inline std::string Join(const std::string& a, const std::string& b)
{
  return (fs::path(a) / fs::path(b)).string();
}


inline bool Exists(const std::string& fname)
{
  return fs::exists(fname);
}


}
}
