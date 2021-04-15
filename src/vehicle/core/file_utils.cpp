#include <string>
#include <vector>
#include <boost/filesystem.hpp>

#include "core/file_utils.hpp"

namespace bm {
namespace core {

namespace fs = boost::filesystem;


// Reads all filenames in a directory and returns the number found.
int FilenamesInDirectory(const std::string& dir, std::vector<std::string>& out, bool sort)
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


std::string Join(const std::string& a, const std::string& b)
{
  return (fs::path(a) / fs::path(b)).string();
}


bool Exists(const std::string& fname)
{
  return fs::exists(fname);
}


bool mkdir(const std::string& folder, bool exist_ok)
{
  if (!exist_ok && Exists(folder)) {
    throw std::runtime_error("Trying to make a directory that already exists: " + folder);
  }

  return fs::create_directory(folder);
}


bool rmdir(const std::string& folder)
{
  return fs::remove_all(folder);
}


}
}
