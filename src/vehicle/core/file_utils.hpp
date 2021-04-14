#pragma once

#include <string>
#include <vector>

namespace bm {
namespace core {


// Reads all filenames in a directory and returns the number found.
int FilenamesInDirectory(const std::string& dir, std::vector<std::string>& out, bool sort = false);


std::string Join(const std::string& a, const std::string& b);


bool Exists(const std::string& fname);


bool mkdir(const std::string& folder, bool exist_ok = true);


bool rmdir(const std::string& folder);


}
}
