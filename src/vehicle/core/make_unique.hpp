#pragma once

#include <memory>

namespace bm {
namespace core {


// Source: Kimera-VIO
// Add compatibility for c++11's lack of make_unique.
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


}
}
