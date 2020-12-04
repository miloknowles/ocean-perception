#pragma once

namespace bm {
namespace core {

struct PinholeModel final {
  float fx = 0;
  float fy = 0;
  float cx = 0;
  float cy = 0;
  float skew = 1;
  float aspect_ratio = 1;
};

}
}
