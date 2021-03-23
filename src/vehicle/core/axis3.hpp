#pragma once

namespace bm {
namespace core {

enum Axis3 : int
{
  X = 0,
  Y = 1,
  Z = 2
};


inline std::string to_string(Axis3 a)
{
  if (a == Axis3::X) {
    return "X";
  } else if (a == Axis3::Y) {
    return "Y";
  } else {
    return "Z";
  }
}


}
}
