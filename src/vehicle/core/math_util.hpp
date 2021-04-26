#pragma once

#include <cmath>
#include <numeric>

#include <glog/logging.h>

#include "core/eigen_types.hpp"

namespace bm {
namespace core {

static const double DEG_TO_RAD_D = M_PI / 180.0;
static const double RAD_TO_DEG_D = 180.0 / M_PI;


inline int NextEvenInt(int x)
{
  return x + (x % 2);
}

inline int NextOddInt(int x)
{
  return x + (1 - x % 2);
}


// Modulo operation that works for positive and negative integers (like in Python).
// Example 1: WrapInt(4, 3) == 1
// Example 2: WrapInt(-1, 3) == 2
// Example 3: WrapInt(-3, 3) == 0
inline int WrapInt(int k, int N)
{
  if (k >= 0) {
    return k % N;
  }
  if ((-k) % N == 0) {
    return 0;
  }
  return N - ((-k) % N);
}


inline double DegToRad(const double deg)
{
  return deg * DEG_TO_RAD_D;
}

inline double RadToDeg(const double rad)
{
  return rad * RAD_TO_DEG_D;
}

// Grabs the items from v based on indices.
template <typename T>
inline std::vector<T> Subset(const std::vector<T>& v, const std::vector<int>& indices)
{
  std::vector<T> out;
  for (int i : indices) {
    out.emplace_back(v.at(i));
  }
  return out;
}


// Grabs the items from v based on a mask m.
template <typename T>
inline std::vector<T> SubsetFromMask(const std::vector<T>& v, const std::vector<bool>& m, bool invert = false)
{
  CHECK_EQ(v.size(), m.size()) << "Vector and mask must be the same size!" << std::endl;

  std::vector<T> out;
  for (size_t i = 0; i < m.size(); ++i) {
    if (m.at(i) && !invert) {
      out.emplace_back(v.at(i));
    }
  }

  return out;
}


// Grabs the items from v based on a mask m.
template <typename T>
inline std::vector<T> SubsetFromMaskCv(const std::vector<T>& v, const std::vector<uint8_t>& m, bool invert = false)
{
  CHECK_EQ(v.size(), m.size()) << "Vector and mask must be the same size!" << std::endl;

  std::vector<T> out;
  for (size_t i = 0; i < m.size(); ++i) {
    if ((m.at(i) == (uint8_t)1) && !invert) {
      out.emplace_back(v.at(i));
    }
  }

  return out;
}


inline void FillMask(const std::vector<int> indices, std::vector<char>& mask)
{
  std::fill(mask.begin(), mask.end(), false);
  for (int i : indices) { mask.at(i) = true; }
}


// Compute the average value in a vector.
inline double Average(const std::vector<double>& v)
{
  if (v.size() == 0) { return 0.0; }
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

}
}
