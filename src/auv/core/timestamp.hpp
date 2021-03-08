#pragma once

#include <cstdint>
#include <limits>

namespace bm {
namespace core {


// Nanoseconds since Unix epoch (same as EuRoC MAV datasets).
typedef uint64_t timestamp_t;

// This timestamp representation is used by GTSAM.
typedef double seconds_t;


// Convert nanoseconds timestamp to seconds.
inline seconds_t ConvertToSeconds(timestamp_t t)
{
  return 1e-9 * static_cast<double>(t);
}

// Convert seconds to nanoseconds timestamp.
inline timestamp_t ConvertToNanoseconds(seconds_t s)
{
  return static_cast<timestamp_t>(1e9 * s);
}

static const timestamp_t kMinTimestamp = std::numeric_limits<timestamp_t>::min();
static const timestamp_t kMaxTimestamp = std::numeric_limits<timestamp_t>::max();

static const seconds_t kMinSeconds = std::numeric_limits<seconds_t>::min();
static const seconds_t kMaxSeconds = std::numeric_limits<seconds_t>::max();

}
}
