#pragma once

namespace bm {
namespace core {


// Represents a duration of time.
class Timedelta {
 public:
  Timedelta(const double sec) : sec_(sec) {}

  double seconds() const { return sec_; }
  double milliseconds() const { return sec_ * 1e3; }
  double microseconds() const { return sec_ * 1e6; }

  Timedelta& operator=(const Timedelta& rhs) {
    sec_ = rhs.seconds();
    return *this;
  }
  bool operator==(const Timedelta& rhs) const { return this->sec_ == rhs.seconds(); }
  bool operator!=(const Timedelta& rhs) const { return !(*this == rhs); }

  friend Timedelta operator+(const Timedelta& lhs, const Timedelta& rhs);
  friend Timedelta operator-(const Timedelta& lhs, const Timedelta& rhs);

  friend bool operator<(const Timedelta& lhs, const Timedelta& rhs) {
    return (lhs.sec_ < rhs.sec_);
  }
  friend bool operator>(const Timedelta& lhs, const Timedelta& rhs) { return rhs < lhs; }
  friend bool operator<=(const Timedelta& lhs, const Timedelta& rhs) { return !(rhs > lhs); }
  friend bool operator>=(const Timedelta& lhs, const Timedelta& rhs) { return !(rhs < lhs); }

 private:
  double sec_ = 0;
};

inline Timedelta operator+(const Timedelta& lhs, const Timedelta& rhs) {
  return Timedelta(lhs.sec_ + rhs.sec_);
}

inline Timedelta operator-(const Timedelta& lhs, const Timedelta& rhs) {
  return Timedelta(lhs.sec_ - rhs.sec_);
}

}
}
