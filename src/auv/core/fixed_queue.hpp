#pragma once

#include <queue>
#include <deque>

namespace bm {
namespace core {


// https://stackoverflow.com/questions/56334492/c-create-fixed-size-queue
template <typename T, int MaxLen, typename Container=std::deque<T>>
class FixedQueue : public std::queue<T, Container> {
public:
  void push(const T& value)
  {
    if (this->size() == MaxLen) {
      this->c.pop_front();
    }
    std::queue<T, Container>::push(value);
  }
};


}
}
