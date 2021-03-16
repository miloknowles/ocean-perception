#pragma once

#include <map>

namespace bm {
namespace vio {


// Stores an ordered "history" of items based on a key (could be a timestamp or an id).
template <typename Key, typename Item>
class ItemHistory final {
 public:
  typedef std::map<Key, Item> Map;

  ItemHistory() = default;

  Key NewestKey() const
  {
    CHECK(!Empty()) << "Cannot get NewestKey() for empty history" << std::endl;
    return key_to_item_map_.rbegin()->first;
  }

  Key OldestKey() const
  {
    CHECK(!Empty()) << "Cannot get OldestKey() for empty history" << std::endl;
    return key_to_item_map_.begin()->first;
  }

  size_t Size() const { return key_to_item_map_.size(); }
  bool Empty() const { return key_to_item_map_.empty(); }
  bool Exists(Key k) const { return key_to_item_map_.count(k) > 0; }

  // Return the item at key k.
  Item at(Key k) const
  {
    if (key_to_item_map_.count(k) == 0) {
      if (key_to_item_map_.size() == 0) {
        throw std::runtime_error("Tried to at(key) but the map is empty!");
      } else {
        const std::string msg = std::string("Tried to at(key) that doesn't exist.") +
                                std::string(" key=") + std::to_string(k) +
                                std::string(" oldest=") + std::to_string(OldestKey()) +
                                std::string(" newest=") + std::to_string(NewestKey());
        throw std::runtime_error(msg);
      }
    }
    return key_to_item_map_.at(k);
  }

  void Update(Key k, const Item& item) { key_to_item_map_.emplace(k, std::move(item)); }

  // Discard all items *before* (but not equal to) the key k.
  void DiscardBefore(Key k)
  {
    // https://stackoverflow.com/questions/22874535/dependent-scope-need-typename-in-front
    typename Map::const_iterator it_first_item_gt_k = key_to_item_map_.lower_bound(k);
    key_to_item_map_.erase(key_to_item_map_.begin(), it_first_item_gt_k);
  }

 private:
  Map key_to_item_map_;
};


}
}
