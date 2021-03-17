#include <gtest/gtest.h>
#include <glog/logging.h>

#include <queue>


struct ExampleData
{
  explicit ExampleData(int id, const std::string& name)
      : id_(id), name_(name)
  {
    LOG(INFO) << "ExampleData() constructor" << std::endl;
  }

  ExampleData() = delete;

  ~ExampleData()
  {
    LOG(INFO) << "~ExampleData() destructor" << std::endl;
  }

  // Delete copy constructor.
  ExampleData(const ExampleData&) = delete;

  // Declare a move constructor.
  ExampleData(ExampleData&& other)
  {
    LOG(INFO) << "ExampleData() move constructor" << std::endl;
    std::swap(id_, other.id_);
    std::swap(name_, other.name_);
  }

  int id_;
  std::string name_;

  void Print() const
  {
    LOG(INFO) << "ExampleData id=" << id_ << " name=" << name_ << std::endl;
  }
};


TEST(MoveTest, Test1)
{
  // Call the normal constructor.
  const ExampleData a(0, "milo");
  a.Print();

  // Call the move constructor.
  const ExampleData b(std::move(ExampleData(1, "milo")));
  b.Print();
}


TEST(MoveTest, TestQueue)
{
  // In this case it looks like the queue uses a move constructor by default.
  LOG(INFO) << "PUSH" << std::endl;
  std::queue<ExampleData> q;
  q.push(ExampleData(0, "milo"));
  q.push(std::move(ExampleData(1, "milo")));

  LOG(INFO) << "POP" << std::endl;
  ExampleData a = std::move(q.front());
  q.pop();
}
