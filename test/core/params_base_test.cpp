#include <gtest/gtest.h>
#include <glog/logging.h>

#include "core/macros.hpp"
#include "params/params_base.hpp"

using namespace bm;
using namespace core;


struct SubtreeStruct final : public ParamsBase
{
  MACRO_PARAMS_STRUCT_CONSTRUCTORS(SubtreeStruct);

  double key1;
  double KEY2;
  bool c;

 void Print() const
 {
  printf("SubtreeStruct key1=%lf KEY2=%lf c=%d\n", key1, KEY2, c);
 }

 private:
  void LoadParams(const YamlParser& parser) override
  {
    parser.GetParam("key1", &key1);
    parser.GetParam("KEY2", &KEY2);
    parser.GetParam("subsubtree/c", &c);
  }
};

inline bool operator==(const SubtreeStruct& lhs, const SubtreeStruct& rhs)
{
  return (lhs.key1 == rhs.key1 && lhs.KEY2 == rhs.KEY2 && lhs.c == rhs.c);
}


struct TestStruct final : public ParamsBase
{
  MACRO_PARAMS_STRUCT_CONSTRUCTORS(TestStruct);

  int a;
  int b;
  Vector3d v;

  SubtreeStruct subtree;

  void Print() const
  {
    printf("TestStruct a=%d b=%d\n", a, b);
    std::cout << "v=" << v.transpose() << std::endl;
    subtree.Print();
  }

 private:
  void LoadParams(const YamlParser& parser) override
  {
    parser.GetParam("a", &a);
    parser.GetParam("b", &b);

    const cv::FileNode& vnode = parser.GetNode("v");
    YamlToVector<Vector3d>(vnode, v);

    subtree = SubtreeStruct(parser.GetNode("SubtreeStruct"));
  }
};


inline bool operator==(const TestStruct& lhs, const TestStruct& rhs)
{
  return (lhs.a == rhs.a && lhs.b == rhs.b && lhs.subtree == rhs.subtree);
}


TEST(ParamsBaseTest, Test_01)
{
  const std::string filepath = "./resources/test_struct_params.yaml";

  TestStruct actual;
  actual.a = 456;
  actual.b = 789;
  actual.v = Vector3d(1, 2, 3);
  actual.subtree.key1 = 3.14159;
  actual.subtree.KEY2 = 2.0;
  actual.subtree.c = false;

  // Default construct with parse afterwards.
  TestStruct params1;
  params1.Parse(filepath);
  params1.Print();
  ASSERT_EQ(actual, params1);

  TestStruct params2(filepath);
  params2.Print();
  ASSERT_EQ(actual, params2);
}
