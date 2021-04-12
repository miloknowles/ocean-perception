#include "zed_recorder.hpp"

using namespace bm;
using namespace zed;


int main(int argc, char const *argv[])
{
  ZedRecorder recorder("/home/milo/datasets/zed/default_dataset");
  recorder.Run(true);
  return 0;
}
