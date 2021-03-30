#include "zed_recorder.hpp"

using namespace bm;
using namespace zed;


int main(int argc, char const *argv[])
{
  ZedRecorder recorder("/path/to/folder");
  recorder.Run(true);
  return 0;
}
