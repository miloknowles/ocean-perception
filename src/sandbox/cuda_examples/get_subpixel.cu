#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>

namespace cu = cv::cuda;


// https://github.com/evlasblom/cuda-opencv-examples/blob/master/src/diff_1.cu
template <typename T>
__device__ __forceinline__
T GetPixel(const cu::PtrStepSz<T> im, int row, int col)
{
  return im(row, col);
}


template <typename T>
__device__ __forceinline__
void SetPixel(cu::PtrStepSz<T> im, int row, int col, T val)
{
  im(row, col) = val;
}


template <typename T>
__device__ __forceinline__
T GetSubpixel(const cu::PtrStepSz<T> im, float row, float col)
{
  if (row < 0 || row >= im.rows || col < 0 || col >= im.cols) {
    return 0;
  }

  int row0 = __float2int_rd(row);
  int row1 = min(__float2int_ru(row), im.rows - 1);
  int col0 = __float2int_rd(col);
  int col1 = min(__float2int_ru(col), im.cols - 1);

  float c00 = im(row0, col0);
  float c01 = im(row0, col1);
  float c10 = im(row1, col0);
  float c11 = im(row1, col1);

  float trow = row - __int2float_rn(row0);
  float tcol = col - __int2float_rn(col0);

  float c0 = (1.0f - trow) * c00 + trow * c10;
  float c1 = (1.0f - trow) * c01 + trow * c11;

  return (1.0f - tcol) * c0 + tcol * c1;
}


__global__
void GetRectSubpixel(const cu::PtrStepSz<float> im, float row0, float col0, int height, int width, cu::PtrStepSz<float> out)
{
  const int tCol = blockIdx.x * blockDim.x + threadIdx.x;
  const int tRow = blockIdx.y * blockDim.y + threadIdx.y;

  // Only use threads within the rect.
  if (tCol < width && tRow < height) {
    const float row = row0 + __int2float_rd(tRow);
    const float col = col0 + __int2float_rd(tCol);
    out(tRow, tCol) = GetSubpixel<float>(im, row, col);
  }
}


int main(int argc, char* argv[])
{
  cv::Mat src_host = cv::imread("./resources/farmsim_01_left.png", CV_LOAD_IMAGE_GRAYSCALE);
  cv::cuda::GpuMat dst, src;
  src.upload(src_host);

  src.convertTo(dst, CV_32FC1);
  // cv::cuda::GpuMat rect(cv::Size(300, 300), CV_32FC1);

  cv::cuda::GpuMat rect, rect1b;
  rect.create(cv::Size(300, 300), CV_32FC1);

	const dim3 block(16, 16);
  const dim3 grid(cu::device::divUp(rect.cols, block.x), cu::device::divUp(rect.rows, block.y));
  GetRectSubpixel<<<grid, block>>>(dst, 53.4, 11.123, rect.rows, rect.cols, rect);

  rect.convertTo(rect1b, CV_8UC1);
  cudaDeviceSynchronize();

  cv::Mat rect_host;
  rect1b.download(rect_host);

  double minVal, maxVal;
  cv::minMaxLoc(rect_host, &minVal, &maxVal);
  printf("minVal=%f maxVal=%f\n", minVal, maxVal);

  cv::imshow("rect", rect_host);
  cv::waitKey();

  return 0;
}
