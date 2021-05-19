#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>

#include "stereo_matching/patchmatch.hpp"

namespace cu = cv::cuda;
namespace bms = bm::stereo;


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
void GetRectSubpixel(const cu::PtrStepSz<float> im,
                     float row0, float col0,
                     int height, int width,
                     cu::PtrStepSz<float> out)
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


__device__ __forceinline__
void GetPatchSubpixel(const cu::PtrStepSz<float> im,
                     float y, float x,
                     int ph, int pw,
                     cu::PtrStepSz<float> out)
{
  const float hh = __int2float_rd(ph / 2);
  const float hw = __int2float_rd(pw / 2);

  for (int py = 0; py < ph; ++py) {
    for (int px = 0; px < pw; ++px) {
      out(py, px) = GetSubpixel<float>(
          im, y + __int2float_rd(py) - hh, x + __int2float_rd(px) - hw);
    }
  }
}


void GetPatch()
{
  cv::Mat src_host = cv::imread("./resources/farmsim_01_left.png", CV_LOAD_IMAGE_GRAYSCALE);
  cv::cuda::GpuMat dst, src;
  src.upload(src_host);
  src.convertTo(dst, CV_32FC1);

  // Compute the image gradient.
  cv::Ptr<cu::Filter> sobel_x = cu::createSobelFilter(CV_32FC1, CV_32FC1, 1, 0, 3);
  cv::Ptr<cu::Filter> sobel_y = cu::createSobelFilter(CV_32FC1, CV_32FC1, 0, 1, 3);

  cv::cuda::GpuMat Gx, Gy, Gmag;
  sobel_x->apply(dst, Gx);
  sobel_y->apply(dst, Gy);
  cu::magnitude(Gx, Gy, Gmag);
  // cu::pow(Gx, 2.0, Gx);
  // cu::pow(Gy, 2.0, Gy);
  // cu::add(Gx, Gy, Gmag);

  cv::cuda::GpuMat rect(cv::Size(300, 300), CV_32FC1);

	const dim3 block(16, 16);
  const dim3 grid(cu::device::divUp(rect.cols, block.x), cu::device::divUp(rect.rows, block.y));
  GetRectSubpixel<<<grid, block>>>(dst, 53.4, 11.123, rect.rows, rect.cols, rect);

  cv::cuda::GpuMat rect1b, Gmag1b;
  rect.convertTo(rect1b, CV_8UC1);
  Gmag.convertTo(Gmag1b, CV_8UC1);

  cv::Mat rect_host, Gmag_host;
  rect1b.download(rect_host);
  Gmag1b.download(Gmag_host);

  cudaDeviceSynchronize();

  // double minVal, maxVal;
  // cv::minMaxLoc(rect_host, &minVal, &maxVal);
  // printf("minVal=%f maxVal=%f\n", minVal, maxVal);

  cv::imshow("rect", rect_host);
  cv::imshow("Gmag", Gmag_host);
  cv::waitKey();
}


int main(int argc, char* argv[])
{
  return 0;
}
