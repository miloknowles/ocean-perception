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


static bms::Image3b VisualizeDisp(const bms::Image1f& disp, int max_disp, int pm_downsample_factor)
{
  bms::Image1b disp8_1c;
  disp.convertTo(disp8_1c, CV_8UC1, std::pow(2, pm_downsample_factor) * 255.0f / max_disp);

  bms::Image3b disp8_3c;
  cv::applyColorMap(disp8_1c, disp8_3c, cv::COLORMAP_PARULA);

  cv::resize(disp8_3c, disp8_3c, disp8_3c.size() * pm_downsample_factor);

  return disp8_3c;
}


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


__device__ __forceinline__
float L1GradientCost(const cu::PtrStepSz<float> pl,
                     const cu::PtrStepSz<float> pr,
                     const cu::PtrStepSz<float> gl,
                     const cu::PtrStepSz<float> gr,
                     float alpha)
{
  float cost = 0;
  for (int row = 0; row < pl.rows; ++row) {
    for (int col = 0; col < pl.cols; ++col) {
      cost += alpha * (pl(row, col) - pr(row, col)) + (1 - alpha) * (gl(row, col) - gr(row, col));
    }
  }
  return cost;
}


__device__ __forceinline__
float L1GradientCost(const cu::PtrStepSz<float> Il,
                     const cu::PtrStepSz<float> Ir,
                     const cu::PtrStepSz<float> Gl,
                     const cu::PtrStepSz<float> Gr,
                     int yl, int xl,
                     float yr, float xr,
                     int ph, int pw,
                     float alpha)
{
  float cost = 0;

  for (int row = 0; row < ph; ++row) {
    for (int col = 0; col < pw; ++col) {
      const float yri = yr - __int2float_rd(ph / 2) + __int2float_rd(row);
      const float xri = xr - __int2float_rd(pw / 2) + __int2float_rd(col);
      const int yli = yl - ph / 2 + row;
      const int xli = xl - pw / 2 + col;
      cost += alpha       * (Il(yli, xli) - GetSubpixel(Ir, yri, xri)) +
              (1 - alpha) * (Gl(yli, xli) - GetSubpixel(Gr, yri, xri));
    }
  }
  return cost;
}


__global__
void Propagate4(const cu::PtrStepSz<float> iml,
                const cu::PtrStepSz<float> imr,
                const cu::PtrStepSz<float> Gl,
                const cu::PtrStepSz<float> Gr,
                cu::PtrStepSz<float> disp,
                int patch_size)
{
  assert(patch_size % 2 != 0);

  const int tCol = blockIdx.x * blockDim.x + threadIdx.x;
  const int tRow = blockIdx.y * blockDim.y + threadIdx.y;

  // Skip rows where there is insufficient padding for patch.
  if (tRow < (patch_size / 2) || tRow > (iml.rows - patch_size / 2 - 1) ||
      tCol < (patch_size / 2) || tCol > (iml.cols - patch_size / 2 - 1)) {
    return;
  }

  const float y = __int2float_rd(tRow);
  const float x = __int2float_rd(tCol);

  float lowest_cost = 1e9;
  float best_disp = 0;

  for (int tNb = 0; tNb < 5; ++tNb) {
    int Nx = tCol;
    int Ny = tRow;
    if (tNb == 1) {         // Left.
      Nx = tCol - 1;
    } else if (tNb == 2) {  // Top.
      Ny = tRow - 1;
    } else if (tNb == 3) {  // Right.
      Nx = tCol + 1;
    } else {                // Bottom.
      Ny = tRow + 1;
    }

    const float d0 = disp(Ny, Nx);
    const float cost = L1GradientCost(
        iml, imr, Gl, Gr, tRow, tCol, y, fmaxf(x - d0, patch_size / 2), patch_size, patch_size, 0.7);

    if (cost < lowest_cost || tNb == 0) {
      lowest_cost = cost;
      best_disp = d0;
    }
  }

  disp(tRow, tCol) = best_disp;
}


__global__
void PropagateRow(const cu::PtrStepSz<float> iml,
                  const cu::PtrStepSz<float> imr,
                  const cu::PtrStepSz<float> Gl,
                  const cu::PtrStepSz<float> Gr,
                  cu::PtrStepSz<float> disp,
                  int direction,
                  int patch_size)
{
  assert(patch_size % 2 != 0);
  assert(direction == -1 || direction == 1);

  const int tRow = blockIdx.y * blockDim.y + threadIdx.y;

  // Skip rows where there is insufficient padding for patch.
  if (tRow < (patch_size / 2) || tRow > (iml.rows - patch_size / 2 - 1)) {
    return;
  }

  const int tColChunk = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread get a "chunk" of a row.
  const int chunkSize = iml.cols / blockDim.x;
  const int minCol = max(tColChunk * chunkSize, patch_size / 2);
  const int maxCol = min((tColChunk + 1)*chunkSize, iml.cols - patch_size / 2 - 1);

  // const int minCol = patch_size / 2;
  // const int maxCol = iml.cols - patch_size / 2 - 1;
  const int start = (direction > 0) ? minCol : maxCol;
  const int end = (direction > 0) ? maxCol : minCol;

  const float y = __int2float_rd(tRow);

  for (int col = start; col < end; col += direction) {
    const float x = __int2float_rd(col);
    const float d0 = disp(tRow, col);
    const float d1 = disp(tRow, col - direction);

    const float cost0 = L1GradientCost(
        iml, imr, Gl, Gr, tRow, col, y, fmaxf(x - d0, patch_size / 2), patch_size, patch_size, 0.7);

    const float cost1 = L1GradientCost(
        iml, imr, Gl, Gr, tRow, col, y, fmaxf(x - d1, patch_size / 2), patch_size, patch_size, 0.7);

    printf("cost0=%f cost1=%f\n", cost0, cost1);

    // If using the neighboring disp improves cost, use it (and clip to valid range).
    if (cost1 < cost0) {
      disp(tRow, col) = fmaxf(x - d1, minCol);
    }
  }
}


static cu::GpuMat GradientMagnitude(const cu::GpuMat& im)
{
  // Compute the image gradient.
  cv::Ptr<cu::Filter> sobel_x = cu::createSobelFilter(CV_32FC1, CV_32FC1, 1, 0, 3);
  cv::Ptr<cu::Filter> sobel_y = cu::createSobelFilter(CV_32FC1, CV_32FC1, 0, 1, 3);

  cv::cuda::GpuMat Gx, Gy, Gmag;
  sobel_x->apply(im, Gx);
  sobel_y->apply(im, Gy);
  cu::magnitude(Gx, Gy, Gmag);

  return Gmag;
}


cv::Mat Patchmatch1()
{
  cv::Mat iml_host = cv::imread("./resources/farmsim_01_left.png", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat imr_host = cv::imread("./resources/farmsim_01_right.png", CV_LOAD_IMAGE_GRAYSCALE);

  cu::GpuMat iml_gpu, imr_gpu, disp_gpu;
  iml_gpu.upload(iml_host);
  iml_gpu.convertTo(iml_gpu, CV_32FC1);

  imr_gpu.upload(imr_host);
  imr_gpu.convertTo(imr_gpu, CV_32FC1);

  const cu::GpuMat Gl = GradientMagnitude(iml_gpu);
  const cu::GpuMat Gr = GradientMagnitude(imr_gpu);

  disp_gpu.create(iml_gpu.size(), CV_32FC1);

  // const dim3 block(8, 32);
  // const dim3 grid(cu::device::divUp(8, block.x), cu::device::divUp(iml_host.rows, block.y));
  // PropagateRow<<<grid, block>>>(iml_gpu, imr_gpu, Gl, Gr, disp_gpu, 1, 5);

  const dim3 block(16, 16);
  const dim3 grid(cu::device::divUp(iml_host.cols, block.x),
                  cu::device::divUp(iml_host.rows, block.y));
  Propagate4<<<grid, block>>>(iml_gpu, imr_gpu, Gl, Gr, disp_gpu, 5);

  cudaDeviceSynchronize();

  cv::Mat disp_host;
  disp_gpu.download(disp_host);

  return disp_host;
}


cv::Mat PatchmatchGpu()
{
  bms::Image1b il = cv::imread("./resources/images/fsl1.png", CV_LOAD_IMAGE_GRAYSCALE);
  bms::Image1b ir = cv::imread("./resources/images/fsr1.png", CV_LOAD_IMAGE_GRAYSCALE);

  // Image1b il = cv::imread("./resources/caddy_32_left.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  // Image1b ir = cv::imread("./resources/caddy_32_right.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  cv::imshow("left image", il);
  cv::imshow("right image", ir);

  const int downsample_factor = 2;
  cv::resize(il, il, il.size() / downsample_factor);
  cv::resize(ir, ir, ir.size() / downsample_factor);

  bms::Patchmatch::Params params;
  float max_disp = 128;

  params.matcher_params.templ_cols = 31;
  params.matcher_params.templ_rows = 11;
  params.matcher_params.max_disp = max_disp;
  params.matcher_params.max_matching_cost = 0.15;
  params.matcher_params.bidirectional = true;
  params.matcher_params.subpixel_refinement = false;

  bms::Patchmatch pm(params);

  int pm_downsample_factor = 1;
  bms::Image1f disp = pm.Initialize(il, ir, pm_downsample_factor);

  bms::Image1b iml_pm, imr_pm;
  cv::resize(il, iml_pm, il.size() / pm_downsample_factor, 0, 0, cv::INTER_LINEAR);
  cv::resize(ir, imr_pm, ir.size() / pm_downsample_factor, 0, 0, cv::INTER_LINEAR);

  cu::GpuMat iml_gpu, imr_gpu, disp_gpu;
  iml_gpu.upload(iml_pm);
  iml_gpu.convertTo(iml_gpu, CV_32FC1);

  imr_gpu.upload(imr_pm);
  imr_gpu.convertTo(imr_gpu, CV_32FC1);

  const cu::GpuMat Gl = GradientMagnitude(iml_gpu);
  const cu::GpuMat Gr = GradientMagnitude(imr_gpu);

  // disp_gpu.create(iml_gpu.size(), CV_32FC1);
  disp_gpu.upload(disp);

  const dim3 block(8, 32);
  const dim3 grid(cu::device::divUp(8, block.x), cu::device::divUp(iml_pm.rows, block.y));
  PropagateRow<<<grid, block>>>(iml_gpu, imr_gpu, Gl, Gr, disp_gpu, 1, 5);

  // const dim3 block(16, 16);
  // const dim3 grid(cu::device::divUp(iml_pm.cols, block.x),
  //                 cu::device::divUp(iml_pm.rows, block.y));

  // for (int i = 0; i < 10; ++i) {
  //   Propagate4<<<grid, block>>>(iml_gpu, imr_gpu, Gl, Gr, disp_gpu, 5);
  //   cudaDeviceSynchronize();
  // }

  cv::Mat1f disp_out;
  disp_gpu.download(disp_out);
  return disp_out;
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
  const cv::Mat1f disp = PatchmatchGpu();
  cv::imshow("disp", VisualizeDisp(disp, 128, 2));
  cv::waitKey(0);

  return 0;
}
