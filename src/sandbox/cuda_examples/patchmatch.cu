#include <cuda_runtime.h>

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


template <typename T>
__device__ __forceinline__
T GetSubpixel(const cu::PtrStepSz<T> im, float row, float col)
{
  const int row0 = __float2int_rd(row);
  const int row1 = __float2int_ru(row);
  const int col0 = __float2int_rd(col);
  const int col1 = __float2int_ru(col);

  const float c00 = im(row0, col0);
  const float c01 = im(row0, col1);
  const float c10 = im(row1, col0);
  const float c11 = im(row1, col1);

  const float trow = row - __int2float_rn(row0);
  const float tcol = col - __int2float_rn(col0);

  const float c0 = (1.0f - trow) * c00 + trow * c10;
  const float c1 = (1.0f - trow) * c01 + trow * c11;
  // const float c0 = (1.0f - (row - floorf(row))) * im(__float2int_rd(row), __float2int_rd(col)) +
  //                   (row - floorf(row)) * im(__float2int_rd(row), __float2int_ru(col));
  // const float c1 = (1.0f - (row - floorf(row))) * im(__float2int_ru(row), __float2int_rd(col)) +
  //                   (row - floorf(row)) * im(__float2int_ru(row), __float2int_ru(col));
  return (1.0f - tcol) * c0 + tcol * c1;
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

  #pragma unroll
  for (int row = 0; row < ph; ++row) {
    for (int col = 0; col < pw; ++col) {
      const float yri = yr - __int2float_rd(ph / 2) + __int2float_rd(row);
      const float xri = xr - __int2float_rd(pw / 2) + __int2float_rd(col);
      const int yli = yl - ph / 2 + row;
      const int xli = xl - pw / 2 + col;
      cost += alpha       * fabsf(Il(yli, xli) - GetSubpixel(Ir, yri, xri)) +
              (1 - alpha) * fabsf(Gl(yli, xli) - GetSubpixel(Gr, yri, xri));
    }
  }
  return cost;
}


__device__ __forceinline__
float L1GradientCost3x3(const cu::PtrStepSz<float> Il,
                        const cu::PtrStepSz<float> Ir,
                        const cu::PtrStepSz<float> Gl,
                        const cu::PtrStepSz<float> Gr,
                        int yl, int xl,
                        float yr, float xr,
                        float alpha)
{
  float cost = 0;

  // ROW 0
  cost += alpha     * fabsf(Il(yl - 1, xl - 1) - GetSubpixel(Ir, yr - 1, xr - 1)) +
          (1 - alpha) * fabsf(Gl(yl - 1, xl - 1) - GetSubpixel(Gr, yr - 1, xr - 1));

  // cost += alpha     * fabsf(Il(yl - 1, xl) - GetSubpixel(Ir, yr - 1, xr)) +
  //         (1 - alpha) * fabsf(Gl(yl - 1, xl) - GetSubpixel(Gr, yr - 1, xr));

  cost += alpha     * fabsf(Il(yl - 1, xl + 1) - GetSubpixel(Ir, yr - 1, xr + 1)) +
          (1 - alpha) * fabsf(Gl(yl - 1, xl + 1) - GetSubpixel(Gr, yr - 1, xr + 1));

  // ROW 1
  // cost += alpha     * fabsf(Il(yl, xl - 1) - GetSubpixel(Ir, yr, xr - 1)) +
  //         (1 - alpha) * fabsf(Gl(yl, xl - 1) - GetSubpixel(Gr, yr, xr - 1));

  cost += alpha     * fabsf(Il(yl, xl) - GetSubpixel(Ir, yr, xr)) +
          (1 - alpha) * fabsf(Gl(yl, xl) - GetSubpixel(Gr, yr, xr));

  // cost += alpha     * fabsf(Il(yl, xl + 1) - GetSubpixel(Ir, yr, xr + 1)) +
  //         (1 - alpha) * fabsf(Gl(yl, xl + 1) - GetSubpixel(Gr, yr, xr + 1));

  // ROW 2
  cost += alpha     * fabsf(Il(yl + 1, xl - 1) - GetSubpixel(Ir, yr + 1, xr - 1)) +
          (1 - alpha) * fabsf(Gl(yl + 1, xl - 1) - GetSubpixel(Gr, yr + 1, xr - 1));

  // cost += alpha     * fabsf(Il(yl + 1, xl) - GetSubpixel(Ir, yr + 1, xr)) +
  //         (1 - alpha) * fabsf(Gl(yl + 1, xl) - GetSubpixel(Gr, yr + 1, xr));

  cost += alpha       * fabsf(Il(yl + 1, xl + 1) - GetSubpixel(Ir, yr + 1, xr + 1)) +
          (1 - alpha) * fabsf(Gl(yl + 1, xl + 1) - GetSubpixel(Gr, yr + 1, xr + 1));

  return cost;
}


// __device__ __forceinline__
// float L1GradientCost(const cu::PtrStepSz<float> pl,
//                      const cu::PtrStepSz<float> pr,
//                      const cu::PtrStepSz<float> gl,
//                      const cu::PtrStepSz<float> gr,
//                      float alpha)
// {
//   float cost = 0;
//   for (int row = 0; row < pl.rows; ++row) {
//     for (int col = 0; col < pl.cols; ++col) {
//       cost += alpha * (pl(row, col) - pr(row, col)) + (1 - alpha) * (gl(row, col) - gr(row, col));
//     }
//   }
//   return cost;
// }


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
                  int patch_size,
                  float alpha)
{
  assert(patch_size % 2 != 0);
  assert(direction == -1 || direction == 1);

  const int patch_radius = patch_size / 2;

  const int tRow = blockIdx.y * blockDim.y + threadIdx.y;

  // Skip rows where there is insufficient padding for patch.
  if (tRow < (patch_size / 2) || tRow > (iml.rows - patch_radius - 1)) {
    return;
  }

  const int tColChunk = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread get a "chunk" of a row.
  const int chunkSize = iml.cols / blockDim.x;

  const int minCol = max(tColChunk * chunkSize - 5, patch_radius);
  const int maxCol = min((tColChunk + 1)*chunkSize + 5, iml.cols - patch_radius - 1);

  // Skip invalid cols.
  if (minCol >= iml.cols) {
    return;
  }

  const int start = (direction > 0) ? minCol : maxCol;
  const int end = (direction > 0) ? maxCol : minCol;

  const float y = __int2float_rd(tRow);

  for (int col = start; direction > 0 ? col < end : col > end; col += direction) {
    const float x = __int2float_rd(col);
    const float d0 = disp(tRow, col);
    const float d1 = disp(tRow, col - direction);

    const float cost0 = L1GradientCost3x3(
        iml, imr, Gl, Gr, tRow, col, y, fmaxf(x - d0, patch_radius), alpha);

    const float cost1 = L1GradientCost3x3(
        iml, imr, Gl, Gr, tRow, col, y, fmaxf(x - d1, patch_radius), alpha);

    // If using the neighboring disp improves cost, use it (and clip to valid range).
    if (cost1 < cost0) {
      disp(tRow, col) = fminf(d1, x - patch_radius);
    }
  }
}


__global__
void PropagateCol(const cu::PtrStepSz<float> iml,
                  const cu::PtrStepSz<float> imr,
                  const cu::PtrStepSz<float> Gl,
                  const cu::PtrStepSz<float> Gr,
                  cu::PtrStepSz<float> disp,
                  int direction,
                  int patch_size,
                  float alpha)
{
  assert(patch_size % 2 != 0);
  assert(direction == -1 || direction == 1);

  const int patch_radius = patch_size / 2;
  const int tCol = blockIdx.x * blockDim.x + threadIdx.x;

  // Skip rows where there is insufficient padding for patch.
  if (tCol < (patch_size / 2) || tCol > (iml.cols - patch_radius - 1)) {
    return;
  }

  const int tRowChunk = blockIdx.y * blockDim.y + threadIdx.y;

  // Each thread get a "chunk" of a row.
  const int chunkSize = iml.rows / blockDim.y;

  const int minRow = max(tRowChunk * chunkSize - 5, patch_radius);
  const int maxRow = min((tRowChunk + 1)*chunkSize + 5, iml.rows - patch_radius - 1);

  // Skip invalid cols.
  if (minRow >= iml.rows) {
    return;
  }

  const int start = (direction > 0) ? minRow : maxRow;
  const int end = (direction > 0) ? maxRow : minRow;

  const float x = __int2float_rd(tCol);

  for (int row = start; direction > 0 ? row < end : row > end; row += direction) {
    const float y = __int2float_rd(row);
    const float d0 = disp(row, tCol);
    const float d1 = disp(row - direction, tCol);

    const float cost0 = L1GradientCost3x3(
        iml, imr, Gl, Gr, row, tCol, y, fmaxf(x - d0, patch_radius), alpha);

    const float cost1 = L1GradientCost3x3(
        iml, imr, Gl, Gr, row, tCol, y, fmaxf(x - d1, patch_radius), alpha);

    // If using the neighboring disp improves cost, use it (and clip to valid range).
    if (cost1 < cost0) {
      disp(row, tCol) = fminf(d1, x - patch_radius);
    }
  }
}


void AddForegroundNoise(cu::GpuMat& disp, const cu::GpuMat& unit_noise, float scale, cu::GpuMat& mask)
{
  cu::threshold(disp, mask, 0.0, 1.0, CV_THRESH_BINARY);
  cu::scaleAdd(unit_noise, scale, disp, disp);
  cu::multiply(disp, mask, disp);
  cu::max(disp, 0, disp);
}


static void GradientMagnitude(const cu::GpuMat& im, cu::GpuMat& Gx, cu::GpuMat& Gy, cu::GpuMat& Gmag)
{
  // Compute the image gradient.
  cv::Ptr<cu::Filter> sobel_x = cu::createSobelFilter(CV_32FC1, CV_32FC1, 1, 0, 3);
  cv::Ptr<cu::Filter> sobel_y = cu::createSobelFilter(CV_32FC1, CV_32FC1, 0, 1, 3);

  sobel_x->apply(im, Gx);
  sobel_y->apply(im, Gy);
  cu::magnitude(Gx, Gy, Gmag);
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

  printf("Image dimensions: %d %d\n", il.cols, il.rows);

  cu::GpuMat iml_gpu, imr_gpu, disp_gpu, tmp;
  tmp.upload(iml_pm);
  tmp.convertTo(iml_gpu, CV_32FC1);
  tmp.upload(imr_pm);
  tmp.convertTo(imr_gpu, CV_32FC1);

  cu::GpuMat _Gx, _Gy, Gl, Gr;
  GradientMagnitude(iml_gpu, _Gx, _Gy, Gl);
  GradientMagnitude(imr_gpu, _Gx, _Gy, Gr);

  cu::GpuMat mask;

  bms::Image1f unit_noise(disp.size(), 0);
  cu::GpuMat unit_noise_gpu;
  cv::RNG rng(123);
  rng.fill(unit_noise, cv::RNG::UNIFORM, -1, 1, true);
  unit_noise_gpu.upload(unit_noise);

  // disp_gpu.create(iml_gpu.size(), CV_32FC1);
  disp_gpu.upload(disp);

  const int column_stripes = 16;
  const int row_stripes = 16;
  const dim3 row_block(column_stripes, 16);
  const dim3 row_grid(cu::device::divUp(column_stripes, row_block.x),
                      cu::device::divUp(iml_pm.rows, row_block.y));
  const dim3 col_block(16, row_stripes);
  const dim3 col_grid(cu::device::divUp(iml_pm.cols, row_block.x),
                      cu::device::divUp(row_stripes, row_block.y));

  const int iters = 3;
  const float alpha = 0.9;
  for (int iter = 0; iter < iters; ++iter) {
    AddForegroundNoise(disp_gpu, unit_noise_gpu, 32 / std::pow(2.0, (float)iter), mask);
    cudaDeviceSynchronize();
    PropagateRow<<<row_grid, row_block>>>(iml_gpu, imr_gpu, Gl, Gr, disp_gpu, 1, 3, alpha);
    cudaDeviceSynchronize();
    PropagateCol<<<col_grid, col_block>>>(iml_gpu, imr_gpu, Gl, Gr, disp_gpu, 1, 3, alpha);
    cudaDeviceSynchronize();
    PropagateRow<<<row_grid, row_block>>>(iml_gpu, imr_gpu, Gl, Gr, disp_gpu, -1, 3, alpha);
    cudaDeviceSynchronize();
    PropagateCol<<<col_grid, col_block>>>(iml_gpu, imr_gpu, Gl, Gr, disp_gpu, -1, 3, alpha);
  }

  cudaDeviceSynchronize();

  cv::Mat1f disp_out;
  disp_gpu.download(disp_out);
  return disp_out;
}



int main(int argc, char* argv[])
{
  const bms::Image1f disp = PatchmatchGpu();
  cv::imshow("disp", VisualizeDisp(disp, 128, 2));
  cv::waitKey(0);

  return 0;
}
