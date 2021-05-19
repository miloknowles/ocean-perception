#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include "patchmatch_gpu/patchmatch_gpu.h"

namespace bm {
namespace pm {


void PatchmatchGpu::Params::LoadParams(const YamlParser& p)
{
  detector_params = ft::FeatureDetector::Params(p.Subtree("FeatureDetector"));
  matcher_params = ft::StereoMatcher::Params(p.Subtree("StereoMatcher"));
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
static float L1GradientCost(const cu::PtrStepSz<float> Il,
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
static float L1GradientCost3x3(const cu::PtrStepSz<float> Il,
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


__global__
void MaskBackground(const cu::PtrStepSz<float> iml,
                    const cu::PtrStepSz<float> imr,
                    const cu::PtrStepSz<float> Gl,
                    const cu::PtrStepSz<float> Gr,
                    cu::PtrStepSz<float> disp,
                    int patch_size,
                    float alpha,
                    float improve_factor)
{
  assert(patch_size % 2 != 0);
  const int patch_radius = patch_size / 2;

  const int tCol = blockIdx.x * blockDim.x + threadIdx.x;
  const int tRow = blockIdx.y * blockDim.y + threadIdx.y;

  // Skip rows where there is insufficient padding for patch.
  if (tRow < (patch_radius) || tRow > (iml.rows - patch_radius - 1) ||
      tCol < (patch_radius) || tCol > (iml.cols - patch_radius - 1)) {
    return;
  }

  const float y = __int2float_rd(tRow);
  const float x = __int2float_rd(tCol);

  const float d1 = disp(tRow, tCol);

  const float cost0 = L1GradientCost3x3(
      iml, imr, Gl, Gr, tRow, tCol, y, x, alpha);

  const float cost1 = L1GradientCost3x3(
      iml, imr, Gl, Gr, tRow, tCol, y, fmaxf(x - d1, patch_radius), alpha);

  // If the estimated disparity does not improve cost by more than improve_factor, mark as background.
  if (!(cost1 < improve_factor*cost0)) {
    disp(tRow, tCol) = 0;
  }
}


__global__
void MaskOcclusions(cu::PtrStepSz<float> displ,
                    cu::PtrStepSz<float> dispr)
{
  const int tCol = blockIdx.x * blockDim.x + threadIdx.x;
  const int tRow = blockIdx.y * blockDim.y + threadIdx.y;

  // Skip rows where there is insufficient padding for patch.
  if (tRow < 0 || (tRow > displ.rows - 1) ||
      tCol < 0 || (tCol > displ.cols - 1)) {
    return;
  }

  const float y = __int2float_rd(tRow);
  const float x = __int2float_rd(tCol);
  const float dl = displ(y, x);
  const float dr = dispr(y, fmaxf(x - dl, 0));

  // If a pixel has higher disparity in the right image, it is occluded in the left.
  if (dr > 1.4*dl || dr < 0.7*dl) {
    displ(y, x) = 0;
  }
}


void AddForegroundNoise(cu::GpuMat& disp, const cu::GpuMat& unit_noise, float scale, cu::GpuMat& mask)
{
  cu::threshold(disp, mask, 0.0, 1.0, CV_THRESH_BINARY);
  cu::scaleAdd(unit_noise, scale, disp, disp);
  cu::multiply(disp, mask, disp);
  cu::max(disp, 0, disp);
}


void GradientMagnitude(const cu::GpuMat& im,
                       cu::GpuMat& Gx,
                       cu::GpuMat& Gy,
                       cu::GpuMat& Gmag)
{
  // Compute the image gradient.
  cv::Ptr<cu::Filter> sobel_x = cu::createSobelFilter(CV_32FC1, CV_32FC1, 1, 0, 3);
  cv::Ptr<cu::Filter> sobel_y = cu::createSobelFilter(CV_32FC1, CV_32FC1, 0, 1, 3);

  sobel_x->apply(im, Gx);
  sobel_y->apply(im, Gy);
  cu::magnitude(Gx, Gy, Gmag);
}


PatchmatchGpu::PatchmatchGpu(const Params& params)
    : params_(params),
      detector_(params.detector_params),
      matcher_(params.matcher_params)
{

}


void PatchmatchGpu::Match(const Image1b& iml,
                          const Image1b& imr,
                          Image1f& disp,
                          Image1f& dispr)
{
  disp = SparseInit(iml, imr, params_.init_dilate_factor);

  // Only allocate the noise image once.
  if (unit_noise_gpu_.empty()) {
    Image1f tmp(iml.size(), 0);
    cv::RNG rng(123);
    rng.fill(tmp, cv::RNG::UNIFORM, -1, 1, true);
    unit_noise_gpu_.upload(tmp);
  }

  tmp_.upload(iml);
  tmp_.convertTo(iml_gpu_, CV_32FC1);
  tmp_.upload(imr);
  tmp_.convertTo(imr_gpu_, CV_32FC1);

  GradientMagnitude(iml_gpu_, Gx_, Gy_, Gl_);
  GradientMagnitude(imr_gpu_, Gx_, Gy_, Gr_);

  disp_gpu_.upload(disp);
  Match(iml_gpu_, imr_gpu_, Gl_, Gr_, disp_gpu_);

  cu::flip(iml_gpu_, iml_gpu_flip_, 1);
  cu::flip(imr_gpu_, imr_gpu_flip_, 1);
  cu::flip(Gl_, Gl_flip_, 1);
  cu::flip(Gr_, Gr_flip_, 1);

  Image1b iml_flip, imr_flip;
  cv::flip(iml, iml_flip, 1);
  cv::flip(imr, imr_flip, 1);
  disp = SparseInit(imr_flip, iml_flip, params_.init_dilate_factor);
  disp_gpu_r_.upload(disp);
  Match(imr_gpu_flip_, iml_gpu_flip_, Gr_flip_, Gl_flip_, disp_gpu_r_);
  cu::flip(disp_gpu_r_, disp_gpu_r_, 1);

  const dim3 block(16, 16);
  const dim3 grid(cu::device::divUp(iml.cols, block.x), cu::device::divUp(iml.rows, block.y));
  MaskOcclusions<<<grid, block>>>(disp_gpu_, disp_gpu_r_);

  disp_gpu_.download(disp);
  disp_gpu_r_.download(dispr);
}


void PatchmatchGpu::Match(const cu::GpuMat& iml,
                          const cu::GpuMat& imr,
                          const cu::GpuMat& Gl,
                          const cu::GpuMat& Gr,
                          cu::GpuMat& disp)
{
  const int column_stripes = 16;
  const int row_stripes = 16;
  const dim3 row_block(column_stripes, 16);
  const dim3 row_grid(cu::device::divUp(column_stripes, row_block.x),
                      cu::device::divUp(iml.rows, row_block.y));
  const dim3 col_block(16, row_stripes);
  const dim3 col_grid(cu::device::divUp(iml.cols, row_block.x),
                      cu::device::divUp(row_stripes, row_block.y));

  for (int iter = 0; iter < params_.patchmatch_iters; ++iter) {
    AddForegroundNoise(disp, unit_noise_gpu_, 32.0 / std::pow(2.0, (float)iter), mask_gpu_);
    cudaDeviceSynchronize();
    PropagateRow<<<row_grid, row_block>>>(iml, imr, Gl, Gr, disp, 1, 3, params_.cost_alpha);
    cudaDeviceSynchronize();
    PropagateCol<<<col_grid, col_block>>>(iml, imr, Gl, Gr, disp, 1, 3, params_.cost_alpha);
    cudaDeviceSynchronize();
    PropagateRow<<<row_grid, row_block>>>(iml, imr, Gl, Gr, disp, -1, 3, params_.cost_alpha);
    cudaDeviceSynchronize();
    PropagateCol<<<col_grid, col_block>>>(iml, imr, Gl, Gr, disp, -1, 3, params_.cost_alpha);
  }

  const dim3 block(16, 16);
  const dim3 grid(cu::device::divUp(iml.cols, block.x), cu::device::divUp(iml.rows, block.y));
  MaskBackground<<<grid, block>>>(iml, imr, Gl, Gr, disp, 3, params_.cost_alpha, params_.cost_improve_factor);

  cudaDeviceSynchronize();
}


Image1f PatchmatchGpu::SparseInit(const Image1b& iml,
                                  const Image1b& imr,
                                  int dilate_factor)
{
  VecPoint2f left_kp;
  detector_.Detect(iml, VecPoint2f(), left_kp);
  const std::vector<double>& left_kp_disps = matcher_.MatchRectified(iml, imr, left_kp);

  // Default to zero disparity (background).
  Image1f disps(iml.size(), 0.0f);

  // Fill in disparity for sparse keypoints.
  for (size_t i = 0; i < left_kp_disps.size(); ++i) {
    const cv::Point2f& kp = left_kp.at(i);
    const float d = (float)left_kp_disps.at(i);

    // Skip negative disparity (invalid).
    if (d >= 0) {
      disps.at<float>(std::round(kp.y), std::round(kp.x)) = (float)d;
    }
  }

  const int dilate_size = (int)std::pow(2, dilate_factor) + 1;
  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(2*dilate_size+1, 2*dilate_size+1), cv::Point(dilate_size, dilate_size));
  cv::dilate(disps, disps, element);

  return disps;
}


}
}
