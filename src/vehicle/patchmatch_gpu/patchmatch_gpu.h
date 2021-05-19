#pragma once

// #include <cuda_runtime.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>

#include "core/macros.hpp"
#include "vision_core/cv_types.hpp"
#include "params/params_base.hpp"
#include "params/yaml_parser.hpp"
#include "feature_tracking/feature_detector.hpp"
#include "feature_tracking/stereo_matcher.hpp"

namespace bm {
namespace pm {

namespace cu = cv::cuda;
using namespace core;


template <typename T>
__device__ __forceinline__
T GetSubpixel(const cu::PtrStepSz<T> im, float row, float col);


__global__
void PropagateRow(const cu::PtrStepSz<float> iml,
                  const cu::PtrStepSz<float> imr,
                  const cu::PtrStepSz<float> Gl,
                  const cu::PtrStepSz<float> Gr,
                  cu::PtrStepSz<float> disp,
                  int direction,
                  int patch_size,
                  float alpha);


__global__
void PropagateCol(const cu::PtrStepSz<float> iml,
                  const cu::PtrStepSz<float> imr,
                  const cu::PtrStepSz<float> Gl,
                  const cu::PtrStepSz<float> Gr,
                  cu::PtrStepSz<float> disp,
                  int direction,
                  int patch_size,
                  float alpha);


__global__
void MaskBackground(const cu::PtrStepSz<float> iml,
                    const cu::PtrStepSz<float> imr,
                    const cu::PtrStepSz<float> Gl,
                    const cu::PtrStepSz<float> Gr,
                    cu::PtrStepSz<float> disp,
                    int patch_size,
                    float alpha,
                    float improve_factor);


void AddForegroundNoise(cu::GpuMat& disp,
                        const cu::GpuMat& unit_noise,
                        float scale,
                        cu::GpuMat& mask);


void GradientMagnitude(const cu::GpuMat& im,
                       cu::GpuMat& Gx,
                       cu::GpuMat& Gy,
                       cu::GpuMat& Gmag);


class PatchmatchGpu final {
 public:
  struct Params final : public ParamsBase {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    ft::FeatureDetector::Params detector_params;
    ft::StereoMatcher::Params matcher_params;

    float cost_alpha = 0.9;
    int patchmatch_iters = 3;
    int init_dilate_factor = 4;
    float cost_improve_factor = 0.8;

   private:
    void LoadParams(const YamlParser& p) override;
  };

  MACRO_DELETE_COPY_CONSTRUCTORS(PatchmatchGpu);

  PatchmatchGpu(const Params& params);

 public:
  void Match(const Image1b& iml,
             const Image1b& imr,
             Image1f& disp);

  void Match(const cu::GpuMat& iml,
             const cu::GpuMat& imr,
             const cu::GpuMat& Gl,
             const cu::GpuMat& Gr,
             cu::GpuMat& disp);

  Image1f SparseInit(const Image1b& iml,
                     const Image1b& imr,
                     int dilate_factor);

 private:
  Params params_;

  ft::FeatureDetector detector_;
  ft::StereoMatcher matcher_;

  // Pre-allocate these GpuMats to save on allocation time.
  cu::GpuMat mask_gpu_, unit_noise_gpu_;
  cu::GpuMat iml_gpu_, imr_gpu_, Gx_, Gy_, Gl_, Gr_, disp_gpu_, tmp_;
};

}
}
