# :notebook: YAML Param System

## Overview

We use a YAML based param system for several reasons:
- It allows us to change system parameters without recompiling
- It avoids passing tons of params into a constructor
- It allows us easily nest paramaters in a hierarchical structure

## Module-Specific Params vs. Shared Params

Some params are **module-specific**, and others are shared across modules. For example, camera calibration data, the direction of gravity, etc. are all **shared params**, while specific input parameters to an algorithm are module-specific.

When constructing a params struct, we always pass in a path to the module-specific YAML, and optionally include a path to the shared YAML.

For example:
```cpp
ObjectMesherLcm::Params params(
  config_path(module_params_path),
  config_path(shared_params_path));

ObjectMesherLcm node(params);
```

## Reading YAML into a C++ Struct

Each `C++` class has a param struct, and that param struct has a 1:1 representation in a YAML file.

For example, in `feature_tracker.hpp`:
```c++
class FeatureTracker final {
 public:
  // Every code module has a 'Params' struct inside of it that inherits from ParamsBase.
  struct Params final : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    int klt_maxiters = 30;
    float klt_epsilon = 0.001;
    int klt_winsize = 21;
    int klt_max_level = 4;

   private:
    void LoadParams(const YamlParser& parser) override;
  };
```

Every `Params` struct overrides the `LoadParams` member function. In `feature_tracker.cpp`:
```cpp
// Override the LoadParams member function.
void FeatureTracker::Params::LoadParams(const YamlParser& parser)
{
  // Here, we get the YAML node named "klt_maxiters" and store it in the member "klt_maxiters".
  parser.GetParam("klt_maxiters", &klt_maxiters);
  parser.GetParam("klt_epsilon", &klt_epsilon);
  parser.GetParam("klt_winsize", &klt_winsize);
  parser.GetParam("klt_max_level", &klt_max_level);
}
```

The corresponding YAML file would contain:
```yaml
klt_maxiters: 10
klt_epsilon: 0.01
klt_winsize: 21
klt_max_level: 4
```

## Nested Parameters

Some high-level classes have other clases as members. The param system is designed to allow class composition through nested params.

For example, in `stereo_tracker.hpp`:
```cpp
class StereoTracker final {
 public:
  // Parameters that control the frontend.
  struct Params final : public ParamsBase
  {
    MACRO_PARAMS_STRUCT_CONSTRUCTORS(Params);

    // This class has an internal FeatureDetector, FeatureTracker, and StereoMatcher.
    // All of these members need their own params.
    FeatureDetector::Params detector_params;
    FeatureTracker::Params tracker_params;
    StereoMatcher::Params matcher_params;

    double stereo_max_depth = 30.0;
    // ...

   private:
    void LoadParams(const YamlParser& parser) override;
  };
```

The YAML for this would look like:
```yaml
StereoTracker:
    stereo_max_depth: 20.0 # m
    # ...

    # Nested subtree!
    FeatureDetector:
      max_features_per_frame: 200
      subpixel_corners: 0 # bool
      # ...

    FeatureTracker:
      klt_maxiters: 10
      klt_epsilon: 0.01
      # ...

    StereoMatcher:
      templ_cols: 31
      templ_rows: 31
      # ...
```

In the high level class' `LoadParams`, we can recursively load params for the internal classes.
```cpp
void StereoTracker::Params::LoadParams(const YamlParser& parser)
{
  // Each sub-module has a subtree in the params.yaml
  detector_params = FeatureDetector::Params(parser.GetNode("FeatureDetector"));
  tracker_params = FeatureTracker::Params(parser.GetNode("FeatureTracker"));
  matcher_params = StereoMatcher::Params(parser.GetNode("StereoMatcher"));

  parser.GetParam("stereo_max_depth", &stereo_max_depth);
  // ...

  // NOTE: Can also validate params like this!
  CHECK(retrack_frames_k >= 1 && retrack_frames_k < 8);
}
```
