# :ocean: Underwater Robotic Perception Software

A codebase with examples of visual-inertial odometry, mesh-based obstacle avoidance, underwater image enhancement, and stereo depth estimation.

![Patchmatch GPU implementation example](/resources/patchmatch_gpu_example.png)

## :bulb: Background

I spent about 15 months working on a startup called Blue Meadow. Our original idea was to develop an
autonomous robot for monitoring and performing tasks on offshore aquaculture farms (primarily
seaweed and oysters). This would help farmers detect disease, optimize growing conditions, and
reduce labor costs through fewer visits to the farm site.

Most ocean robots are extremely expensive due to their reliance on acoustic
navigation (e.g sidescan sonar). One of our main goals was to reduce hardware cost by adopting a
visual-inertial approach to navigation, which relies on a cheap IMU and camera. To help the vehicle
constrain its absolute position in the world, it also receives range measurements from one or more
acoustic beacons attached to the farm.

Eventually, we moved away from the idea of a mobile task-automating robot, and started developing
and simpler static sensor package. Since I spent a significant amount of time developing a vision-based
navigation system for our original idea, I thought I'd make the code public in case it's useful
or interesting to others.

**NOTE**: This codebase is not in a very user-friendly state right now, but I'm working on making
it easier for others to use. If there are particular modules you're interested, let me know
and I can prioritize those.

## :tv: Demos

- [Underwater navigation a simulated ocean environment](https://youtu.be/yT-qm5_dXxk)
- [Hybrid smoother/filter state estimator demo](https://youtu.be/Q3swMWAAizs)
- [Stereo object meshing algorithm (simulated)](https://youtu.be/F7nSvaf0kpo)
- [Stereo object meshing algorithm (real world; ZED camera)](https://www.youtube.com/watch?v=TdSf_Qc2J94)

## :memo: Repository Overview

### Software Modules

The main software modules are located in `src/vehicle`:
- `core`: widely used math and data types
- `dataset`: classes for working with a few underwater stereo datasets
- `feature_tracking`: classes for sparse feature detection, optical flow, and stereo matching
- `imaging`: underwater image enhancement algorithms
- `lcm_util`: utils to convert between internal C++ types and LCM types
- `mesher`: applies Delaunay triangulation to tracked stereo features in order to approximate local obstacles
- `params`: a home-grown system for loading params into C++ classes
- `patchmatch_gpu`: faster CUDA implementation of the Patchmatch stereo algorithm
- ~~`rrt`: ignore; not fully implemented or tested~~
- `stereo`: classic OpenCV block matching and a Patchmatch implementation
- `vio`: a full stereo visual odometry pipeline, using GTSAM as a backend
- `vision_core`: widely used computer vision types

Most of these modules have correspond tests in the `test` directory.

**If you're taking a quick glance at this codebase, the modules I'm most proud of are `vio` and `patchmatch_gpu`.**

### Configuration Files

We use a homegrown system for loading YAML configuration files into C++ classes, allowing parameters to change without recompiling. It also avoids having to write massive contructors for classes, or set lots of class members manually.

The YAML configuration files in the `config` folder. For a guide on how our param system is designed, see `src/vehicle/params/README.md`.

### LCM (Lightweight Communications and Marshalling)

We use the [LCM](https://lcm-proj.github.io/) library for communicating across processes and languages. This allows us to define a message type once, and generate bindings in `C++`, `Python`, and our `C#` Unity simulator.

See `lcmtypes` for message type definitions.

## :construction: Next Steps

- [x] Make repository public
- [ ] Add better demos and pictures of outputs
- [x] Stop using `catkin`; switch to `cmake` and make build more lightweight
- [x] Improve setup/build/demo documentation
- [ ] Add documentation to each module
- [ ] Remove abandoned modules

## :hammer: First-Time Setup

### Installing GTSAM

[GTSAM](https://gtsam.org/) is the Georgia-Tech Smoothing and Mapping library. It's used widely
in the robotics community to represent and solve problems as factor graphs.

I'm working off of a fork of GTSAM [here](https://github.com/miloknowles/gtsam). It has a couple
factors that aren't fixed/merged into the GTSAM develop branch yet.

To install this dependency:
- Clone the fork of GTSAM
- `git checkout develop` just to be safe
- `mkdir build && cd build && cmake .. && make`
- Can run tests with `make check`
- `sudo make install` to put the custom library into `/usr/local`
- This repo should include from and link against the installed fork version

### Installing LCM

I had to install Java *before* building and installing LCM from source to get the `lcm-spy` tool.
```bash
sudo apt install openjdk-8-jre
sudo apt install openjdk-8-jdk
```

- Clone our [fork](https://github.com/bluemeadowrobotics/lcm) of LCM
- This has a fix for `lcm-spy` (latest version `1.4.0` fails on Ubuntu 18.04)
- `mkdir build && cd build && cmake .. && sudo make install`

### Building this Repository

Use the usual `cmake` process:
```bash
mkdir build && cd build
cmake ..
make -j8
```

## :question: Other

### Some Notes on Using Eigen

- https://github.com/ethz-asl/eigen_catkin/wiki/Eigen-Memory-Issues#mixed-use-of-StdVector
- I've run into a boatload of issues when using Eigen types and structs containing Eigen types with `std::vector`, `std::make_shared` and other memory-allocating things.
- DO NOT include `eigen3/Eigen/StdVector> and try to use that workaround. It only caused lower-level bugs to show up.
- DO use `EIGEN_MAKE_ALIGNED_OPERATOR_NEW` in any struct that has an Eigen type member
- `std::bad_alloc` exceptions seem to implicate an Eigen type allocation issue
