# :ocean: Underwater Robotic Vision Software

Main codebase for AUV/USV vision software.
***

## :bulb: Background

I spent about a year working on a startup called Blue Meadow. Our original idea was to develop an
autonomous robot for monitoring and performing tasks on offshore aquaculture farms (primarily
seaweed and oysters). Most ocean robots are extremely expensive due to their reliance on acoustic
navigation (e.g sidescan sonar). One of our main goals was to reduce cost by adopting a "camera-first" approach to
vehicle perception.

This repository contains a few months of work on that project. It's not in a very
user-friendly state right now, but I'm working on making it easier for others to use.

## :memo: Repository Overview

The main software modules are located in `src/vehicle`:
- `core`: widely used math and data types
- `dataset`: classes for working with a few underwater stereo datasets
- `feature_tracking`: classes for sparse feature detection, optical flow, and stereo matching
- `imaging`: underwater image enhancement algorithms
- `lcm_util`: utils to convert between internal C++ types and LCM types
- `mesher`: applies Delaunay triangulation to tracked stereo features in order to approximate local obstacles
- `params`: a home-grown system for loading params into C++ classes
- `patchmatch_gpu`: faster CUDA implementation of the Patchmatch stereo algorithm
- `rrt`: ignore; not fully implemented or tested
- `stereo`: classic OpenCV block matching and a Patchmatch implementation
- `vio`: a full stereo visual odometry pipeline, using GTSAM as a backend
- `vision_core`: widely used computer vision types

**The modules I'm most proud of are `vio` and `patchmatch_gpu`.**

## :soon: Next Steps

- [x] Make repository public
- [ ] Add better demos and pictures of outputs
- [ ] Stop using catkin; make build more lightweight
- [ ] Reduce dependencies; try to make standalone modules
- [ ] Improve setup/build/demo documentation
- [ ] Add documentation to each module
- [ ] Remove abandoned modules

## :hammer: First-Time Setup

(Optional) Add these lines to your `.bashrc`:
```bash
# Run this once before developing in a new terminal window.
alias bm-shell='source ~/blue-meadow/catkin_ws/src/vehicle/setup/setup.bash'
```

### Create Catkin Workspace

Clone this repo inside of catkin workspace.

### Installing GTSAM

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

## :question: Other

### Some Notes on Using Eigen

- https://github.com/ethz-asl/eigen_catkin/wiki/Eigen-Memory-Issues#mixed-use-of-StdVector
- I've run into a boatload of issues when using Eigen types and structs containing Eigne types with `std::vector`, `std::make_shared` and other memory-allocating things.
- DO NOT include `eigen3/Eigen/StdVector> and try to use that workaround. It only caused lower-level bugs to show up.
- DO use `EIGEN_MAKE_ALIGNED_OPERATOR_NEW` in any struct that has an Eigen type member
- `std::bad_alloc` exceptions seem to implicate an Eigen type allocation issue

