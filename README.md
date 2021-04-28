# Blue Meadow Vehicle Software

Main codebase for AUV/USV perception, planning, and controls software.

## First-Time Setup

Add these lines to your `.bashrc`:
```bash
# Run this once before developing in a new terminal window.
alias bm-shell='source ~/blue-meadow/catkin_ws/src/vehicle/setup/setup.bash'
```

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

## Some Notes on Using Eigen

- https://github.com/ethz-asl/eigen_catkin/wiki/Eigen-Memory-Issues#mixed-use-of-StdVector
- I've run into a boatload of issues when using Eigen types and structs containing Eigne types with `std::vector`, `std::make_shared` and other memory-allocating things.
- DO NOT include `eigen3/Eigen/StdVector> and try to use that workaround. It only caused lower-level bugs to show up.
- DO use `EIGEN_MAKE_ALIGNED_OPERATOR_NEW` in any struct that has an Eigen type member
- `std::bad_alloc` exceptions seem to implicate an Eigen type allocation issue

## Boost Graph Cheat Sheet

