# Blue Meadow AUV

Main codebase for vehicle software.

## First-Time Setup

Add these lines to your `.bashrc`:
```bash
# Run this once before developing in a new terminal window.
alias bm-shell='source ~/blue-meadow/catkin_ws/src/auv/setup/setup.bash'
```

## Some Notes on Using Eigen

- https://github.com/ethz-asl/eigen_catkin/wiki/Eigen-Memory-Issues#mixed-use-of-StdVector
- I've run into a boatload of issues when using Eigen types and structs containing Eigne types with `std::vector`, `std::make_shared` and other memory-allocating things.
- DO NOT include `eigen3/Eigen/StdVector> and try to use that workaround. It only caused lower-level bugs to show up.
- DO use `EIGEN_MAKE_ALIGNED_OPERATOR_NEW` in any struct that has an Eigen type member
- `std::bad_alloc` exceptions seem to implicate an Eigen type allocation issue
