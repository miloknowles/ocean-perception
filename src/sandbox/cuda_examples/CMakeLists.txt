project(cuda_examples LANGUAGES CXX CUDA)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_60")

message(${CUDA_NVCC_FLAGS})
add_executable(add_arrays add_arrays.cu)
