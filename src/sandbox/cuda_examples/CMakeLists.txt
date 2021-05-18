project(cuda_examples LANGUAGES CXX CUDA)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_60")

add_executable(add_arrays add_arrays.cu)
add_executable(gpu_image gpu_image.cu)

target_link_libraries(gpu_image ${OpenCV_LIBRARIES})
