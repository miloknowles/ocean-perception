project(cuda_examples LANGUAGES CXX CUDA)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_60")

add_executable(add_arrays add_arrays.cu)
add_executable(gpu_image gpu_image.cu)
add_executable(get_subpixel get_subpixel.cu)

target_link_libraries(gpu_image ${OpenCV_LIBRARIES})
target_link_libraries(get_subpixel ${OpenCV_LIBRARIES}
  vehicle_stereo_matching
  vehicle_params
  vehicle_core
  vehicle_vision_core
  vehicle_ft)
