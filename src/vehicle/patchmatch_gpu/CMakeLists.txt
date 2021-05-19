SET(LIBRARY_NAME ${PROJECT_NAME}_pm_gpu)

SET(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_60")

SET(LIBRARY_SRC
  patchmatch_gpu.cu
  patchmatch_gpu.h)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
target_include_directories(${LIBRARY_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${LIBRARY_NAME}
  ${PROJECT_NAME}_core
  ${PROJECT_NAME}_params
  ${PROJECT_NAME}_vision_core
  ${PROJECT_NAME}_ft
  ${OpenCV_LIBRARIES})
