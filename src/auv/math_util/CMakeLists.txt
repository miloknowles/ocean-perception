SET(LIBRARY_NAME
  ${PROJECT_NAME}_math_util
)

SET(LIBRARY_SRC
  math_util.cpp
)

add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SRC})
set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${LIBRARY_NAME} PUBLIC "-std=c++11")
target_include_directories(${LIBRARY_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)

install(TARGETS ${LIBRARY_NAME} EXPORT ${LIBRARY_NAME}_targets
  INCLUDES DESTINATION include
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# Install headers.
install(DIRECTORY .
DESTINATION include/auv
FILES_MATCHING PATTERN "*.h*")
