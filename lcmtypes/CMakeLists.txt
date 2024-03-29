cmake_minimum_required(VERSION 3.3.0)

find_package(lcm REQUIRED)
include(${LCM_USE_FILE})
include(GenerateExportHeader)

find_package(PythonInterp)
find_package(Java)

if(PYTHONINTERP_FOUND)
  set(python_args PYTHON_SOURCES python_install_sources)
endif()

if(JAVA_FOUND)
  include(UseJava)
  set(java_args JAVA_SOURCES java_sources)
endif()

# Get all .lcm files in this directory.
file(GLOB LCMTYPE_FILES "*.lcm")

lcm_wrap_types(
  C_EXPORT vehicle_lcmtypes
  C_SOURCES c_sources
  C_HEADERS c_install_headers
  CPP_HEADERS cpp_install_headers
  ${python_args}
  ${java_args}
  ${LCMTYPE_FILES}
)

# Add the C lcmtypes library.
lcm_add_library(vehicle_lcmtypes C ${c_sources} ${c_install_headers})
generate_export_header(vehicle_lcmtypes)
target_include_directories(vehicle_lcmtypes INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

# Add the C++ lcmtypes library.
lcm_add_library(vehicle_lcmtypes_cpp CPP ${cpp_install_headers})
target_include_directories(vehicle_lcmtypes_cpp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

# Install c headers.
lcm_install_headers(DESTINATION include
  ${CMAKE_CURRENT_BINARY_DIR}/vehicle_lcmtypes_export.h
  ${c_install_headers}
  ${cpp_install_headers}
)

# Install Python bindings.
if(PYTHONINTERP_FOUND)
  lcm_install_python(${python_install_sources})
endif()

# Build and install a JAR for Java support.
if (JAVA_FOUND)
  add_jar(vehicle_lcmtypes_jar
    OUTPUT_NAME vehicle_lcmtypes
    INCLUDE_JARS lcm-java
    SOURCES ${java_sources}
  )
  install_jar(vehicle_lcmtypes_jar share/java)
endif()

# NOTE(milo): Don't want to install these to the system for now.
# install(TARGETS vehicle_lcmtypes vehicle_lcmtypes_cpp
#   EXPORT ${PROJECT_NAME}Targets
#   RUNTIME DESTINATION bin
#   LIBRARY DESTINATION lib${LIB_SUFFIX}
#   ARCHIVE DESTINATION lib${LIB_SUFFIX}
#   INCLUDES DESTINATION include
# )
