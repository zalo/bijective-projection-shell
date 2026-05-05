cmake_minimum_required(VERSION 3.13)
project(prism)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(PRISM_EXTERNAL ${CMAKE_CURRENT_SOURCE_DIR}/external/)

option(PRISM_WASM "Build for WASM (no CGAL/embree/HDF5/TBB/geogram)" OFF)

if(PRISM_WASM)
  add_compile_definitions(PRISM_NO_CGAL PRISM_NO_HDF5 IGL_PARALLEL_FOR_FORCE_SERIAL)
endif()

include(PrismDependencies)

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  if(NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG AND NOT PRISM_WASM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -flto")
  endif()
endif()

set(PRISM_LIBRARY_SOURCES
    src/correspondence.cpp
    src/prism/PrismCage.cpp
    src/prism/extraction.cpp
    src/prism/feature_utils.cpp
    src/prism/cage_utils.cpp
    src/prism/bevel_utils.cpp
    src/prism/phong/projection.cpp
    src/prism/phong/tangent_orientation.cpp
    src/prism/phong/query_correspondence.cpp
    src/prism/predicates/inside_prism_tetra.cpp
    src/prism/predicates/inside_octahedron.cpp
    src/prism/predicates/positive_prism_volume_12.cpp
    src/prism/predicates/triangle_triangle_intersection.cpp
    src/prism/predicates/tetrahedron_overlap.cpp
    src/prism/predicates/orient_robust.cpp
    src/prism/energy/prism_quality.cpp
    src/prism/energy/map_distortion.cpp
    src/prism/energy/smoother_pillar.cpp
    src/prism/local_operations/remesh_pass.cpp
    src/prism/local_operations/smooth_pass.cpp
    src/prism/local_operations/remesh_pass_collapse.cpp
    src/prism/local_operations/section_remesh.cpp
    src/prism/local_operations/mesh_coloring.cpp
    src/prism/local_operations/local_mesh_edit.cpp
    src/prism/local_operations/validity_checks.cpp
    src/prism/local_operations/retain_triangle_adjacency.cpp
    src/prism/spatial-hash/AABB_hash.cpp
    src/prism/spatial-hash/self_intersection.cpp
    src/prism/osqp/osqp_normal.cpp
    src/prism/cage_check.cpp
    src/prism/intersections.cpp
    external/robust_predicates/predicates.c
)

if(PRISM_WASM)
  list(APPEND PRISM_LIBRARY_SOURCES
       src/prism/geogram/geogram_utils_stub.cpp
       src/prism/geogram/AABB_libigl.cpp)
else()
  list(APPEND PRISM_LIBRARY_SOURCES
       src/prism/geogram/geogram_utils.cpp
       src/prism/geogram/AABB.cpp
       src/prism/cgal/polyhedron_self_intersect.cpp
       src/prism/cgal/triangle_triangle_intersection.cpp
       src/prism/cgal/tetrahedron_tetrahedron_intersection.cpp
       src/prism/cgal/QP.cpp)
endif()

add_library(prism_library ${PRISM_LIBRARY_SOURCES})

target_compile_features(prism_library PUBLIC cxx_std_17)
target_include_directories(prism_library PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/
    ${CMAKE_CURRENT_SOURCE_DIR}/external/)

if(PRISM_WASM)
  target_link_libraries(prism_library PUBLIC spdlog igl::core osqp mitsuba_autodiff)
else()
  target_link_libraries(prism_library PUBLIC spdlog igl::core igl::cgal osqp highfive geogram mitsuba_autodiff)
endif()

option(PYPRISM "Activate Python Bindings" OFF)
if(PYPRISM)
  find_package(PythonInterp QUIET)
  prism_download_pybind11()
  add_subdirectory(${PRISM_EXTERNAL}/pybind11/ pybind11)
  add_subdirectory(src/seism/python)
endif()

option(PRISM_TESTS "Unit Tests" ON)
if(PRISM_TESTS AND NOT PRISM_WASM)
  include(CTest)
  add_subdirectory(tests)
endif()

if(NOT PRISM_WASM)
  add_executable(breve_bin src/main.cpp
      src/construct_shell.cpp
      src/remesh_section_in_shell.cpp
      src/getRSS.c
  )
  target_link_libraries(breve_bin prism_library CLI11::CLI11 json)
endif()

if(PRISM_WASM)
  add_executable(prism_full_wasm wasm/prism_full_wasm.cpp)
  target_link_libraries(prism_full_wasm PRIVATE prism_library)
  set_target_properties(prism_full_wasm PROPERTIES
      OUTPUT_NAME "prism_full_wasm"
      SUFFIX ".js")
  target_link_options(prism_full_wasm PRIVATE
      "-lembind"
      "-sMODULARIZE=1"
      "-sEXPORT_ES6=1"
      "-sEXPORT_NAME=PrismWASM"
      "-sALLOW_MEMORY_GROWTH=1"
      "-sENVIRONMENT=web"
      "-sINITIAL_MEMORY=64MB"
      "-sSTACK_SIZE=4MB"
      "-sASSERTIONS=1"
      "-sDISABLE_EXCEPTION_CATCHING=0")
endif()
