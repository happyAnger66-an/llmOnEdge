# SPDX-License-Identifier: Apache-2.0

add_library(
  llmOnEdgeMemory STATIC
  src/memory_types.cpp
  src/memory_stats.cpp
  src/element_type.cpp
  src/buffer.cpp
  src/buffer_manager.cpp
  src/tensor.cpp)

target_include_directories(llmOnEdgeMemory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(llmOnEdgeMemory PUBLIC CUDA::cudart)
set_target_properties(llmOnEdgeMemory PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(LLMONEDGE_BUILD_EDGE_LLM_TESTS "Build edgeLLM memory unit tests and e2e" ON)

if(LLMONEDGE_BUILD_EDGE_LLM_TESTS)
  enable_testing()

  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0)
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(
    edge_memory_unit_tests
    tests/unit/test_memory_stats.cpp
    tests/unit/test_buffer.cpp
    tests/unit/test_tensor.cpp
    tests/unit/test_buffer_manager.cpp)
  target_link_libraries(edge_memory_unit_tests PRIVATE llmOnEdgeMemory GTest::gtest_main CUDA::cudart)
  target_include_directories(
    edge_memory_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/tests
                                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit)
  add_test(NAME edge_memory_unit_tests COMMAND edge_memory_unit_tests)

  add_executable(edge_memory_e2e tests/e2e/memory_e2e.cpp)
  target_link_libraries(edge_memory_e2e PRIVATE llmOnEdgeMemory CUDA::cudart)
  target_include_directories(
    edge_memory_e2e PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/tests
                            ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e)
  add_test(NAME edge_memory_e2e COMMAND edge_memory_e2e)
endif()
