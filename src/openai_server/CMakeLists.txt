# SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

include(FetchContent)
set(HTTPLIB_VERSION v0.18.3)
FetchContent_Declare(
  cpphttplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG ${HTTPLIB_VERSION})
FetchContent_GetProperties(cpphttplib)
if(NOT cpphttplib_POPULATED)
  FetchContent_Populate(cpphttplib)
endif()

set(CMAKE_BUILD_TYPE Debug)
add_library(cpphttplib INTERFACE)
target_include_directories(cpphttplib SYSTEM INTERFACE ${cpphttplib_SOURCE_DIR})

add_executable(llm_openai_server main.cpp llm_session.cpp openai_mapping.cpp)
target_include_directories(
  llm_openai_server PRIVATE ${TRT_EDGELLM_ROOT}/examples/utils ${CMAKE_CURRENT_SOURCE_DIR}
                            ${EDGELLM_COMMON_INCLUDE_DIRS})
target_link_libraries(
  llm_openai_server
  PRIVATE llmOnEdgeInputParse
          edgellmCore
          edgellmTokenizer
          exampleUtils
          commonLibraryExt
          cpphttplib
          Threads::Threads)
llm_onedge_cross_link(llm_openai_server)

if(LLMONEDGE_BUILD_OPENAI_SERVER_TESTS)
  enable_testing()
  # 不用 googletest 自带的 add_subdirectory：其 internal_utils 对 cxx_std_17 使用
  # target_compile_features，在旧版 CMake + GCC 13 上会报 “no known features”。
  # 用 gtest-all.cc 融合源手动建库，仅用 CXX_STANDARD 即可。
  if(NOT TARGET gtest)
    set(_LLMONEDGE_GTEST_DIR "${TRT_EDGELLM_ROOT}/3rdParty/googletest/googletest")
    add_library(gtest STATIC "${_LLMONEDGE_GTEST_DIR}/src/gtest-all.cc")
    target_include_directories(
      gtest PUBLIC "${TRT_EDGELLM_ROOT}/3rdParty/googletest/googletest/include" "${_LLMONEDGE_GTEST_DIR}")
    target_link_libraries(gtest PUBLIC Threads::Threads)
    set_target_properties(gtest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
    if(CMAKE_USE_PTHREADS_INIT)
      target_compile_definitions(gtest PUBLIC GTEST_HAS_PTHREAD=1)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(gtest PRIVATE -w)
    endif()

    add_library(gtest_main STATIC "${_LLMONEDGE_GTEST_DIR}/src/gtest_main.cc")
    target_link_libraries(gtest_main PUBLIC gtest)
    set_target_properties(gtest_main PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(gtest_main PRIVATE -w)
    endif()
  endif()
  add_executable(openai_mapping_test tests/test_openai_mapping.cpp openai_mapping.cpp)
  target_include_directories(
    openai_mapping_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                ${CMAKE_CURRENT_SOURCE_DIR}/../llm_input_parse
                                ${EDGELLM_COMMON_INCLUDE_DIRS})
  target_link_libraries(openai_mapping_test PRIVATE gtest gtest_main llmOnEdgeInputParse)
  add_test(NAME openai_mapping_test COMMAND openai_mapping_test)

  set(_LLMONEDGE_OPENAI_E2E_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e_openai_server_cli.cmake")
  add_test(
    NAME openai_server_e2e_cli_help
    COMMAND ${CMAKE_COMMAND} -DBINARY_PATH=$<TARGET_FILE:llm_openai_server> -DCASE_NAME=help -P
            ${_LLMONEDGE_OPENAI_E2E_SCRIPT})
  add_test(
    NAME openai_server_e2e_cli_invalid_log_level
    COMMAND ${CMAKE_COMMAND} -DBINARY_PATH=$<TARGET_FILE:llm_openai_server> -DCASE_NAME=invalid_log_level -P
            ${_LLMONEDGE_OPENAI_E2E_SCRIPT})
  add_test(
    NAME openai_server_e2e_cli_missing_engine_dir
    COMMAND ${CMAKE_COMMAND} -DBINARY_PATH=$<TARGET_FILE:llm_openai_server> -DCASE_NAME=missing_engine_dir -P
            ${_LLMONEDGE_OPENAI_E2E_SCRIPT})
endif()
