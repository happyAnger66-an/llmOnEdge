if(NOT DEFINED BINARY_PATH)
  message(FATAL_ERROR "BINARY_PATH is required")
endif()

if(NOT DEFINED CASE_NAME)
  message(FATAL_ERROR "CASE_NAME is required")
endif()

if(CASE_NAME STREQUAL "help")
  execute_process(
    COMMAND "${BINARY_PATH}" --help
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "Expected --help to exit 0, got ${rc}. stderr: ${err}")
  endif()
  string(FIND "${err}" "--log-level LEVEL" has_log_level)
  string(FIND "${err}" "--no-log" has_no_log)
  if(has_log_level EQUAL -1 OR has_no_log EQUAL -1)
    message(FATAL_ERROR "Help output does not contain new log options. stderr: ${err}")
  endif()
  return()
endif()

if(CASE_NAME STREQUAL "invalid_log_level")
  execute_process(
    COMMAND "${BINARY_PATH}" --engine-dir /tmp/fake-engine --log-level trace
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
  if(rc EQUAL 0)
    message(FATAL_ERROR "Expected invalid log level to fail, but exited 0")
  endif()
  string(FIND "${err}" "Usage:" has_usage)
  if(has_usage EQUAL -1)
    message(FATAL_ERROR "Expected usage message for invalid log level. stderr: ${err}")
  endif()
  return()
endif()

if(CASE_NAME STREQUAL "missing_engine_dir")
  execute_process(
    COMMAND "${BINARY_PATH}" --log-level debug
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
  if(rc EQUAL 0)
    message(FATAL_ERROR "Expected missing --engine-dir to fail, but exited 0")
  endif()
  string(FIND "${err}" "--engine-dir is required" has_required)
  if(has_required EQUAL -1)
    message(FATAL_ERROR "Expected required engine-dir error. stderr: ${err}")
  endif()
  return()
endif()

message(FATAL_ERROR "Unknown CASE_NAME=${CASE_NAME}")
