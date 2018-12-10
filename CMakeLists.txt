cmake_minimum_required(VERSION 3.6)
project(ccs-l3-coursework-2018)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES test.c optimised-sparsemm.c basic-sparsemm.c utils.c)

add_executable(ccs-l3-coursework-2018 ${SOURCE_FILES})