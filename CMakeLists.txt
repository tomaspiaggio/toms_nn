cmake_minimum_required(VERSION 3.20)
project(toms_nn)

file(GLOB SOURCES src/*.c)
add_executable(app main.c ${SOURCES})
target_include_directories(app PRIVATE include)
