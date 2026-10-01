cmake_minimum_required(VERSION 3.14)
project(MiniANN_Early VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

include_directories(include)

add_library(miniann_early
  src/activation.cpp
  src/neuron.cpp
  src/layer.cpp
  src/network.cpp
  src/training.cpp
  src/dataset.cpp
)

add_executable(activation_demo demos/activation_demo.cpp)
target_link_libraries(activation_demo miniann_early)

add_executable(forward_demo demos/forward_demo.cpp)
target_link_libraries(forward_demo miniann_early)

add_executable(test_activation tests/test_activation.cpp)
target_link_libraries(test_activation miniann_early)
