cmake_minimum_required(VERSION 3.18)
project(cuda_path_tracer LANGUAGES CUDA CXX)
add_subdirectory(../path_tracer)
add_subdirectory(../stream_compaction)