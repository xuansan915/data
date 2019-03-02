cmake_minimum_required(VERSION 2.8)
project(FaceDetectMtcnn)

set(CMAKE_PREFIX_PATH "/usr/local")
#set(CMAKE_BUILD_TYPE Release)
SET(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -pthread -fopenmp" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O0 -pthread -Wall -g -ggdb  -fopenmp" )

#include(../../include/dlib-19.7/dlib/cmake)

# Requires OpenCV v3.0.0 or later
FIND_PACKAGE( OpenCV REQUIRED )
IF (${OpenCV_VERSION} VERSION_LESS 3.0.0)
    MESSAGE(FATAL_ERROR "OpenCV version is not compatible : ${OpenCV_VERSION}. This requires atleast OpenCV v3.0.0")
ENDIF()

include_directories(
   /home/jerry/tensorflow-r1.13
   /home/jerry/tensorflow-lite/downloads
   /home/jerry/tensorflow-lite/downloads/flatbuffers/include
   /usr/local/include/eigen3

)


link_directories(/home/jerry/tensorflow-r1.13/tensorflow/lite/tools/make/gen/linux_x86_64/lib)

add_definitions(-DUSE_SSE2_INSTRUCTIONS=ON -DUSE_SSE4_INSTRUCTIONS=ON -DUSE_AVX_INSTRUCTIONS=ON)

AUX_SOURCE_DIRECTORY(./ DIR_SRCS)

add_executable( FaceDetectMtcnn ${DIR_SRCS} ${MTCNN_SRCS} ${UTILS})
#target_link_libraries( FaceDetectMtcnn ${OpenCV_LIBS}  tensorflow_cc tensorflow_framework glog gflags)
target_link_libraries( FaceDetectMtcnn ${OpenCV_LIBS}  tensorflow-lite dl)
