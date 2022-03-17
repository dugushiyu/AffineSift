CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# Author : Pierre Moulon
# Date : December 2010.
# README :
#          The following cmake works for windows.
#	Openmp is accessible only on the professionnal version of Visual Studio.
# 	In order to use OPENMP in visual your have to add the preprocessor _OPENMP
# 	and enable OPENMP library in C/C++/Language.
PROJECT(ASIFT)
string(REGEX REPLACE "(.+)/asift_match.*" "\\1" PROJECT_INIT_PATH ${PROJECT_SOURCE_DIR})
message("-- PROJECT_SOURCE_DIR path in ${PROJECT_SOURCE_DIR}")

#ADD_SUBDIRECTORY(libMatch)
ADD_SUBDIRECTORY(libNumerics)

FIND_PACKAGE(OpenMP) 
if (OPENMP_FOUND) 
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)

IF(MSVC)
   ADD_DEFINITIONS(/arch:SSE2)
ENDIF(MSVC)

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory( ${PROJECT_SOURCE_DIR}/src ASIFT_srcs)

#set(ASIFT_srcs
	#numerics1.cpp frot.cpp splines.cpp fproj.cpp
	#library.cpp flimage.cpp filter.cpp
#	demo_lib_sift.cpp compute_asift_keypoints.cpp
	#compute_asift_matches.cpp
	#orsa.cpp demo_ASIFT.cpp	
	#io_png/io_png.c
#	)
	
include_directories(demo_ASIFT
	${OpenCV_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/include
	)
					
add_executable(demo_ASIFT ${ASIFT_srcs})
TARGET_LINK_LIBRARIES(demo_ASIFT Numerics  opencv_world341d opencv_world341 
# Match
 )
