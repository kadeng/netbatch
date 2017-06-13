CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(netbatch)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "-g -Wall -Wno-unused-variable")

SET(EXECUTABLE_OUTPUT_PATH "bin")
SET(CUDA_HOST_COMPILER ${CMAKE_C_COMPILER})

if(CUDA_FOUND)
    # pascal arch
    set(CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61;")
    # fermi arch
    # set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode=arch=compute_20,code=sm_20;")
    # add -Wextra compiler flag for gcc compilations
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wall;")
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -g")
    # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()

# Include dlib
INCLUDE(../dlib/dlib/cmake)

# Include protobuf subproject
ADD_SUBDIRECTORY(proto)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

FIND_PACKAGE(Protobuf REQUIRED)
ADD_CUSTOM_TARGET(${PROJECT_NAME} DEPENDS batchserver pyproto recordfiler jpegNPP)

ADD_EXECUTABLE( batchserver
                "batchserver.cpp"
              )

ADD_EXECUTABLE( recordfiler
                          "recordfiler.cpp"
                        )
                        
SET (jpegNPP_files jpegNPP.cpp Endianess.h)

#CUDA_ADD_EXECUTABLE(jpegNPP ${jpegNPP_files})
#message(${CUDA_TOOLKIT_ROOT_DIR}/samples/7_CUDALibraries/common/FreeImage/include)

#TARGET_INCLUDE_DIRECTORIES( jpegNPP
#		PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/samples/7_CUDALibraries/common/FreeImage/include#
#		PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/samples/7_CUDALibraries/common/UtilNPP
#		PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
#		)
#TARGET_LINK_LIBRARIES(jpegNPP ${CUDA_nppc_LIBRARY} ${CUDA_nppi_LIBRARY} nanomsg)

TARGET_LINK_LIBRARIES(  batchserver
                        nanomsg
                        proto
                        ${PROTOBUF_LIBRARY}
                        dlib::dlib
                      )

TARGET_LINK_LIBRARIES(  recordfiler
                        proto
                        ${PROTOBUF_LIBRARY}
                        dlib::dlib
                        )
                        
       