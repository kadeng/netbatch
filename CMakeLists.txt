CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(netbatch)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "-g -Wall -Wno-unused-variable")

INCLUDE(../../build/dlib/dlib/cmake)

ADD_SUBDIRECTORY(proto)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})


FIND_PACKAGE(Protobuf REQUIRED)
ADD_CUSTOM_TARGET(${PROJECT_NAME} DEPENDS batchserver pyproto)

ADD_EXECUTABLE( batchserver
                "batchserver.cpp"
              )

ADD_EXECUTABLE( recordfiler
                          "recordfiler.cpp"
                        )

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
