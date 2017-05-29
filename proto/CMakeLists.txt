INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf REQUIRED)
INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})
PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER message.proto)
ADD_LIBRARY(proto ${PROTO_HEADER} ${PROTO_SRC})

add_custom_command(OUTPUT message_pb2.py
    COMMAND protoc --python_out=. --proto_path=${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/message.proto
    COMMAND cp ./message_pb2.py ${CMAKE_CURRENT_SOURCE_DIR}/../python
    DEPENDS message.proto
    COMMENT "Generating Python Protobuf Sources"
    VERBATIM
)

add_custom_target(pyproto DEPENDS message_pb2.py)
