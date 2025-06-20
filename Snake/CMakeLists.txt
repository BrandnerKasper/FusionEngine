cmake_minimum_required(VERSION 3.31)

project(Snake)

set(CMAKE_CXX_STANDARD 23)

add_executable(Snake
        main.cpp
        random.h
        Player.cpp
        Player.h
        input.cpp
        input.h
        types.h
        Board.cpp
        Board.h)