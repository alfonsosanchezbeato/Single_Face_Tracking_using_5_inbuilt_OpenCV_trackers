#!/bin/sh -ex

g++ tracking.cpp $(pkg-config --cflags --libs /usr/lib/x86_64-linux-gnu/pkgconfig/opencv.pc) -g -o tracking
