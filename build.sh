#!/bin/sh -ex

g++ tracking.cpp $(pkg-config --cflags --libs /usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)/pkgconfig/opencv.pc) -g -o tracking -Wall -Werror
