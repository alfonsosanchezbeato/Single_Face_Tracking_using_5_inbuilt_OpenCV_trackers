#!/bin/sh -ex

g++ tracking.cpp $(pkg-config --cflags --libs /usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)/pkgconfig/opencv.pc) -lpthread -g -o tracking -Wall -Werror
