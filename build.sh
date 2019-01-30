#!/bin/sh -ex

g++ facetracker.cpp $(pkg-config --cflags --libs /usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)/pkgconfig/opencv.pc) -lpthread -g -o facetracker -Wall -Werror
