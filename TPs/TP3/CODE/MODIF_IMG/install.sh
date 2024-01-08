#!/bin/bash

cd FreeImage
tar xzvf FreeImage3180.tar.gz
cd FreeImage
make
make install
export LD_LIBRARY_PATH=${HOME}/softs/FreeImage/lib:$LD_LIBRARY_PATH
