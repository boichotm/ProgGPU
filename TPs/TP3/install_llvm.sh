#!/bin/bash

export WORKDIR=
export LLVM_SRC_PREFIX=
cd $LLVM_SRC_PREFIX

export LLVM_PATH=$WORKDIR/llvm/17.x
export build_path=$LLVM_SRC_PREFIX/build_llvm

#rm -rf $build_path
echo $build_path
mkdir -p $build_path
cd $build_path

#LLVM 17 config
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$LLVM_PATH -DCMAKE_C_COMPILER=`which gcc` -DCMAKE_CXX_COMPILER=`which g++` -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_PROJECTS="clang;compiler-rt;openmp" $LLVM_SRC_PREFIX/llvm

make -j && make install
