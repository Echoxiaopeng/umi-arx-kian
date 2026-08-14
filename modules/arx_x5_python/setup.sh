#!/bin/bash

# 设置 C++ 动态库路径
export LD_LIBRARY_PATH=$(pwd)/bimanual/api/arx_x5_src:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# 设置 Python 模块路径，让 Python 能找到 arx_x5_python 模块
export PYTHONPATH=$(pwd)/bimanual/api/arx_x5_python:$PYTHONPATH
