#!/bin/bash
SOURCE_ROOT="`pwd`"
export PYTHONPATH="$SOURCE_ROOT/submodules/data_tools/:$PYTHONPATH"
export PYTHONPATH="$SOURCE_ROOT/submodules/fcn_maker/:$PYTHONPATH"
export PYTHONPATH="$SOURCE_ROOT/submodules/ignite/:$PYTHONPATH"
echo "PYTHONPATH=$PYTHONPATH"
