#!/bin/bash
SOURCE_ROOT="`pwd`"
LINK_DIR="$SOURCE_ROOT/submodules/links"
if [ ! -d "$LINK_DIR" ]; then
    mkdir "$LINK_DIR"
    ln -s "$SOURCE_ROOT/submodules/data_tools/data_tools" "$LINK_DIR/data_tools"
    ln -s "$SOURCE_ROOT/submodules/fcn_maker/fcn_maker" "$LINK_DIR/fcn_maker"
    ln -s "$SOURCE_ROOT/submodules/ignite/ignite" "$LINK_DIR/ignite"
    echo "Created submodule links in \"$LINK_DIR\"."
fi
export PYTHONPATH="$LINK_DIR:$PYTHONPATH"
echo "Added \"$LINK_DIR\" to PYTHONPATH"
echo "PYTHONPATH=$PYTHONPATH"
