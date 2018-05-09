#!/bin/bash
if [ ! -d roboschool-files/bullet3 ]; then
    echo "Roboschool files should be downloaded first!"
    exit 1
fi
cd roboschool-files
ROBOSCHOOL_PATH=$(pwd)/roboschool-master
mkdir bullet3/build
cd    bullet3/build
cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=$ROBOSCHOOL_PATH/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF ..
make -j4
make install
cd ../..
pip3 install -e $ROBOSCHOOL_PATH
