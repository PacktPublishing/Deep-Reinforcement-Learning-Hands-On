#!/bin/bash
rm -rf roboschool-files
mkdir roboschool-files
cd roboschool-files
wget https://github.com/openai/roboschool/archive/master.zip 
unzip master.zip
wget https://github.com/olegklimov/bullet3/archive/roboschool_self_collision.zip
unzip roboschool_self_collision.zip
mv bullet3-roboschool_self_collision bullet3
cd ..
