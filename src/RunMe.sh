#!/bin/bash.sh
mkdir data
cd data
wget http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.tar.gz
tar -xzf coil-100.tar.gz
cd coil-100
perl ../../convertGroupppm2png.pl
cd ../../

mkvirtualenv trals
pip install -r requirements.txt

python -m unittest test.TRALSTest.test1

#python -m unittest test

