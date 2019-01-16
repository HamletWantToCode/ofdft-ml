#!/bin/bash

python=/home/hongbin/.conda/envs/workspace/bin/python
mpirun=/home/hongbin/.conda/envs/workspace/bin/mpirun
date > out

cd main/
(time nohup $mpirun --host 192.168.0.103 -n 10 $python database.py &) >>../out 2>&1

