#!/bin/bash

python=/home/hongbin/.conda/envs/workspace/bin/python
date > out_CV

cd main/
(time nohup $python CV_template.py &) >>../out_CV 2>&1

