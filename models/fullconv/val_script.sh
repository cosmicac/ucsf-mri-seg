#!/bin/bash
#for i in $(seq 24 30);
#do
#  python raseg_predict.py --imgn=$i
#done
cd postprocess
python postprocess.py
