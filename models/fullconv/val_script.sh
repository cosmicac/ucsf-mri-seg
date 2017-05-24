#!/bin/bash
#for i in $(seq 0 7);
#do
#  python raseg_predict.py --imgn=$i
#done
cd postprocess
python postprocess.py
