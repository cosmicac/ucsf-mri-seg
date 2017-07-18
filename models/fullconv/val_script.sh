#!/bin/bash
#for i in $(seq 0 7);
validx=(0 1 2 3 54 55 56 57 58 59 60)
for i in ${validx[@]}; 
do
  python raseg_predict.py --imgn=$i
done
cd postprocess
python postprocess.py
