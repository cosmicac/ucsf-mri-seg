#!/bin/bash
for i in {0..19}
do
   python3 raseg_eval.py --imgn=11 --depthn=$i
done
