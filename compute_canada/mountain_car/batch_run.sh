#!/bin/bash

lamda_a=(0 .25 .5 .75 1)
lamda_v=(0 .25 .5 .75 1)

for a in ${lamda_a[@]}
do
  for v in ${lamda_v[@]}
  do
    sbatch compute_canada/mountain_car/run_eac.sh $a $v
    sleep 1
  done
done



