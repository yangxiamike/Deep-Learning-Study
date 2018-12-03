#!/bin/sh

#/bin/bash
#BSUB -J cyclegan.py
#BSUB -e /nfsshare/home/xiayang/code/cycle-gan/cycle.err
#BSUB -o /nfsshare/home/xiayang/code/cycle-gan/cycle.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -R "select [ngpus>0] rusage [ngpus_excl_p=4]"
python /nfsshare/home/xiayang/code/cycle-gan/cyclegan.py
