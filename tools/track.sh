#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


set -x
set -o pipefail

args=$(cat configs/motrv2.args)
# python3 submit_dance.py ${args} --exp_name tracker --resume ./pretrained/motrv2_dancetrack.pth --output_dir /home/nathan.candre/other/MOTRv2_results
# python3 track.py ${args} --exp_name tracker --resume ./pretrained/motrv2_dancetrack.pth -i $1 -o $2k
python3 track.py ${args} --exp_name tracker --resume ./pretrained/motrv2_dancetrack.pth -i /data/stot/datasets_mot/fmv/frames/M1102_m5 -o /home/nathan.candre/other/MOTRv2_results
