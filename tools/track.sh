#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


set -x
set -o pipefail

args=$(cat configs/motrv2.args)
python3 track_predet.py ${args} --exp_name tracker --resume ./pretrained/motrv2_dancetrack.pth