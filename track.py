# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from copy import deepcopy
import json

import os
import sys
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from motmodels import build_model
from util.tool import load_model
from main import get_args_parser

from motmodels.structures import Instances
from torch.utils.data import Dataset, DataLoader

class YOLOv5():
    def __init__(self, repo = 'ultralytics/yolov5', variation = 'yolov5s', device = "cpu", verbose=False):
        self.model = torch.hub.load(repo, variation, device = device, _verbose = verbose)
        
    def infer(self, img):
        results = self.model(img).xywhn[0] #xcenter, ycenter, width, height (normalized)
        
        visdrone_classes = [2, 3, 5, 7]
        
        try:
            # visdrone_results = results[torch.where(results[:, -1] == 115)]
            # print(visdrone_results)
            visdrone_results = torch.stack([res for res in results if res[-1] in visdrone_classes])
            det_results = visdrone_results[:, :-1]            
        except:
            det_results = torch.empty((0,5))
        
                
        return det_results
    
class ListImgDataset(Dataset):
    def __init__(self, img_list, detector) -> None:
        super().__init__()
        self.img_list = img_list

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.detector = detector

    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(f_path)
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        proposals = []
        im_h, im_w = cur_img.shape[:2]

        proposals = self.detector.infer(cur_img)
        
        return cur_img, proposals

    def init_img(self, img, proposals):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img, proposals

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, proposals = self.load_img_from_file(self.img_list[index])
        return self.init_img(img, proposals)


class Tracker(object):
    def __init__(self, args, model, vid, output_path, detector):
        self.args = args
        self.detr = model
        
        self.detector = detector

        self.vid = vid
        
        self.seq_num = os.path.basename(vid)

        img_list = [os.path.join(vid, img) for img in os.listdir(vid) if not os.path.isdir(os.path.join(vid, img))]
        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)

        self.predict_path = os.path.join(output_path, args.exp_name)
        
        os.makedirs(self.predict_path, exist_ok=True)

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    def detect(self, prob_threshold=0.6, area_threshold=100, vis=False):
        total_dts = 0
        total_occlusion_dts = 0

        track_instances = None
        loader = DataLoader(ListImgDataset(self.img_list, self.detector), 1, num_workers=2)
        lines = []
        for i, data in enumerate(tqdm(loader)):
            cur_img, ori_img, proposals = [d[0] for d in data]
            # cur_img, proposals = cur_img.cuda(), proposals.cuda()

            # # track_instances = None
            # if track_instances is not None:
            #     track_instances.remove('boxes')
            #     track_instances.remove('labels')
            seq_h, seq_w, _ = ori_img.shape

            # res = self.detr.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
            # track_instances = res['track_instances']

            # dt_instances = deepcopy(track_instances)

            # # filter det instances by score.
            # dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
            # dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

            # total_dts += len(dt_instances)

            # bbox_xyxy = dt_instances.boxes.tolist()
            # identities = dt_instances.obj_idxes.tolist()

            save_format = '{frame} {id} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f} 1 -1 -1 -1\n'
            # for xyxy, track_id in zip(bbox_xyxy, identities):
            #     if track_id < 0 or track_id is None:
            #         continue
            #     x1, y1, x2, y2 = xyxy
            #     w, h = x2 - x1, y2 - y1
            #     lines.append(save_format.format(frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h))
                
            for proposal in proposals:
                lines.append(save_format.format(frame=i + 1, id=1, x1=proposal[0] * seq_w, y1=proposal[1] * seq_h, w=proposal[2] * seq_w, h=proposal[3] * seq_h))
                
        with open(os.path.join(self.predict_path, f'{self.seq_num}.txt'), 'w') as f:
            f.writelines(lines)
        print("totally {} dts {} occlusion dts".format(total_dts, total_occlusion_dts))

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)
    parser.add_argument('-i', '--input_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    torch.multiprocessing.set_start_method("spawn")    
    sys.path.append("./yolov5") 

    # load model and weights
    detr, _, _ = build_model(args)
    detr.track_embed.score_thr = args.update_score_threshold
    detr.track_base = RuntimeTrackerBase(args.score_threshold, args.score_threshold, args.miss_tolerance)
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.cuda()

    input_path, output_path = args.input_path, args.output_path
    
    yolo = YOLOv5(device = "cpu")
    
    det = Tracker(args, model=detr, vid=input_path, output_path=output_path, detector = yolo)
    det.detect(args.score_threshold)