import torch
import cv2

model1 = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# im = 'https://ultralytics.com/images/zidane.jpg'

im = cv2.imread("/data/stot/datasets_mot/fmv/frames/city_pexels-kabirou-kanlanfeyi-9850032/001.png")

def yolov5_infer(model, img):
    results = model(img)
    
    # results.save()
    
    results = results.xywhn[0]
    
    visdrone_classes = [2, 3, 5, 7]

    visdrone_results = torch.stack([res for res in results if res[-1] in visdrone_classes])
    # visdrone_results = results[torch.where(results[:, -1] == 0)]
    # det_results = visdrone_results[:, :-1]
    det_results = visdrone_results
    
    return det_results
    
res = yolov5_infer(model1, im)
print(res)