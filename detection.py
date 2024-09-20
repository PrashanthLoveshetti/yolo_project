import argparse
import torch
from yolov5.models.yolo import Model
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
from pathlib import Path
import cv2

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/exp/weights/best.pt', help='model path')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # path to test images
    parser.add_argument('--img', type=int, default=640, help='image size')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()

def main(opt):
    # Select device
    device = select_device(opt.device)
    
    # Load the model
    model = torch.load(opt.weights, map_location=device)['model'].float()
    model.to(device).eval()
    
    # Load images
    dataset = LoadImages(opt.source, img_size=opt.img)
    
    # Run inference
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0  # Normalize to [0, 1]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        pred = model(img)[0]
        pred = non_max_suppression(pred, opt.conf, 0.45)  # Apply NMS
        
        # Process detections
        for i, det in enumerate(pred):
            p, s, im0 = path, '', im0s
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label)
            
            cv2.imshow(p, im0)
            cv2.waitKey(1)
    
    print('Detection completed.')

if _name_ == "_main_":
    opt = parse_opt()
    main(opt)