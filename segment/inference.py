import os
import sys
import json
import torch
import argparse
from pathlib import Path

YOLO5_REPO_PATH = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'ultralytics_yolov5_master')
if not os.path.exists(YOLO5_REPO_PATH):
    os.system("git clone https://github.com/ultralytics/yolov5 {}".format(YOLO5_REPO_PATH))

sys.path.append(YOLO5_REPO_PATH)
    

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (Profile, check_file, check_img_size, increment_path, non_max_suppression, scale_boxes, scale_segments)
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device, smart_inference_mode


def load_model(weights):
    """ 
    Loads the model from the weights file

    Args:
        weights (str): path to weights file
    
    Returns:
        model (DetectMultiBackend): the model loaded from the weights file
    """
    model = DetectMultiBackend(weights)
    return model

@smart_inference_mode()
def run(
    source='data/images',  # file/dir/URL/glob/screen/0(webcam)
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    project='runs/predict-seg',  # save results to project/name
    model=None,
):
    source = str(source)
    segments_output = {}
    segments_output['segments'] = []
    segments_output['image_path'] = source
    segments_output['num_detections'] = 0

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    if is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = Path(project)
    os.makedirs(save_dir, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride="")

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            if len(det):
                segments_output['num_detections'] = len(det)
                
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                segments = [
                    scale_segments(im.shape[2:], x, im0.shape, normalize=True)
                        for x in reversed(masks2segments(masks))]

                # TODO: need to change if there are more than 1 classes in the future
                for j, _ in enumerate(reversed(det[:, :6])):
                    seg = segments[j]
                    segments_output['segments'].append(seg.tolist())

    return segments_output


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="", help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default='runs/predict-seg', help='save results to project/name')
    parser.add_argument('--model', action='store_true', help='whether to load model')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(source, filename):
    opt = parse_opt()
    equipment_segmentation_weight = ""
    project= "output"

    model = load_model(equipment_segmentation_weight)

    opt.project = project
    opt.model = model
    opt.source = source

    segmentation_output = run(**vars(opt))

    with open(os.path.join(opt.project, filename + ".json"), 'w') as f:
        json.dump(segmentation_output, f)
    return segmentation_output


def get_equipment_segmentation(image_path, filename):
    """ Get the equipment segmentation from the image path 
    
    Args:
        image_path (str): path to the image
        filename (str): filename of the image

    Returns:
        segmentation_output (dict): dictionary containing the segmentation output

    {
        "segments": [
            [0.005, 0.001],
            [0.006, 0.002],
            [0.007, 0.003],
        ],
        [
            [0.005, 0.001],
            [0.006, 0.002],
            [0.007, 0.003],
        ]
        "image_path": "data/images/1.jpg",
        "num_detections": 2
    }
    """ 
    segmentation_output = main(source=image_path, filename=filename)
    return segmentation_output
