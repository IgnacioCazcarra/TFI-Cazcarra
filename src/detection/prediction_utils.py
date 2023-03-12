from ..training_utils.training import load_model

import os
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision


def filter_predictions(predictions, score_threshold=0.5, nms_threshold=0.5):
    boxes = predictions['boxes'][predictions['scores'] >= score_threshold]
    scores = predictions['scores'][predictions['scores'] >= score_threshold]
    valid_idx = torchvision.ops.nms(boxes, scores, nms_threshold)
    return boxes[valid_idx], scores[valid_idx]


def draw_bbox(img, xmin, ymin, xmax, ymax, color=(255,0,0), thickness=1): 
    return cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), 
                         color, thickness)


def visualize_boxes(img, boxes, **kwargs):
    img = np.array(img)
    for b in boxes:
        xmin = b[0]
        ymin = b[1]
        xmax = b[2]
        ymax = b[3]
        img = draw_bbox(img, xmin, ymin, xmax, ymax, **kwargs)
    return Image.fromarray(img)


def _validate_model_options(object_to_predict, object_types = ['tablas', 'cardinalidades']):
    if object_to_predict.lower() not in object_types:
        raise ValueError(f"Opción no válida. Las opciones son: {object_types}")


def get_model(object_to_predict, models_path="/home/nacho/TFI-Cazcarra/models"):    
    _validate_model_options(object_to_predict)

    if object_to_predict == "tablas":
        return load_model(os.path.join(models_path, f"retinanet_tablas.pt"))
    else:
        return torch.hub.load('ultralytics/yolov5', 'custom', 
                              os.path.join(models_path, "yolo_cardinalidades.pt"))






