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


def _validate_model_options(model_name, object_to_predict, 
                            model_types = ['yolov3', 'retinanet', 'fasterrcnn'],
                            object_types = ['tablas', 'cardinalidades']):
    if model_name.lower() not in model_types:
        raise ValueError(f"Opción no válida. Modelos disponibles: {model_types}")
    if object_to_predict.lower() not in object_types:
        raise ValueError(f"Opción no válida. Los modelos disponibles son para los objetos: {object_types}")


def choose_model(model_name, object_to_predict):
    print("Por ahora los modelos están en path diferentes. TODO: Habría que unificar.")
    
    PATH_MODELS = "/home/nacho/TFI-Cazcarra/data/models"
    PATH_YOLO = "/home/nacho/TFI-Cazcarra/yolov3/runs/train"
    
    _validate_model_options(model_name, object_to_predict)
    
    if model_name == "yolov3":
        num_exp = "exp9" if object_to_predict == "cardinalidades" else "exp5"
        return torch.hub.load('ultralytics/yolov5', 'custom', os.path.join(PATH_YOLO, 
                                    num_exp, "weights", f"best_{object_to_predict}.pt"))
    else:
        return load_model(os.path.join(PATH_MODELS, f"model_best_{object_to_predict}_{model_name}.pt"))







