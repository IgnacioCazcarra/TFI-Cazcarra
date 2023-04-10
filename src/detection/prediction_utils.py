from ..training_utils.training import load_model
from ..constants import PATH
import math
import os
import cv2
import logging
from PIL import Image
import numpy as np
import torch
import torchvision

logging.basicConfig(level = logging.INFO)


def filter_predictions(predictions, score_threshold=0.5, nms_threshold=0.5, return_labels=False):
    boxes = predictions['boxes'][predictions['scores'] >= score_threshold]
    scores = predictions['scores'][predictions['scores'] >= score_threshold]
    labels = predictions['labels'][predictions['scores'] >= score_threshold]
    valid_idx = torchvision.ops.nms(boxes, scores, nms_threshold)
    if return_labels:
        return boxes[valid_idx], scores[valid_idx], labels[valid_idx]
    return boxes[valid_idx], scores[valid_idx]


def draw_bbox(img, xmin, ymin, xmax, ymax, color=(255,0,0), thickness=1, score=None):
    if score:
        try:
            THICKNESS_SCALE = 3e-3 
            FONT_SCALE = 2e-3  # Adjust for larger font size in all images
            w,h = img.shape[:2]
            img = cv2.putText(img, str(round(score,2)), (int(xmax)-1,int(ymin)-1), 
                          cv2.FONT_HERSHEY_SIMPLEX, fontScale=min(w, h)*FONT_SCALE, color=color, 
                          thickness=math.ceil(min(w, h) * THICKNESS_SCALE), lineType=cv2.LINE_AA)
        except Exception as e:
            logging.warn(e)
            logging.warn("No se pudo graficar el puntaje de la predicción debido a su posición. Salteando..")
    return cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), 
                         color, thickness)


def visualize_boxes(img, boxes, scores=None, **kwargs):
    img = np.array(img)
    if scores is None:
        scores = [None]*len(boxes)
    for b,s in zip(boxes, scores):
        xmin = b[0]
        ymin = b[1]
        xmax = b[2]
        ymax = b[3]
        img = draw_bbox(img, xmin, ymin, xmax, ymax, score=s, **kwargs)
    return Image.fromarray(img)


def _validate_model_options(object_to_predict, object_types = ['tablas', 'cardinalidades']):
    if object_to_predict.lower() not in object_types:
        raise ValueError(f"Opción no válida. Las opciones son: {object_types}")


def get_model(object_to_predict, models_path=os.path.join(PATH, "models")):    
    _validate_model_options(object_to_predict)

    if object_to_predict == "tablas":
        return load_model(os.path.join(models_path, f"retinanet_tablas.pt"))
    else:
        return torch.hub.load('ultralytics/yolov5', 'custom', 
                              os.path.join(models_path, "yolo_cardinalidades.pt"), verbose=False)






