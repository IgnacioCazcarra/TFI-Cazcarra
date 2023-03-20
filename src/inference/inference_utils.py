import os
import yaml
import logging
from PIL import Image
from torchvision import transforms as T

from ..detection.prediction_utils import get_model, filter_predictions, visualize_boxes
from ..slides_utils.slides_utils import predict_tiles
from ..line_detection.hough import get_pairs
from ..ocr_utils.ocr import get_ocr_model, predict_ocr, generate_db

logging.basicConfig(level = logging.INFO)


def save_sql(sql_code, path_to_save):
    name, ext = os.path.splitext(path_to_save)
    if not ext:
        path_to_save += ".sql"
    with open(path_to_save, 'w') as f:
        f.write(sql_code)
    logging.info(f"Guardado correctamente en {path_to_save}")


def read_yaml(yaml_path="/home/nacho/TFI-Cazcarra/inference_params.yaml"):
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def prediction_wrapper(img_path, path_to_save, yaml_path, plot):
    config = read_yaml(yaml_path)
    transform = T.Compose([T.ToTensor()])

    model_tablas = get_model(object_to_predict="tablas")
    model_cardinalidades = get_model(object_to_predict="cardinalidades")
    model_ocr = get_ocr_model()

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)

    logging.info("Prediciendo tablas...")
    tablas_pred = model_tablas([img_tensor])[1][0]
    tablas_boxes, tablas_scores = filter_predictions(tablas_pred, 
                                         nms_threshold=config['tablas']['nms_threshold'], 
                                         score_threshold=config['tablas']['score_threshold'])
    
    logging.info("Prediciendo cardinalidades...")
    cardinalidades_pred = predict_tiles(img, model=model_cardinalidades, is_yolo=True, 
                                        transform=transform)
    cardinalidades_boxes, cardinalidades_scores = filter_predictions(cardinalidades_pred, 
                                         nms_threshold=config['cardinalidades']['nms_threshold'], 
                                         score_threshold=config['cardinalidades']['score_threshold'])
    
    if plot:
        img2 = visualize_boxes(img, tablas_boxes, color=(0,150,255), thickness=3, 
                               scores=tablas_scores.tolist())
        img2 = visualize_boxes(img2, cardinalidades_boxes, color=(0,0,255), thickness=3, 
                            scores=cardinalidades_scores.tolist())
        basepath_to_save, _ = os.path.splitext(path_to_save)
        path_to_save_img = basepath_to_save + ".png"
        path_to_save_conexiones = basepath_to_save + "_conexiones.png"
        img2.save(path_to_save_img)

    logging.info("Generando conexiones...")
    conexiones = get_pairs(tablas_boxes, cardinalidades_boxes, img=img, offset_tablas=config['tablas']['offset'], plot=plot, path_to_save_conexiones=path_to_save_conexiones)

    logging.info("Reconociendo texto...")
    tablas_boxes = tablas_boxes.detach().numpy().astype(int)
    all_tables, tables_names = predict_ocr(img=img, tablas=tablas_boxes, 
                                        ocr_model=model_ocr, scale_percent=config['ocr']['reescale_percent'], lang=config['ocr']['lang'])
    
    logging.info("Generando codigo SQL...")
    code = generate_db(conexiones, all_tables, tables_names, config['ocr']['lang'])
    save_sql(code, path_to_save)