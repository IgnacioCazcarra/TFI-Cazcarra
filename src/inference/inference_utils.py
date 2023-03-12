import os
import logging
from PIL import Image
from torchvision import transforms as T

from ..detection.prediction_utils import get_model, filter_predictions
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
    logging.info(f"Successfully saved at {path_to_save}..!")


def prediction_wrapper(img_path, path_to_save):
    transform = T.Compose([T.ToTensor()])

    model_tablas = get_model(object_to_predict="tablas")
    model_cardinalidades = get_model(object_to_predict="cardinalidades")
    model_ocr = get_ocr_model()

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)

    logging.info("Prediciendo tablas...")
    tablas_pred = model_tablas([img_tensor])[1][0]
    tablas_boxes, _ = filter_predictions(tablas_pred, nms_threshold=0.5, score_threshold=0.5)
    
    logging.info("Prediciendo cardinalidades...")
    cardinalidades_pred = predict_tiles(img, model=model_cardinalidades, is_yolo=True, 
                                        transform=transform)
    cardinalidades_boxes, _ = filter_predictions(cardinalidades_pred, nms_threshold=0.5, 
                                                                score_threshold=0.5)
    
    logging.info("Generando conexiones...")
    conexiones = get_pairs(tablas_boxes, cardinalidades_boxes, img=img, plot=True)

    logging.info("Reconociendo texto...")
    tablas_boxes = tablas_boxes.detach().numpy().astype(int)
    all_tables, tables_names = predict_ocr(img=img, tablas=tablas_boxes, 
                                        ocr_model=model_ocr, scale_percent=100)
    
    logging.info("Generando codigo SQL...")
    code = generate_db(conexiones, all_tables, tables_names, lang="en")
    save_sql(code, path_to_save)