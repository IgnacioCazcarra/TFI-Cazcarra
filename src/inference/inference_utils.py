import yaml
import pprint
import logging
from torchvision import transforms as T

from ..detection.prediction_utils import get_model, filter_predictions
from ..slides_utils.slides_utils import predict_tiles
from ..line_detection.hough import get_pairs
from ..ocr_utils.ocr import get_ocr_model, predict_ocr, generate_db

logging.basicConfig(level=logging.INFO)


def style_code(sql_code):
    sql_code = (
        pprint.pformat(sql_code, indent=2).strip().replace("'", "").replace("\\n", "")
    )
    if sql_code.startswith("("):
        sql_code = sql_code[1:]
    if sql_code.endswith(")"):
        sql_code = sql_code[:-1]
    return sql_code


def read_yaml(yaml_path):
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def update_yaml(data, yaml_path):
    with open(yaml_path, "w") as file:
        yaml.dump(data, file)


def check_empty_tables(tablas_boxes):
    if len(tablas_boxes) == 0 or tablas_boxes is None:
        raise ValueError(
            "ERROR: No se detectaron tablas. Pruebe bajando los umbrales en el archivo inference_params.yaml"
        )


def api_prediction_wrapper(img, config):
    config = read_yaml(yaml_path="/TFI-Cazcarra/inference_params.yaml")

    transform = T.Compose([T.ToTensor()])

    model_tablas = get_model(object_to_predict="tablas")
    model_cardinalidades = get_model(object_to_predict="cardinalidades")
    model_ocr = get_ocr_model()

    img_tensor = transform(img)

    logging.info("Prediciendo tablas...")
    tablas_pred = model_tablas([img_tensor])[1][0]
    tablas_boxes, tablas_scores = filter_predictions(
        tablas_pred,
        nms_threshold=config["tablas"]["nms_threshold"],
        score_threshold=config["tablas"]["score_threshold"],
    )
    check_empty_tables(tablas_boxes)

    logging.info("Prediciendo cardinalidades...")
    cardinalidades_pred = predict_tiles(
        img, model=model_cardinalidades, is_yolo=True, transform=transform
    )
    (
        cardinalidades_boxes,
        cardinalidades_scores,
        cardinalidades_labels,
    ) = filter_predictions(
        cardinalidades_pred,
        nms_threshold=config["cardinalidades"]["nms_threshold"],
        score_threshold=config["cardinalidades"]["score_threshold"],
        return_labels=True,
    )

    logging.info("Generando conexiones...")
    conexiones, conexiones_labels = get_pairs(
        boxes_tablas=tablas_boxes,
        boxes_cardinalidades=cardinalidades_boxes,
        labels_cardinalidades=cardinalidades_labels,
        img=img,
        offset_tablas=config["tablas"]["offset"],
        distance_threshold=config["cardinalidades"]["distance_threshold"],
    )

    logging.info("Reconociendo texto...")
    tablas_boxes = tablas_boxes.detach().numpy().astype(int)
    all_tables, tables_names = predict_ocr(
        img=img,
        tablas=tablas_boxes,
        ocr_model=model_ocr,
        scale_percent=config["ocr"]["reescale_percent"],
        lang=config["ocr"]["lang"],
    )

    logging.info("Generando codigo SQL...")
    code = generate_db(
        conexiones, conexiones_labels, all_tables, tables_names, config["ocr"]["lang"]
    )

    return code
