import argparse
from src.inference.inference_utils import prediction_wrapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'Predict')
    parser.add_argument('--img_path', help="Ubicación de la imagen a predecir.") 
    parser.add_argument('--path_to_save', help="Nombre y ubicación del archivo SQL a generar.", 
                        default="./untitled_prediction.sql") 
    args = parser.parse_args()

    prediction_wrapper(img_path=args.img_path, path_to_save=args.path_to_save)