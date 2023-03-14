import argparse
from src.inference.inference_utils import prediction_wrapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'Predict')
    parser.add_argument('--img_path', help="Ubicación de la imagen a predecir.") 
    parser.add_argument('--path_to_save', help="Nombre y ubicación del archivo SQL a generar.", 
                        default="./untitled_prediction.sql")
    parser.add_argument('--yaml_path', help="Ubicación del archivo de configuración YAML. Si no se movió de lugar, no hace falta especificarlo.", 
                        default="./inference_params.yaml")
    parser.add_argument('--plot', help="Flag para indicar si guardar un plot de los objetos predecidos o no.", 
                        action='store_true')   
    parser.set_defaults(plot=False)
    args = parser.parse_args()

    prediction_wrapper(img_path=args.img_path, path_to_save=args.path_to_save,
                       yaml_path=args.yaml_path, plot=args.plot)