# TFI-Cazcarra
Converting ER Diagrams to .sql scripts using neural networks.

For predicting an image, the steps are the following:

1) Create an environment and install the packages in requirements.txt
2) Download the models. See models/ for more info.
3) Execute predict.py

Example usage:

```
python predict.py --img_path ./imagenes/imagen.png --path_to_save ./resultados/resultados_imagen.sql
```