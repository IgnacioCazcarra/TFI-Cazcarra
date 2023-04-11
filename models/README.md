# Download models from GDrive

To download the trained models, execute the 'download_model' bash file.

Example usage:
```sh
# This would download the models in the current directory
bash download_models.sh --output_folder ./
```

For more information about the parameters, use:
```sh
bash download_models.sh --help
```

If you run into some issues with this part, you can opt for manually downloading the models from [here](https://drive.google.com/drive/u/0/folders/1Sn0GfpVJukFNLkc0cKZAEUPlV2DQlR9J) and placing them in this folder.
Required models are ```retinanet_tablas.pt``` and ```yolo_cardinalidades.pt```.