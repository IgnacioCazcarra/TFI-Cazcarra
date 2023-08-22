#!/bin/bash

programname=$0

usage () {
    echo ""
    echo "Descarga los modelos necesarios para ejecutar el programa."
    echo ""
    echo "usage: $programname --output_folder string --all"
    echo ""
    echo "  --output_folder string  Dirección donde guardar los modelos. Default: directorio donde se está parado"
    echo "                          (example: sh download_models.sh --output_folder ./carpeta/)"
    echo ""
}

while [ $# -gt 0 ]; do
    if [[ $1 == "--help" ]]; then
        usage
        exit 0
    elif [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

gdrive_folder="https://drive.google.com/drive/u/0/folders/1Sn0GfpVJukFNLkc0cKZAEUPlV2DQlR9J";
cardinalidades="1e-4_Nz-3YT2FPO267YOvUKTcobOggIlp"
tablas="1U8TWr5a2zBdmu7nQxSBN-bdbXVaW1eOF"

if [[ -z $output_folder ]]; then
    output_folder="$PWD/"
    echo "Not output_folder supplied. Saving to $output_folder"
fi

gdown $tablas --output $output_folder
gdown $cardinalidades --output $output_folder