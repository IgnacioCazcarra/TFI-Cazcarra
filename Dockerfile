FROM python:3.8

WORKDIR /TFI-Cazcarra

COPY ./api /TFI-Cazcarra/api
COPY ./src /TFI-Cazcarra/src
COPY ./models /TFI-Cazcarra/models
COPY ./setup.py /TFI-Cazcarra/setup.py
COPY ./requirements.txt /TFI-Cazcarra/requirements.txt
COPY ./inference_params.yaml /TFI-Cazcarra/inference_params.yaml

RUN apt install dpkg
RUN apt install wget
RUN apt-get update && apt-get install -y libgl1

RUN pip install --upgrade pip
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r /TFI-Cazcarra/requirements.txt

RUN wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
RUN dpkg -i libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb

RUN chmod +x /TFI-Cazcarra/models/download_models.sh
RUN /TFI-Cazcarra/models/download_models.sh --output_folder /TFI-Cazcarra/models/
RUN python /TFI-Cazcarra/setup.py

EXPOSE 8080
CMD ["uvicorn", "api.app:app", "--port", "8080", "--host", "0.0.0.0"]
