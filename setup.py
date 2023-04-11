import logging
import spacy
import subprocess
from paddleocr import PaddleOCR

logging.basicConfig(level = logging.INFO)


def _validate_spacy_downloads():
    '''
    Install required language packages for spacy. 
    '''
    if not spacy.util.is_package("es_core_news_sm"):
        subprocess.run("python -m spacy download es_core_news_sm", shell=True)
    if not spacy.util.is_package("en_core_web_sm"):
        subprocess.run("python -m spacy download en_core_web_sm", shell=True)


def _validate_ocr_download():
    '''
    Download the OCR version so it doesn't download on the middle of a prediction.
    '''
    ocr = PaddleOCR(ocr_version='PP-OCRv3', use_angle_cls=False, show_log=False, det_db_score_mode="slow", lang="en")


def setup():
    logging.info("Initializing setup...")
    _validate_spacy_downloads()
    _validate_ocr_download()
    logging.info("Setup finished!")


if __name__ == "__main__":
    setup()
    

