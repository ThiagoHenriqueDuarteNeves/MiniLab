# requirements:
# opencv-python-headless
# pillow
# albumentations
# requests
# jiwer
# tqdm

import os
import cv2
import glob
import base64
import json
import time
import numpy as np
import albumentations as A
import requests
from tqdm import tqdm
from jiwer import wer, cer
from io import BytesIO
from PIL import Image

# ─────────── CONFIGURAÇÃO ───────────
DATASET_DIR       = r"C:\Users\Thiago\Downloads\BFL_Database\Cartas Manuscritas"
GROUND_TRUTH_PATH = r"C:\Users\Thiago\Documents\EnvioOCR\data\examples\GroundTruth.txt"
API_URL           = "http://localhost:1234/v1/chat/completions"
MODEL_NAME        = "qwen/qwen2.5-vl-7b"

# reutiliza conexão HTTP para evitar handshakes repetidos
session = requests.Session()
session.headers.update({"Content-Type": "application/json"})

# ─────────── BLOCO DE PRÉ-PROCESSAMENTO ───────────
def deskew(img, **kwargs):
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 255))
    if coords.size == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def clahe(img, **kwargs):
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((l2,a,b)), cv2.COLOR_LAB2BGR)

def adaptive_binarize(img, **kwargs):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = cv2.adaptiveThreshold(g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

def make_pipeline(use_deskew=1, use_clahe=1, use_binar=0):
    # resize menor (512px) para reduzir payload
    return A.Compose([
        A.Lambda(image=deskew,  p=use_deskew),
        A.Lambda(image=clahe,   p=use_clahe),
        A.Lambda(image=adaptive_binarize, p=use_binar),
        A.Resize(height=512, width=512, p=1.0)
    ])

# ─────────── API OCR (LM Studio) ───────────
def lmstudio_ocr(img_bgr, api_url, model_name):
    # converte para JPEG em memória (payload menor que PNG)
    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil    = Image.fromarray(rgb)
    buffer = BytesIO()
    pil.save(buffer, format="JPEG", quality=75)
    b64    = base64.b64encode(buffer.getvalue()).decode("utf-8")

    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Extraia exatamente o texto desta imagem. Responda apenas com o texto extraído."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }],
        "max_tokens": 1000,
        "temperature": 0.1
    }

    resp = session.post(api_url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

# ─────────── AVALIAÇÃO ───────────
def evaluate(image_paths, pipeline, api_url, model_name, ground_truth_path):
    reference = open(ground_truth_path, encoding="utf-8").read().strip()
    wer_list, cer_list = [], []

    for path in tqdm(image_paths, desc="Processando imagens"):
        img = cv2.imread(path)
        t0  = time.time()
        proc = pipeline(image=img)["image"]
        t1  = time.time()
        pred = lmstudio_ocr(proc, api_url, model_name)
        t2  = time.time()

        wer_list.append(wer(reference, pred))
        cer_list.append(cer(reference, pred))

        # log de latências (opcional, descomente se quiser ver)
        # print(f"Pré-proc: {(t1-t0):.2f}s; OCR: {(t2-t1):.2f}s")

        time.sleep(0.2)  # pausa leve para não sobrecarregar

    return float(np.mean(wer_list)), float(np.mean(cer_list))

# ─────────── EXECUÇÃO PRINCIPAL ───────────
if __name__ == "__main__":
    np.random.seed(42)

    # especifique apenas as imagens que deseja testar
    image_paths = [
        os.path.join(DATASET_DIR, fn)
        for fn in os.listdir(DATASET_DIR)
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))
    ]

    configs = {
        "Baseline":           make_pipeline(0,0,0),
        "Deskew+CLAHE":       make_pipeline(1,1,0),
        "Deskew+CLAHE+Binar": make_pipeline(1,1,1),
    }

    results = {}
    for name, pipe in configs.items():
        results[name] = evaluate(image_paths, pipe, API_URL, MODEL_NAME, GROUND_TRUTH_PATH)

    # exibe comparativo
    w0, c0 = results["Baseline"]
    print(f"Baseline       WER={w0:.4f}  CER={c0:.4f}")
    for name in ("Deskew+CLAHE", "Deskew+CLAHE+Binar"):
        w1, c1 = results[name]
        print(f"{name:<17} WER={w1:.4f}  CER={c1:.4f}  ΔWER={(w1-w0):+.4f}  ΔCER={(c1-c0):+.4f}")

    import sys
    sys.exit(0)
