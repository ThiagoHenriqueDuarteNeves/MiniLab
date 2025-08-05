# requirements:
# opencv-python-headless
# pillow
# albumentations
# requests
# jiwer
# tqdm

import os
import cv2
import base64
import time
import sys
import numpy as np
import albumentations as A
import requests
from tqdm import tqdm
from jiwer import wer, cer
from io import BytesIO
from PIL import Image

# ─────────── CONFIGURAÇÃO ───────────
PROJECT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR       = r"C:\Users\Thiago\Downloads\BFL_Database\Cartas Manuscritas"
API_URL           = "http://localhost:1234/v1/chat/completions"
MODEL_NAME        = "qwen/qwen2.5-vl-7b"

# diretórios de saída no projeto
GROUND_TRUTH_DIR  = os.path.join(PROJECT_DIR, "ground_truth")
SENT_IMAGES_ROOT  = os.path.join(PROJECT_DIR, "sent_images")
TRANSCRIPT_ROOT   = os.path.join(PROJECT_DIR, "transcriptions")

# garante existência dos diretórios principais
for d in (GROUND_TRUTH_DIR, SENT_IMAGES_ROOT, TRANSCRIPT_ROOT):
    os.makedirs(d, exist_ok=True)

# arquivo de ground-truth esperado em ground_truth/GroundTruth.txt
GROUND_TRUTH_PATH = os.path.join(GROUND_TRUTH_DIR, "GroundTruth.txt")

# session HTTP para reuso de conexão
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
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

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
    return A.Compose([
        A.Lambda(image=deskew,  p=use_deskew),
        A.Lambda(image=clahe,   p=use_clahe),
        A.Lambda(image=adaptive_binarize, p=use_binar),
        # Redimensionar apenas se for muito grande, mantendo proporção
        A.LongestMaxSize(max_size=2048, p=1.0)  # máximo 2048px no lado maior
    ])

# ─────────── API OCR (LM Studio) ───────────
def lmstudio_ocr(img_bgr, api_url, model_name):
    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil    = Image.fromarray(rgb)
    buffer = BytesIO()
    # Usar PNG sem compressão para manter qualidade máxima
    pil.save(buffer, format="PNG")
    b64    = base64.b64encode(buffer.getvalue()).decode("utf-8")

    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extraia exatamente o texto desta imagem. Responda apenas com o texto extraído."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }],
        "max_tokens": 1000,
        "temperature": 0.1
    }

    try:
        resp = session.post(api_url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ Erro na API: {e}")
        return "ERRO_API"

# ─────────── AVALIAÇÃO ───────────
def evaluate(image_paths, pipeline, api_url, model_name, ground_truth_path, sent_dir, trans_dir):
    os.makedirs(sent_dir, exist_ok=True)
    os.makedirs(trans_dir, exist_ok=True)

    reference = open(ground_truth_path, encoding="utf-8").read().strip()
    wer_list, cer_list = [], []

    for path in tqdm(image_paths, desc=f"Processando ({os.path.basename(sent_dir)})"):
        img = cv2.imread(path)
        if img is None:
            print(f"❌ Erro ao carregar imagem: {path}")
            continue
            
        proc = pipeline(image=img)["image"]

        # salva imagem pré-processada por configuração
        sent_path = os.path.join(sent_dir, os.path.basename(path))
        cv2.imwrite(sent_path, proc)

        # obtém OCR e salva transcrição
        pred = lmstudio_ocr(proc, api_url, model_name)
        if pred == "ERRO_API":
            continue
            
        out_txt = os.path.join(trans_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(pred)

        wer_list.append(wer(reference, pred))
        cer_list.append(cer(reference, pred))
        time.sleep(0.2)

    if not wer_list:
        return float('nan'), float('nan')
    return float(np.mean(wer_list)), float(np.mean(cer_list))

# ─────────── EXECUÇÃO PRINCIPAL ───────────
if __name__ == "__main__":
    np.random.seed(42)

    # verificar se o diretório do dataset existe
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Erro: Diretório {DATASET_DIR} não encontrado!")
        sys.exit(1)
    
    # verificar se o arquivo de ground truth existe
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"❌ Erro: Arquivo de ground truth {GROUND_TRUTH_PATH} não encontrado!")
        print(f"Por favor, coloque o arquivo GroundTruth.txt em: {GROUND_TRUTH_DIR}")
        sys.exit(1)

    # buscar apenas as primeiras 5 imagens para teste rápido
    all_images = [
        os.path.join(DATASET_DIR, fn)
        for fn in os.listdir(DATASET_DIR)
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))
    ]
    
    if not all_images:
        print(f"❌ Erro: Nenhuma imagem encontrada em {DATASET_DIR}")
        sys.exit(1)
    
    # usar apenas as primeiras 5 imagens para teste
    image_paths = sorted(all_images)[:5]
    
    print(f"🖼️  Processando {len(image_paths)} imagens de teste (de {len(all_images)} disponíveis)")
    print(f"📄 Ground truth: {GROUND_TRUTH_PATH}")
    print(f"💾 Resultados serão salvos em: {PROJECT_DIR}")
    print("-" * 70)

    configs = {
        "baseline":           make_pipeline(0,0,0),
        "deskew_clahe":       make_pipeline(1,1,0),
        "deskew_clahe_binar": make_pipeline(1,1,1),
    }

    results = {}
    for name, pipe in configs.items():
        sent_dir = os.path.join(SENT_IMAGES_ROOT, name)
        trans_dir = os.path.join(TRANSCRIPT_ROOT, name)
        print(f"\n📊 Executando configuração: {name.upper()}")
        results[name] = evaluate(
            image_paths, pipe, API_URL, MODEL_NAME,
            GROUND_TRUTH_PATH, sent_dir, trans_dir
        )

    # Exibir resultados detalhados
    print("\n" + "="*70)
    print("                    RESULTADOS FINAIS")
    print("="*70)
    
    w0, c0 = results["baseline"]
    print(f"📋 BASELINE:                 WER={w0:.4f}  CER={c0:.4f}")
    print("-" * 70)
    
    for name in ("deskew_clahe", "deskew_clahe_binar"):
        w1, c1 = results[name]
        delta_wer = w1 - w0
        delta_cer = c1 - c0
        
        # Indicadores de melhoria/piora
        wer_indicator = "📈" if delta_wer > 0 else "📉" if delta_wer < 0 else "➡️"
        cer_indicator = "📈" if delta_cer > 0 else "📉" if delta_cer < 0 else "➡️"
        
        config_name = name.replace("_", "+").upper()
        print(f"🔧 {config_name:<15}: WER={w1:.4f}  CER={c1:.4f}")
        print(f"   {wer_indicator} ΔWER={(delta_wer):+.4f}  {cer_indicator} ΔCER={(delta_cer):+.4f}")
        print("-" * 70)
    
    print("\n📁 Arquivos salvos em:")
    print(f"   • Imagens processadas: {SENT_IMAGES_ROOT}")
    print(f"   • Transcrições: {TRANSCRIPT_ROOT}")
    print(f"   • Ground truth: {GROUND_TRUTH_PATH}")
    
    print("\n✅ Processamento concluído com sucesso!")
    print(f"\n💡 Para processar todas as {len(all_images)} imagens, altere a linha:")
    print(f"    image_paths = sorted(all_images)[:5]")
    print(f"    para:")
    print(f"    image_paths = sorted(all_images)")
    
    sys.exit(0)
