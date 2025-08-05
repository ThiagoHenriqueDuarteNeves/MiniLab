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
import argparse
import shutil
import csv
import datetime
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
RESULTS_DIR       = os.path.join(PROJECT_DIR, "results")

# garante existência dos diretórios principais
for d in (GROUND_TRUTH_DIR, SENT_IMAGES_ROOT, TRANSCRIPT_ROOT, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

# arquivo de ground-truth esperado em ground_truth/GroundTruth.txt
GROUND_TRUTH_PATH = os.path.join(GROUND_TRUTH_DIR, "GroundTruth.txt")

# session HTTP para reuso de conexão
session = requests.Session()
session.headers.update({"Content-Type": "application/json"})

# ─────────── LIMPEZA DE DIRETÓRIOS ───────────
def clean_output_directories():
    """
    Limpa os diretórios de saída antes de cada execução
    """
    dirs_to_clean = [
        os.path.join(SENT_IMAGES_ROOT, "no_treatment"),
        os.path.join(SENT_IMAGES_ROOT, "text_only"),
        os.path.join(TRANSCRIPT_ROOT, "no_treatment"),
        os.path.join(TRANSCRIPT_ROOT, "text_only")
    ]
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"🧹 Diretório limpo: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Diretório criado: {dir_path}")

# ─────────── SALVAMENTO DE RESULTADOS ───────────
def save_individual_results(results_no, results_text, timestamp, num_images):
    """
    Salva os resultados individuais em arquivo CSV
    """
    filename = f"results_{timestamp}_{num_images}imgs.csv"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['imagem', 'sem_trat_wer', 'sem_trat_cer', 'com_trat_wer', 'com_trat_cer', 
                     'wer_diff_percent', 'cer_diff_percent']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for res_no, res_text in zip(results_no, results_text):
            wer_diff = ((res_text['wer'] - res_no['wer']) / res_no['wer']) * 100 if res_no['wer'] != 0 else 0
            cer_diff = ((res_text['cer'] - res_no['cer']) / res_no['cer']) * 100 if res_no['cer'] != 0 else 0
            
            writer.writerow({
                'imagem': res_no['image'],
                'sem_trat_wer': f"{res_no['wer']:.4f}",
                'sem_trat_cer': f"{res_no['cer']:.4f}",
                'com_trat_wer': f"{res_text['wer']:.4f}",
                'com_trat_cer': f"{res_text['cer']:.4f}",
                'wer_diff_percent': f"{wer_diff:+.2f}%",
                'cer_diff_percent': f"{cer_diff:+.2f}%"
            })
    
    print(f"💾 Resultados individuais salvos em: {filepath}")
    return filepath

def save_summary_results(wer_no, cer_no, wer_text, cer_text, timestamp, num_images):
    """
    Salva um resumo dos resultados em arquivo CSV
    """
    filename = f"summary_{timestamp}_{num_images}imgs.csv"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    wer_diff = ((wer_text - wer_no) / wer_no) * 100 if not np.isnan(wer_no) and wer_no != 0 else float('nan')
    cer_diff = ((cer_text - cer_no) / cer_no) * 100 if not np.isnan(cer_no) and cer_no != 0 else float('nan')
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'num_images', 'sem_trat_wer', 'sem_trat_cer', 
                     'com_trat_wer', 'com_trat_cer', 'wer_diff_percent', 'cer_diff_percent']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerow({
            'timestamp': timestamp,
            'num_images': num_images,
            'sem_trat_wer': f"{wer_no:.4f}" if not np.isnan(wer_no) else "NaN",
            'sem_trat_cer': f"{cer_no:.4f}" if not np.isnan(cer_no) else "NaN",
            'com_trat_wer': f"{wer_text:.4f}" if not np.isnan(wer_text) else "NaN",
            'com_trat_cer': f"{cer_text:.4f}" if not np.isnan(cer_text) else "NaN",
            'wer_diff_percent': f"{wer_diff:+.2f}%" if not np.isnan(wer_diff) else "NaN",
            'cer_diff_percent': f"{cer_diff:+.2f}%" if not np.isnan(cer_diff) else "NaN"
        })
    
    print(f"📊 Resumo dos resultados salvo em: {filepath}")
    return filepath

# ─────────── BLOCO DE PRÉ-PROCESSAMENTO ───────────
def enhance_text_only(img, **kwargs):
    """
    Escurece apenas o texto, mantendo o fundo claro, via limiarização adaptativa.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=10
    )
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# pipeline que aplica apenas o tratamento de escurecimento de texto
def make_pipeline_text_only():
    return A.Compose([
        A.Lambda(image=enhance_text_only, p=1.0),
        A.LongestMaxSize(max_size=2048, p=1.0)
    ])

# pipeline sem tratamento (apenas redimensionamento)
def make_pipeline_no_treatment():
    return A.Compose([
        A.LongestMaxSize(max_size=2048, p=1.0)
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
        resp = session.post(api_url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ Erro na API: {e}")
        return "ERRO_API"

# ─────────── AVALIAÇÃO ───────────
def evaluate(image_paths, pipeline, api_url, model_name, ground_truth_path, sent_dir, trans_dir, pipeline_name):
    os.makedirs(sent_dir, exist_ok=True)
    os.makedirs(trans_dir, exist_ok=True)

    reference = open(ground_truth_path, encoding="utf-8").read().strip()
    wer_list, cer_list = [], []
    individual_results = []

    for i, path in enumerate(tqdm(image_paths, desc=f"Processando ({pipeline_name})")):
        img = cv2.imread(path)
        if img is None:
            print(f"❌ Erro ao carregar imagem: {path}")
            continue
            
        proc = pipeline(image=img)["image"]

        # salva imagem pré-processada
        sent_path = os.path.join(sent_dir, os.path.basename(path))
        success = cv2.imwrite(sent_path, proc)
        if success:
            print(f"✅ Imagem {i+1} salva: {sent_path}")
        else:
            print(f"❌ Erro ao salvar imagem: {sent_path}")

        # obtém OCR e salva transcrição
        pred = lmstudio_ocr(proc, api_url, model_name)
        if pred == "ERRO_API":
            print(f"❌ Erro na API para: {os.path.basename(path)}")
            continue
            
        out_txt = os.path.join(trans_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(pred)
        print(f"📝 Transcrição {i+1} salva: {out_txt}")
        print(f"🔍 Texto extraído: {pred[:100]}{'...' if len(pred) > 100 else ''}")

        # calcular métricas individuais
        wer_score = wer(reference, pred)
        cer_score = cer(reference, pred)
        
        wer_list.append(wer_score)
        cer_list.append(cer_score)
        
        individual_results.append({
            'image': os.path.basename(path),
            'wer': wer_score,
            'cer': cer_score
        })
        
        print(f"📊 Imagem {i+1} - WER: {wer_score:.4f}, CER: {cer_score:.4f}")
        time.sleep(0.2)

    if not wer_list:
        return float('nan'), float('nan'), []
    
    return float(np.mean(wer_list)), float(np.mean(cer_list)), individual_results

# ─────────── EXECUÇÃO PRINCIPAL ───────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processamento OCR com escurecimento de texto apenas')
    parser.add_argument('-n', '--num-images', type=int, default=None,
                        help='Número de imagens a processar (padrão: todas)')
    parser.add_argument('--all', action='store_true',
                        help='Processar todas as imagens (ignora -n)')
    
    args = parser.parse_args()
    np.random.seed(42)
    
    # criar timestamp para esta execução
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # verificar diretórios e arquivo
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Erro: Diretório {DATASET_DIR} não encontrado!")
        sys.exit(1)
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"❌ Erro: Arquivo de ground truth {GROUND_TRUTH_PATH} não encontrado!")
        sys.exit(1)

    # buscar imagens
    all_image_paths = [
        os.path.join(DATASET_DIR, fn)
        for fn in os.listdir(DATASET_DIR)
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))
    ]
    if not all_image_paths:
        print(f"❌ Erro: Nenhuma imagem encontrada em {DATASET_DIR}")
        sys.exit(1)
    all_image_paths = sorted(all_image_paths)

    # selecionar imagens
    if args.all:
        image_paths = all_image_paths
        processing_msg = "todas as"
    elif args.num_images is not None:
        num_to_process = min(args.num_images, len(all_image_paths))
        image_paths = all_image_paths[:num_to_process]
        processing_msg = f"{num_to_process} de {len(all_image_paths)}"
    else:
        image_paths = all_image_paths
        processing_msg = "todas as"

    print(f"\n🖼️  Processando {processing_msg} imagens (de {len(all_image_paths)} disponíveis)")
    print(f"📄 Ground truth: {GROUND_TRUTH_PATH}")
    print(f"💾 Resultados serão salvos em: {PROJECT_DIR}")
    
    # limpar diretórios de saída
    print("\n🧹 Limpando diretórios de resultados anteriores...")
    clean_output_directories()
    print("-" * 70)

    # configurar pipelines
    pipeline_no_treatment = make_pipeline_no_treatment()
    pipeline_text_only = make_pipeline_text_only()

    # avaliar sem tratamento
    print("\n🔸 PROCESSANDO SEM TRATAMENTO...")
    sent_dir_no = os.path.join(SENT_IMAGES_ROOT, "no_treatment")
    trans_dir_no = os.path.join(TRANSCRIPT_ROOT, "no_treatment")
    wer_no, cer_no, results_no = evaluate(
        image_paths, pipeline_no_treatment, API_URL, MODEL_NAME,
        GROUND_TRUTH_PATH, sent_dir_no, trans_dir_no, "Sem tratamento"
    )

    # avaliar com escurecimento de texto
    print("\n🔸 PROCESSANDO COM ESCURECIMENTO DE TEXTO...")
    sent_dir_text = os.path.join(SENT_IMAGES_ROOT, "text_only")
    trans_dir_text = os.path.join(TRANSCRIPT_ROOT, "text_only")
    wer_text, cer_text, results_text = evaluate(
        image_paths, pipeline_text_only, API_URL, MODEL_NAME,
        GROUND_TRUTH_PATH, sent_dir_text, trans_dir_text, "Escurecimento de texto"
    )

    # exibir resultados
    print("\n" + "="*80)
    print("                           RESULTADOS COMPARATIVOS")
    print("="*80)
    
    # resultados médios
    print(f"� SEM TRATAMENTO:          WER={wer_no:.4f}    CER={cer_no:.4f}")
    print(f"📊 ESCURECIMENTO DE TEXTO:  WER={wer_text:.4f}  CER={cer_text:.4f}")
    
    # diferença percentual
    if not np.isnan(wer_no) and not np.isnan(wer_text):
        wer_diff = ((wer_text - wer_no) / wer_no) * 100
        cer_diff = ((cer_text - cer_no) / cer_no) * 100
        print(f"🔄 DIFERENÇA PERCENTUAL:    WER={wer_diff:+.2f}%  CER={cer_diff:+.2f}%")
    
    print("-" * 80)
    
    # resultados individuais
    if results_no and results_text:
        print("📋 RESULTADOS INDIVIDUAIS POR IMAGEM:")
        print(f"{'Imagem':<15} {'Sem Trat WER':<12} {'Sem Trat CER':<12} {'Com Trat WER':<12} {'Com Trat CER':<12}")
        print("-" * 80)
        
        for i, (res_no, res_text) in enumerate(zip(results_no, results_text)):
            print(f"{res_no['image']:<15} {res_no['wer']:<12.4f} {res_no['cer']:<12.4f} {res_text['wer']:<12.4f} {res_text['cer']:<12.4f}")
    
    print("="*80)
    print(f"📁 Imagens sem tratamento salvas em: {sent_dir_no}")
    print(f"📁 Imagens com tratamento salvas em: {sent_dir_text}")
    print(f"📄 Transcrições sem tratamento salvas em: {trans_dir_no}")
    print(f"📄 Transcrições com tratamento salvas em: {trans_dir_text}")
    
    # salvar resultados em CSV
    if results_no and results_text:
        individual_file = save_individual_results(results_no, results_text, timestamp, len(image_paths))
        summary_file = save_summary_results(wer_no, cer_no, wer_text, cer_text, timestamp, len(image_paths))
    
    print("="*80)

    print("✅ Processamento concluído com sucesso!")
    sys.exit(0)