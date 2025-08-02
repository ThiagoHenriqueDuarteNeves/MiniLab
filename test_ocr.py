#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste simples para verificar se a API OCR está funcionando
"""

import cv2
import numpy as np
from app import lmstudio_ocr, make_pipeline

# Configurações
API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "qwen/qwen2.5-vl-7b"

def test_with_sample_image():
    """Cria uma imagem de teste simples com texto"""
    # Criar uma imagem branca com texto preto
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    
    # Adicionar texto usando OpenCV
    cv2.putText(img, "Hello World!", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "Test OCR", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    print("Testando OCR com imagem sintética...")
    
    try:
        # Testar sem pré-processamento
        result = lmstudio_ocr(img, API_URL, MODEL_NAME)
        print(f"Resultado OCR: '{result}'")
        
        # Testar com pipeline de pré-processamento
        pipeline = make_pipeline(1, 1, 0)  # Deskew + CLAHE
        processed = pipeline(image=img)["image"]
        result_processed = lmstudio_ocr(processed, API_URL, MODEL_NAME)
        print(f"Resultado OCR (pré-processado): '{result_processed}'")
        
    except Exception as e:
        print(f"Erro ao testar OCR: {e}")
        print("Verifique se o LM Studio está rodando e se a URL/modelo estão corretos.")

if __name__ == "__main__":
    test_with_sample_image()
