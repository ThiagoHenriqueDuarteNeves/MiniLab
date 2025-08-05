# 📋 Guia de Uso do Script OCR

## 🚀 Como Executar

### 1. **Modo Interativo (Recomendado)**
```bash
python app.py
```
O script apresentará um menu para escolher quantas imagens processar:
- 5 imagens (teste rápido)
- 10, 25, 50 imagens
- Todas as imagens
- Número personalizado

### 2. **Linha de Comando - Número Específico**
```bash
# Processar apenas 10 imagens
python app.py -n 10

# Processar 25 imagens
python app.py --num-images 25
```

### 3. **Linha de Comando - Todas as Imagens**
```bash
python app.py --all
```

### 4. **Ajuda**
```bash
python app.py --help
```

## 📊 O que o Script Faz

1. **Carrega as imagens** do diretório configurado
2. **Aplica três configurações** de pré-processamento:
   - `baseline`: sem pré-processamento
   - `deskew_clahe`: correção de inclinação + melhoria de contraste
   - `deskew_clahe_binar`: correção + contraste + binarização

3. **Envia para OCR** via API LM Studio
4. **Calcula métricas** WER e CER comparando com ground truth
5. **Salva resultados**:
   - Imagens processadas em `sent_images/`
   - Transcrições em `transcriptions/`
   - Relatório no terminal

## 🎯 Recomendações

- **Primeiro teste**: Use 5 imagens para verificar se tudo funciona
- **Desenvolvimento**: Use 10-25 imagens para ajustes
- **Experimento completo**: Use todas as imagens para resultados finais

## 📁 Estrutura de Saída

```
MiniLab/
├── sent_images/
│   ├── baseline/          # Imagens sem pré-processamento
│   ├── deskew_clahe/      # Imagens com deskew + CLAHE
│   └── deskew_clahe_binar/# Imagens com todos os processos
├── transcriptions/
│   ├── baseline/          # Transcrições do baseline
│   ├── deskew_clahe/      # Transcrições com pré-processamento
│   └── deskew_clahe_binar/# Transcrições completas
└── ground_truth/
    └── GroundTruth.txt    # Texto de referência
```

## ⚙️ Configuração

Edite as seguintes variáveis no início do `app.py`:

```python
DATASET_DIR = r"C:\seu\caminho\para\imagens"
API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "seu-modelo"
```

## 🔧 Troubleshooting

- **Erro de imagens não encontradas**: Verifique o `DATASET_DIR`
- **Erro de ground truth**: Coloque o arquivo em `ground_truth/GroundTruth.txt`
- **Erro de API**: Verifique se o LM Studio está rodando
- **Imagens corrompidas**: O script continuará com as próximas imagens
