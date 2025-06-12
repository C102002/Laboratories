# Laboratorio 7: Reconocimiento de voz

Este repositorio contiene el código para transcribir archivos de audio en español y aplicar técnicas de preprocesamiento de PLN (normalización, tokenización y eliminación de stop-words).

---

## Requisitos

- Python 3.10  
- [FFmpeg](https://ffmpeg.org/) (para convertir MP3 a WAV con pydub)  
- Git (opcional, para clonar este repositorio)  

---

## 1. Instalación y configuración

1. Crear el ambiente virtual con python en la version 3.10.11
2. Descargar los requerimientos
```bash
# Ejecutar el siguiente comando
pip install -r requirements.txt
```
3. Actualizar los requerimientos
```bash
pip freeze > requirements.txt
```
4. Ejecutar el laboratorio
```bash
# En la ruta del laboratorio_7
# (.venv) PS C:\..\laboratorio_7> python .\laboratorio_7.py
python .\laboratorio_7.py

# Despues de correr la transcripcion
python .\laboratorio_7_parte_2.py

```
5. Extra
Intalacion de dependencias manual
```bash
# En windows
pip install huggingsound --no-deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets transformers jiwer librosa spacy pydub pyspellchecker
python -m spacy download es_core_news_sm
pip install spacy
python -m spacy download es_core_news_sm
```

# 🎤 Extra: Modelo de Speech Recognition para Análisis de Video

Este proyecto implementa un modelo de **Speech Recognition** para analizar el contenido de un video de clase. Se basa en la conversión de audio a texto, permitiendo realizar preprocesamiento y extracción de información relevante.

## 🔗 Enlace al Colab  
Puedes ejecutar el modelo directamente en Google Colab:  
➡️ [Acceder al Notebook](https://colab.research.google.com/drive/1guc0_eZazL3Ou3ZBPvGZFawmtidSI0SB?authuser=2#scrollTo=MAtz2e03OxXx)