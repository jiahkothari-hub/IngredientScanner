# Ingredient Scanner - container for the Streamlit app (CPU only).
#   docker build -t ingredient-scanner .
#   docker run -p 8501:8501 ingredient-scanner        ->  http://localhost:8501
FROM python:3.11-slim

# libgl/libglib are needed by OpenCV (used by EasyOCR)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-app.txt

# code, configuration, knowledge base, lexicons, fine-tuned model and OCR models (all in the repository)
COPY src/ src/
COPY configs/ configs/
COPY .streamlit/ .streamlit/
COPY data/knowledge_base/ data/knowledge_base/
COPY data/processed/additives_reference.csv data/processed/function_classes.csv data/processed/ingredient_vocabulary.txt data/processed/products.csv data/processed/
COPY data/images/ data/images/
COPY models/ingredient-ner-distilbert/ models/ingredient-ner-distilbert/
COPY models/easyocr/ models/easyocr/

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
