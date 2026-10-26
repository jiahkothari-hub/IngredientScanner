ABOUT THIS ZIP (Ingredient Scanner v2.1 (redesigned UI), app edition)

This zip has all code, notebooks, reports, the knowledge base, the 60 example photos and the 16 new test
photos. To stay under the 30 MB download limit it does NOT include the large model files:
  models/ingredient-ner-distilbert/  (fine-tuned DistilBERT, 128 MB)
  models/crf/, models/easyocr/, data/splits/ and three large intermediate data files

The app works without them: the sidebar classifier "Hybrid (recommended)" then runs as
dictionary rules + fuzzy knowledge-base matching (no DistilBERT). To add DistilBERT, copy the folder
models/ingredient-ner-distilbert/ from the full project:
  https://github.com/jiahkothari-hub/IngredientScanner  (Code -> Download ZIP, or git clone)

Run (Windows):
  cd IngredientScanner
  python -m venv .venv
  .venv\Scripts\activate
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements-app.txt
  streamlit run src/app/streamlit_app.py
(macOS / Linux: source .venv/bin/activate instead of the activate line)
