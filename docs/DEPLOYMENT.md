# Running and deploying the Ingredient Scanner app

The app needs about **2 GB of RAM**: PyTorch, the DistilBERT model (~265 MB in memory) and the
RapidOCR PP-OCRv6 models (~30 MB, shipped inside the `rapidocr` pip package). It runs on CPU only;
reading one photo takes roughly 2–10 seconds.

## 1. Local (fastest way to demo)

```bash
cd IngredientScanner
pip install -r requirements.txt          # or requirements-app.txt for the app only
streamlit run src/app/streamlit_app.py   # opens http://localhost:8501
```

On a phone in the same Wi-Fi, open `http://<your-laptop-ip>:8501`. The **Camera** tab then uses the
phone camera. Browsers only allow the camera on `localhost` or HTTPS, so for phone use a deployed
HTTPS URL (below) is easier.

## 2. Docker

```bash
docker build -t ingredient-scanner .
docker run -p 8501:8501 ingredient-scanner
```

The image contains the code, knowledge base, fine-tuned model and OCR models, so nothing is
downloaded at runtime.

## 3. Hugging Face Spaces (recommended free hosting, HTTPS, 16 GB RAM)

1. Create a Space at https://huggingface.co/new-space → SDK **Docker** → hardware *CPU basic (free)*.
2. Push the contents of this repository to the Space repository. Large files (`*.safetensors`,
   `*.pth`) must be tracked with Git LFS:
   ```bash
   git lfs install
   git lfs track "*.safetensors" "*.pth"
   git add .gitattributes . && git commit -m "Ingredient Scanner app" && git push
   ```
3. In the Space's `README.md` header set `app_port: 8501`. The Space builds the Dockerfile and serves
   the app at `https://<user>-<space>.hf.space`. That address works with the phone camera.

## 4. Streamlit Community Cloud

Possible, but the free tier has ~1 GB RAM, which is tight for PyTorch + DistilBERT + OCR. If you try:
select the repository, main file `src/app/streamlit_app.py`, and the requirements file
`requirements-app.txt`. If memory runs out, choose **Dictionary rules only** in the sidebar
(no Transformer), or use Hugging Face Spaces.

## Notes

* Model files are stored as float16 shards under 45 MB, so they fit in a normal GitHub repository.
* Example photos are from Open Food Facts (CC BY-SA); keep the attribution when you publish the app.
* The app shows a disclaimer: it identifies and explains ingredients and makes no health claims.
