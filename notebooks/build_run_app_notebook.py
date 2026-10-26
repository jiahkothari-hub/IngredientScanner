"""Build notebooks/run_streamlit_app.ipynb (run from the project root):  python notebooks/build_run_app_notebook.py"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Run the Ingredient Scanner app (Streamlit) from a notebook

Run the cells from top to bottom. Works in:

| Where | What to do first |
|---|---|
| **Jupyter / VS Code / Anaconda on your laptop** | Open this notebook from the project's `notebooks/` folder. Nothing else needed. |
| **Google Colab** | *Runtime → Run all*. Cell 2 asks you to upload the project zip (`IngredientScanner.zip`) (or clones the GitHub repository, see the comments in that cell). |

What happens:
1. find the project folder;
2. install the packages that are missing;
3. start Streamlit in the background;
4. show the app inside the notebook (plus a link to open it in its own tab, which you need for the camera);
5. stop the app when you are done.

If the app cannot be shown, the last cell runs the same pipeline directly in the notebook, without Streamlit.
""")

md("## 1. Settings")
code("""
import os, sys, time, subprocess, zipfile, urllib.request
from pathlib import Path

PORT = 8501                        # change if this port is busy
IN_COLAB = "google.colab" in sys.modules
ZIP_NAME = "IngredientScanner.zip"
print("Running in Google Colab" if IN_COLAB else "Running in a local Jupyter")
""")

md("## 2. Find (or get) the project folder")
code("""
def find_project():
    here = Path.cwd()
    for candidate in [here, here.parent, Path("/content/IngredientScanner"), Path("/content/ingredient-scanner"),
                      Path("/content/SkillConnect/ingredient-scanner")]:
        if (candidate / "src" / "app" / "streamlit_app.py").exists():
            return candidate.resolve()
    return None

ROOT = find_project()

if ROOT is None and IN_COLAB:
    # Option A (default): upload the zip you received
    zip_path = Path("/content") / ZIP_NAME
    if not zip_path.exists():
        from google.colab import files
        print(f"Please choose {ZIP_NAME} in the dialog ...")
        uploaded = files.upload()
        zip_path = Path("/content") / next(iter(uploaded))
    with zipfile.ZipFile(zip_path) as z:
        z.extractall("/content")
    ROOT = find_project()

    # Option B: clone the GitHub repository instead of uploading the zip
    # (private repository: use https://<YOUR_TOKEN>@github.com/jiahkothari-hub/IngredientScanner.git)
    # !git clone https://github.com/jiahkothari-hub/IngredientScanner.git /content/IngredientScanner
    # ROOT = find_project()

if ROOT is None:
    raise SystemExit("Project not found. Open this notebook from the project's notebooks/ folder, "
                     "or (Colab) upload IngredientScanner.zip.")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("Project folder:", ROOT)
print("Fine-tuned model present:", (ROOT / "models/ingredient-ner-distilbert/config.json").exists())
print("OCR dictionary present:  ", (ROOT / "data/processed/ocr_dictionary.txt").exists())
""")

md("""
## 3. Install missing packages

Only packages that are not installed yet are installed (Colab already has PyTorch).
On a laptop **without** PyTorch, installing the CPU version first is much smaller:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`
""")
code("""
import importlib.util
needed = {"streamlit": "streamlit", "rapidocr": "rapidocr", "onnxruntime": "onnxruntime", "symspellpy": "symspellpy",
          "sklearn": "scikit-learn", "transformers": "transformers", "torch": "torch",
          "torchcrf": "pytorch-crf", "rapidfuzz": "rapidfuzz", "cv2": "opencv-python-headless",
          "pandas": "pandas", "yaml": "PyYAML", "requests": "requests", "PIL": "Pillow"}
missing = [pkg for module, pkg in needed.items() if importlib.util.find_spec(module) is None]
if missing:
    print("Installing:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("All packages are installed.")
""")

md("## 4. Start the Streamlit app in the background")
code("""
def app_is_up(port=PORT):
    try:
        return urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2).read() == b"ok"
    except Exception:
        return False

if app_is_up():
    print(f"An app is already running on port {PORT}.")
else:
    command = [sys.executable, "-m", "streamlit", "run", "src/app/streamlit_app.py",
               "--server.port", str(PORT), "--server.headless", "true"]
    if IN_COLAB:   # Colab serves the app through a proxy: these settings allow photo uploads through it
        command += ["--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
    log_file = open(ROOT / "streamlit.log", "w")
    app_process = subprocess.Popen(command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    for _ in range(90):                     # wait up to ~3 minutes (first start loads the models)
        if app_is_up():
            break
        if app_process.poll() is not None:
            break
        time.sleep(2)
    print("App is running." if app_is_up() else "App did not start - see the log in cell 6.")
""")

md("""
## 5. Open the app

The first scan of a photo takes a little longer because the models are loaded; then about 3-15 s per photo on a CPU.
**Camera tab:** browsers only allow the camera on `localhost` or HTTPS pages, so use the *open in a new tab* link.
""")
code("""
from IPython.display import IFrame, Markdown, display

if IN_COLAB:
    from google.colab import output
    output.serve_kernel_port_as_window(PORT)            # link: open the app in its own browser tab (camera works there)
    output.serve_kernel_port_as_iframe(PORT, height=1000)
else:
    display(Markdown(f"**Open the app in its own tab:** [http://localhost:{PORT}](http://localhost:{PORT})"))
    display(IFrame(f"http://localhost:{PORT}", width="100%", height=1000))
""")

md("## 6. Troubleshooting: last lines of the app log")
code("""
log = (ROOT / "streamlit.log")
print(log.read_text()[-3000:] if log.exists() else "no log yet")
""")

md("## 7. Stop the app (run this when you are finished)")
code("""
try:
    app_process.terminate()
    app_process.wait(timeout=20)
    print("App stopped.")
except NameError:
    print("The app was not started from this notebook session.")
""")

md("""
## 8. Optional: use the pipeline directly in the notebook (no Streamlit)

Same pipeline as the app: photo → OCR (RapidOCR + spelling correction) → ingredients section → hybrid NER
(rules + DistilBERT + fuzzy KB) → knowledge base → summary, allergens, nutrition and the daily intake guide.
Change `PHOTO` to your own image path, or set `TEXT` to analyse typed text.
""")
code("""
import pandas as pd
from IPython.display import Image as ShowImage
from src.app.scanner import IngredientScanner

PHOTO = "data/images/packets/8901063162518.jpg"   # any photo of an ingredient list
TEXT = None                                         # e.g. "Sugar, glucose syrup, acidity regulator (INS 330)"

scanner = IngredientScanner("auto")                 # hybrid: rules + fine-tuned DistilBERT + fuzzy matching
if TEXT:
    result = scanner.scan_text(TEXT)
else:
    display(ShowImage(PHOTO, width=350))
    result = scanner.scan_image(PHOTO)
    print("Ingredient text read from the photo:\\n", result["text"], "\\n")

print("Classifier:", result["ner_model"])
display(pd.DataFrame(result["entities"])[["text", "label", "canonical_name", "function", "description"]])
for category, items in result["summary"]["groups"].items():
    if items:
        print(f"{category:13}", ", ".join(items))
print("\\nHidden names:", result["summary"]["hidden"])
print("Allergens:", result["allergens"])

# daily intake guide for a 30 g portion (nutrition from the photo, else an estimate from a similar product)
from src.app.intake import PROFILES, assess_portion, headline
from src.app.nutrition import estimate_from_similar_product
per100 = result["nutrition"]["per_100g"] or estimate_from_similar_product(result["input_text"]).get("per_100g", {})
if per100:
    rows = assess_portion(per100, 30, PROFILES["Adult (2000 kcal)"])
    print(headline(rows, 30)[0])
    display(pd.DataFrame(rows)[["nutrient", "in_portion", "unit", "daily_value", "share", "traffic_light", "message"]])
print("\\n" + result["disclaimer"])
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb["metadata"]["colab"] = {"provenance": []}
nbf.write(nb, "notebooks/run_streamlit_app.ipynb")
print("written", len(cells), "cells")
