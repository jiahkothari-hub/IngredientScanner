"""Take screenshots of the running Streamlit app (for the report and to check the UI really works).

Usage:  streamlit run src/app/streamlit_app.py &      (in another terminal)
        python -m src.app.screenshot_app [--url http://localhost:8501]

Produces reports/figures/app_text.png (paste-text tab) and app_photo.png (example packet photo scanned).
Needs: pip install playwright (and a Chromium; `playwright install chromium` if none is installed).
"""
import argparse
import os

from playwright.sync_api import sync_playwright

from src.utils.config import project_path


def launch(p):
    for path in ("/opt/pw-browsers/chromium-1194/chrome-linux/chrome", None):
        try:
            return p.chromium.launch(executable_path=path) if path and os.path.exists(path) else p.chromium.launch()
        except Exception:
            continue
    raise RuntimeError("no Chromium found - run `playwright install chromium`")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    args = parser.parse_args()
    out = project_path("reports/figures")
    with sync_playwright() as p:
        browser = launch(p)
        page = browser.new_page(viewport={"width": 1400, "height": 2600})   # Streamlit scrolls internally
        page.goto(args.url, wait_until="networkidle")
        page.get_by_text("Ingredient Scanner").first.wait_for(timeout=120_000)

        page.get_by_role("tab", name="⌨️ Paste text").click()
        page.get_by_role("button", name="Analyse text").click()
        page.get_by_text("What we found").wait_for(timeout=180_000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(out / "app_text.png"), full_page=True)

        page.get_by_role("tab", name="📦 Example packets").click()
        page.get_by_role("button", name="Scan this packet").click()
        page.get_by_text("Details: OCR").first.wait_for(timeout=600_000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(out / "app_photo.png"), full_page=True)
        browser.close()
    print("Saved reports/figures/app_text.png and app_photo.png")


if __name__ == "__main__":
    main()
