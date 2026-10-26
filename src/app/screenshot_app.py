"""Take screenshots of the running Streamlit app (for the report and to check the UI really works).

Usage:  streamlit run src/app/streamlit_app.py &      (in another terminal)
        python -m src.app.screenshot_app [--url http://localhost:8501]

Produces reports/figures/app_*.png: landing page, a sample product (ingredients, daily guide, additives)
and an uploaded photo (result and 'how it was read').
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


def shot(page, path):
    page.wait_for_timeout(2000)
    page.screenshot(path=str(path), full_page=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--photo", default="data/images/packets/8901499009289.jpg", help="photo to upload")
    args = parser.parse_args()
    out = project_path("reports/figures")
    with sync_playwright() as p:
        browser = launch(p)
        page = browser.new_page(viewport={"width": 1400, "height": 1700})   # Streamlit scrolls internally
        page.goto(args.url, wait_until="networkidle")
        page.get_by_text("Ingredient Scanner").first.wait_for(timeout=120_000)
        shot(page, out / "app_landing.png")

        page.get_by_role("button", name="Nutella").click()                  # sample product (typed list)
        page.get_by_text("The label, classified").first.wait_for(timeout=180_000)
        shot(page, out / "app_text.png")
        page.get_by_role("tab", name="⚖️ Is this amount OK?").click()
        shot(page, out / "app_guide.png")
        page.get_by_role("tab", name="⚗️ Additives").click()
        shot(page, out / "app_additives.png")

        page.get_by_role("button", name="↺ New scan").click()
        page.get_by_role("tab", name="🖼️ Upload photo").click()
        page.locator("input[type=file]").first.set_input_files(str(project_path(args.photo)))
        page.get_by_text("The label, classified").first.wait_for(timeout=600_000)
        shot(page, out / "app_photo.png")
        page.get_by_role("tab", name="🔍 How it was read").click()
        shot(page, out / "app_reading.png")
        browser.close()
    print("Saved reports/figures/app_landing/text/guide/additives/photo/reading.png")


if __name__ == "__main__":
    main()
