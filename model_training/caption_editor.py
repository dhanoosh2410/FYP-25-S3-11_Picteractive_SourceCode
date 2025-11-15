from pathlib import Path
from typing import Dict, List

import gradio as gr
from PIL import Image

IMAGES_DIR = Path("server/data/Images")
CAPTIONS_FILE = Path("server/data/captions.txt")

captions: Dict[str, List[str]] = {}


def load_captions() -> Dict[str, List[str]]:
    """
    Load captions from `server/data/captions.txt`.

    Supports the same formats as the training script and CLIP captioner:
    - TSV: image.jpg<TAB>caption
    - CSV: image.jpg,caption   (header 'image,caption' is skipped)
    """
    caps: Dict[str, List[str]] = {}
    if not CAPTIONS_FILE.is_file():
        return caps

    with CAPTIONS_FILE.open("r", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)

        if "\t" in first:
            # TSV
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                img, cap = line.split("\t", 1)
                img = img.split("#", 1)[0].strip()
                cap = cap.strip()
                if img and cap:
                    caps.setdefault(img, []).append(cap)
        else:
            # CSV
            import csv

            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                # Skip header if present
                if (
                    row[0].strip().lower() == "image"
                    and row[1].strip().lower() == "caption"
                ):
                    continue
                img = row[0].split("#", 1)[0].strip()
                cap = row[1].strip()
                if img and cap:
                    caps.setdefault(img, []).append(cap)

    return caps


def save_all_captions():
    CAPTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CAPTIONS_FILE.open("w", encoding="utf-8") as f:
        for img, caps_list in captions.items():
            for c in caps_list:
                f.write(f"{img}\t{c}\n")


def show_image_and_captions(img_name: str):
    if not img_name:
        return None, ""
    img_path = IMAGES_DIR / img_name
    if not img_path.is_file():
        return None, "Image file not found."

    img = Image.open(img_path).convert("RGB")
    lines = captions.get(img_name, [])
    text = "\n".join(lines)
    return img, text


def update_captions(img_name: str, text: str):
    # split text area into lines, strip blanks
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    captions[img_name] = lines
    save_all_captions()
    return f"Saved {len(lines)} caption(s) for {img_name}."


def main():
    global captions
    captions = load_captions()
    img_names = sorted(captions.keys())

    if not img_names:
        raise SystemExit(f"No captions found in {CAPTIONS_FILE}")

    with gr.Blocks(title="Flickr8k Caption Editor") as demo:
        gr.Markdown("# Flickr8k Caption Editor")
        gr.Markdown(
            "Select an image ID, view the image and its captions, "
            "edit them, then click **Save captions** to update `captions.txt`."
        )

        img_dropdown = gr.Dropdown(
            choices=img_names,
            value=img_names[0],
            label="Image filename",
        )

        with gr.Row():
            img_widget = gr.Image(label="Image", interactive=False)
            captions_box = gr.Textbox(
                label="Captions (one per line)",
                lines=8,
                placeholder="Enter captions here...",
            )

        status = gr.Textbox(
            label="Status",
            value="",
            interactive=False,
        )

        def on_select(name):
            return show_image_and_captions(name)

        img_dropdown.change(
            on_select,
            inputs=img_dropdown,
            outputs=[img_widget, captions_box],
        )

        save_btn = gr.Button("Save captions")

        save_btn.click(
            update_captions,
            inputs=[img_dropdown, captions_box],
            outputs=status,
        )

        # load first image on startup
        demo.load(
            on_select,
            inputs=img_dropdown,
            outputs=[img_widget, captions_box],
        )

    demo.launch()


if __name__ == "__main__":
    main()
