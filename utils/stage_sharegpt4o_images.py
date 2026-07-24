import os
import sys
import zipfile

from lmm_serving import config as cfg

from huggingface_hub import hf_hub_download

os.makedirs(cfg.IMAGE_DIR, exist_ok=True)
print(f"IMAGE_DIR : {cfg.IMAGE_DIR}")
print(f"HF_HOME   : {os.environ.get('HF_HOME', '~/.cache/huggingface')}")

print("Downloading images.zip (~6.5 GB; cached after first run)...")
zip_path = hf_hub_download("OpenGVLab/ShareGPT-4o", "images.zip", repo_type="dataset")
print(f"Archive   : {zip_path}")

written = skipped = 0
with zipfile.ZipFile(zip_path) as zf:
    members = [
        m for m in zf.namelist() if m.lower().endswith(".jpg") and not m.endswith("/")
    ]
    print(f"Extracting {len(members)} images...")
    for i, m in enumerate(members):
        # Flatten any internal folder structure: trace ids are bare basenames.
        out = os.path.join(cfg.IMAGE_DIR, os.path.basename(m))
        if os.path.exists(out) and os.path.getsize(out) > 0:
            skipped += 1
            continue
        with zf.open(m) as src, open(out, "wb") as dst:
            dst.write(src.read())
        written += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(members)}  (written={written} skipped={skipped})")

print(f"Done. written={written} skipped={skipped} -> {cfg.IMAGE_DIR}")
