import os

from trainer import model
from trainer.config import OUT_MODEL_PATH
from trainer.datafill import dem

if not os.path.exists(OUT_MODEL_PATH):
    print("Model file not found. Choosing default model...")
    model.train()
else:
    print("Model file already exists.")

print("Downloading tiles...")
dem.download_tiles()

print("All setup complete.")
