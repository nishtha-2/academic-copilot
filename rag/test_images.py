import os
import json

from image_extractor import extract_images

pdf_folder = "../data/documents"
image_folder = "../data/images"

all_images = []

for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(
            pdf_folder,
            file
        )

        images = extract_images(
            pdf_path,
            image_folder
        )

        all_images.extend(images)

with open(
    "../data/metadata/images.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_images,
        f,
        indent=4,
        ensure_ascii=False
    )

print(f"\nSaved {len(all_images)} image records")