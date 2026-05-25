from pathlib import Path

import cv2
import numpy as np


def create_clean_part(path: str):
    image = np.full((520, 720, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (160, 120), (560, 390), (95, 140, 185), -1)
    cv2.rectangle(image, (160, 120), (560, 390), (45, 75, 120), 3)
    cv2.circle(image, (280, 255), 45, (220, 220, 225), -1)
    cv2.circle(image, (440, 255), 45, (220, 220, 225), -1)
    cv2.putText(image, "CLEAN SAMPLE", (220, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    cv2.imwrite(path, image)


def create_scratched_part(path: str):
    image = np.full((520, 720, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (160, 120), (560, 390), (95, 140, 185), -1)
    cv2.rectangle(image, (160, 120), (560, 390), (45, 75, 120), 3)
    cv2.line(image, (215, 170), (510, 335), (25, 25, 25), 5)
    cv2.line(image, (240, 340), (500, 190), (40, 40, 40), 3)
    cv2.circle(image, (360, 255), 35, (30, 30, 30), -1)
    cv2.putText(image, "SCRATCH SAMPLE", (210, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    cv2.imwrite(path, image)


def create_glare_part(path: str):
    image = np.full((520, 720, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (160, 120), (560, 390), (95, 140, 185), -1)
    cv2.circle(image, (450, 200), 70, (255, 255, 255), -1)
    cv2.ellipse(image, (380, 265), (125, 45), 20, 0, 360, (250, 250, 250), -1)
    cv2.rectangle(image, (160, 120), (560, 390), (45, 75, 120), 3)
    cv2.putText(image, "GLARE SAMPLE", (225, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    cv2.imwrite(path, image)


def create_blurry_part(path: str):
    image = np.full((520, 720, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (160, 120), (560, 390), (95, 140, 185), -1)
    cv2.line(image, (230, 180), (500, 330), (25, 25, 25), 4)
    image = cv2.GaussianBlur(image, (29, 29), 0)
    cv2.putText(image, "BLUR SAMPLE", (235, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    cv2.imwrite(path, image)


if __name__ == "__main__":
    out_dir = Path("data/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    create_clean_part(str(out_dir / "clean_part.png"))
    create_scratched_part(str(out_dir / "scratched_part.png"))
    create_glare_part(str(out_dir / "glare_part.png"))
    create_blurry_part(str(out_dir / "blurry_part.png"))

    print("Sample inspection images created in data/samples")
