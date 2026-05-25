# Vision Quality Gate with OpenCV and Gradio

A lightweight computer vision demo for visual quality inspection. The project uses OpenCV to detect blur, glare, contour irregularities, and possible surface defects, then presents the result in a Gradio interface.

This project is intentionally different from a backend dashboard project. It is built as an interactive vision demo, not as another FastAPI or Streamlit app.

## Features

- Gradio based image inspection interface
- Upload any JPG or PNG image
- Detect blur using Laplacian variance
- Detect glare using bright pixel ratio
- Detect contour irregularities using OpenCV edges and contours
- Generate an inspection decision
- Return annotated image with highlighted regions
- Generate synthetic sample inspection images
- Includes pytest tests and GitHub Actions CI

## Quick start

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.generate_samples
python -m app.gradio_app
```

Open the local Gradio URL shown in the terminal, usually:

```text
http://127.0.0.1:7860
```

## Resume version

**Vision Quality Gate with OpenCV and Gradio**

**Goal:**  
Develop an interactive computer vision workflow for checking image quality and possible surface defects in inspection images.

**Description:**  
- Built a Gradio based image inspection demo using OpenCV for contour and defect visualization.  
- Added blur, glare, and irregular contour checks to support image quality assessment.  
- Generated annotated outputs with inspection score, decision label, and sample test images.

**Tools:**  
Python, OpenCV, Gradio, NumPy, Pillow, Pytest, GitHub Actions

## Limitations

- This project uses classical computer vision, not a trained deep learning model.
- It is suitable for a demo workflow and basic inspection logic.
- Future work can add YOLO, segmentation, live camera input, and dataset based validation.
