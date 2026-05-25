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

<img width="1639" height="901" alt="image" src="https://github.com/user-attachments/assets/1dba074e-e6f1-48ce-bc05-2acd1576aae5" />


## Limitations

- This project uses classical computer vision, not a trained deep learning model.
- It is suitable for a demo workflow and basic inspection logic.
- Future work can add YOLO, segmentation, live camera input, and dataset based validation.
