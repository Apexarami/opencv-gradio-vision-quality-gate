from pathlib import Path

import gradio as gr
from PIL import Image

from src.inspection import inspect_image


def run_inspection(image: Image.Image):
    if image is None:
        return None, None, "Please upload an image first."

    result = inspect_image(image)
    return result.annotated_image, result.defect_mask, result.summary


def load_sample(sample_name: str):
    sample_path = Path("data/samples") / sample_name
    if not sample_path.exists():
        return None
    return Image.open(sample_path)


with gr.Blocks(title="Vision Quality Gate") as demo:
    gr.Markdown(
        """
        # Vision Quality Gate with OpenCV and Gradio

        Upload an inspection image or choose a generated sample. The tool checks whether
        the image is usable for inspection and highlights possible visual irregularities.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            sample_dropdown = gr.Dropdown(
                choices=[
                    "clean_part.png",
                    "scratched_part.png",
                    "glare_part.png",
                    "blurry_part.png",
                ],
                label="Load sample image",
            )
            input_image = gr.Image(type="pil", label="Input image")
            inspect_button = gr.Button("Run inspection", variant="primary")

        with gr.Column(scale=1):
            annotated_output = gr.Image(type="pil", label="Annotated inspection result")
            mask_output = gr.Image(type="pil", label="Detected irregularity mask")
            summary_output = gr.Textbox(label="Inspection report", lines=16)

    sample_dropdown.change(fn=load_sample, inputs=sample_dropdown, outputs=input_image)
    inspect_button.click(
        fn=run_inspection,
        inputs=input_image,
        outputs=[annotated_output, mask_output, summary_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )
