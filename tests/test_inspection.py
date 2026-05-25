from PIL import Image

from src.generate_samples import create_scratched_part
from src.inspection import inspect_image


def test_inspection_returns_result(tmp_path):
    image_path = tmp_path / "scratched_part.png"
    create_scratched_part(str(image_path))

    image = Image.open(image_path)
    result = inspect_image(image)

    assert result.inspection_score >= 0
    assert result.decision in {"Acceptable", "Review recommended", "High defect risk"}
    assert result.annotated_image is not None
    assert result.defect_mask is not None
    assert "Inspection score" in result.summary
