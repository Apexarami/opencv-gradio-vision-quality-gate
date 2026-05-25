from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class InspectionResult:
    decision: str
    inspection_score: float
    quality_status: str
    defect_status: str
    blur_score: float
    glare_ratio: float
    smoke_fog_score: float
    defect_area_ratio: float
    defect_region_count: int
    annotated_image: Image.Image
    defect_mask: Image.Image
    summary: str


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_for_processing(image: np.ndarray, max_width: int = 950) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / width
    return cv2.resize(image, (max_width, int(height * scale)))


def inspect_image(image: Image.Image) -> InspectionResult:
    bgr = resize_for_processing(pil_to_bgr(image))
    annotated = bgr.copy()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    glare_mask = cv2.inRange(hsv, np.array([0, 0, 245]), np.array([180, 65, 255]))
    glare_ratio = float(np.count_nonzero(glare_mask) / glare_mask.size)

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 150)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    smoke_fog_score = calculate_smoke_fog_score(
        brightness=brightness,
        contrast=contrast,
        edge_density=edge_density,
    )

    roi_x, roi_y, roi_w, roi_h = find_primary_roi(gray)
    cv2.rectangle(annotated, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 180, 0), 2)

    roi_gray = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    defect_mask_roi, defect_contours = detect_surface_irregularities(roi_gray)

    full_defect_mask = np.zeros_like(gray)
    full_defect_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = defect_mask_roi

    defect_region_count = 0
    for contour in defect_contours:
        area = cv2.contourArea(contour)
        if area < 45:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 4 or h < 4:
            continue
        defect_region_count += 1
        cv2.rectangle(
            annotated,
            (roi_x + x, roi_y + y),
            (roi_x + x + w, roi_y + y + h),
            (0, 0, 255),
            2,
        )

    roi_area = max(roi_w * roi_h, 1)
    defect_area_ratio = float(np.count_nonzero(defect_mask_roi) / roi_area)

    quality_status = classify_quality(
        blur_score=blur_score,
        glare_ratio=glare_ratio,
        smoke_fog_score=smoke_fog_score,
        brightness=brightness,
    )

    defect_status = classify_defect(
        defect_area_ratio=defect_area_ratio,
        defect_region_count=defect_region_count,
    )

    inspection_score = calculate_score(
        blur_score=blur_score,
        glare_ratio=glare_ratio,
        smoke_fog_score=smoke_fog_score,
        defect_area_ratio=defect_area_ratio,
        defect_region_count=defect_region_count,
    )

    decision = final_decision(quality_status, defect_status, inspection_score)

    add_overlay_text(annotated, decision, inspection_score, quality_status, defect_status)

    summary = build_summary(
        decision=decision,
        inspection_score=inspection_score,
        quality_status=quality_status,
        defect_status=defect_status,
        blur_score=blur_score,
        brightness=brightness,
        contrast=contrast,
        glare_ratio=glare_ratio,
        smoke_fog_score=smoke_fog_score,
        edge_density=edge_density,
        defect_area_ratio=defect_area_ratio,
        defect_region_count=defect_region_count,
    )

    return InspectionResult(
        decision=decision,
        inspection_score=round(inspection_score, 2),
        quality_status=quality_status,
        defect_status=defect_status,
        blur_score=round(blur_score, 2),
        glare_ratio=round(glare_ratio, 4),
        smoke_fog_score=round(smoke_fog_score, 2),
        defect_area_ratio=round(defect_area_ratio, 4),
        defect_region_count=defect_region_count,
        annotated_image=bgr_to_pil(annotated),
        defect_mask=Image.fromarray(full_defect_mask),
        summary=summary,
    )


def find_primary_roi(gray: np.ndarray) -> tuple[int, int, int, int]:
    height, width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 120)
    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0, 0, width, height

    min_area = width * height * 0.04
    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not candidates:
        return 0, 0, width, height

    largest = max(candidates, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    pad = 8
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(width - x, w + 2 * pad)
    h = min(height - y, h + 2 * pad)

    return x, y, w, h


def detect_surface_irregularities(roi_gray: np.ndarray) -> tuple[np.ndarray, list]:
    roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)

    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    blackhat = cv2.morphologyEx(roi_gray, cv2.MORPH_BLACKHAT, blackhat_kernel)

    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    tophat = cv2.morphologyEx(roi_gray, cv2.MORPH_TOPHAT, tophat_kernel)

    _, dark_defects = cv2.threshold(blackhat, 22, 255, cv2.THRESH_BINARY)
    _, bright_defects = cv2.threshold(tophat, 38, 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_or(dark_defects, bright_defects)

    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_mask = np.zeros_like(combined)
    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 45:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)

        if area >= 45 and 0.05 <= aspect <= 20:
            filtered_contours.append(contour)
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return filtered_mask, filtered_contours


def calculate_smoke_fog_score(brightness: float, contrast: float, edge_density: float) -> float:
    low_contrast_component = max(0.0, (48.0 - contrast) / 48.0)
    low_edge_component = max(0.0, (0.055 - edge_density) / 0.055)
    bright_haze_component = 1.0 if brightness > 115 else 0.5 if brightness > 90 else 0.0

    score = 100.0 * (
        0.45 * low_contrast_component
        + 0.40 * low_edge_component
        + 0.15 * bright_haze_component
    )
    return float(min(100.0, max(0.0, score)))


def classify_quality(blur_score: float, glare_ratio: float, smoke_fog_score: float, brightness: float) -> str:
    if brightness < 45:
        return "Re-capture: too dark"
    if blur_score < 65:
        return "Re-capture: blurry image"
    if glare_ratio > 0.055:
        return "Re-capture: strong glare"
    if smoke_fog_score > 72:
        return "Re-capture: fog or smoke suspected"
    if blur_score < 130 or glare_ratio > 0.025 or smoke_fog_score > 48:
        return "Usable but quality warning"
    return "Image quality acceptable"


def classify_defect(defect_area_ratio: float, defect_region_count: int) -> str:
    if defect_area_ratio > 0.035 or defect_region_count >= 8:
        return "High surface irregularity"
    if defect_area_ratio > 0.012 or defect_region_count >= 3:
        return "Review surface irregularity"
    return "No major surface irregularity"


def calculate_score(
    blur_score: float,
    glare_ratio: float,
    smoke_fog_score: float,
    defect_area_ratio: float,
    defect_region_count: int,
) -> float:
    blur_component = 25 if blur_score < 65 else 14 if blur_score < 130 else 0
    glare_component = min(glare_ratio * 500, 25)
    haze_component = min(smoke_fog_score * 0.25, 25)
    defect_component = min(defect_area_ratio * 900 + defect_region_count * 4.5, 45)

    return float(min(100.0, blur_component + glare_component + haze_component + defect_component))


def final_decision(quality_status: str, defect_status: str, score: float) -> str:
    if quality_status.startswith("Re-capture"):
        return "Re-capture image"
    if score >= 70 or defect_status.startswith("High"):
        return "High defect risk"
    if score >= 35 or defect_status.startswith("Review") or "warning" in quality_status.lower():
        return "Review recommended"
    return "Acceptable"


def add_overlay_text(
    annotated: np.ndarray,
    decision: str,
    inspection_score: float,
    quality_status: str,
    defect_status: str,
) -> None:
    if decision == "Acceptable":
        color = (0, 180, 0)
    elif decision == "Review recommended":
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 92), (245, 245, 245), -1)
    cv2.putText(annotated, f"{decision} | Score {inspection_score:.1f}", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(annotated, f"Quality: {quality_status}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 40, 40), 1)
    cv2.putText(annotated, f"Surface: {defect_status}", (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 40, 40), 1)


def build_summary(
    decision: str,
    inspection_score: float,
    quality_status: str,
    defect_status: str,
    blur_score: float,
    brightness: float,
    contrast: float,
    glare_ratio: float,
    smoke_fog_score: float,
    edge_density: float,
    defect_area_ratio: float,
    defect_region_count: int,
) -> str:
    return (
        f"Final decision: {decision}\n"
        f"Inspection score: {inspection_score:.1f}/100\n\n"
        "Image quality checks:\n"
        f"- Quality status: {quality_status}\n"
        f"- Blur score: {blur_score:.2f}\n"
        f"- Brightness: {brightness:.2f}\n"
        f"- Contrast: {contrast:.2f}\n"
        f"- Edge density: {edge_density:.4f}\n"
        f"- Glare ratio: {glare_ratio:.4f}\n"
        f"- Fog or smoke score: {smoke_fog_score:.2f}/100\n\n"
        "Surface inspection:\n"
        f"- Surface status: {defect_status}\n"
        f"- Defect area ratio: {defect_area_ratio:.4f}\n"
        f"- Detected irregular regions: {defect_region_count}\n\n"
        "Recommended action:\n"
        f"- {recommended_action(decision)}\n\n"
        "Note: This is a rule based OpenCV quality gate. It is useful for image quality screening "
        "and visible irregularity detection. It is not a trained defect classifier."
    )


def recommended_action(decision: str) -> str:
    if decision == "Re-capture image":
        return "Capture a new image before inspection because the current image quality is not reliable."
    if decision == "High defect risk":
        return "Send the part for human review or a stronger AI inspection model."
    if decision == "Review recommended":
        return "Keep the image for inspection, but mark it for additional review."
    return "Image can pass the first quality gate."
