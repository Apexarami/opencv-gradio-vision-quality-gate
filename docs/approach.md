# Vision Quality Gate Approach

This project is a rule based computer vision quality gate. It does not pretend to be a trained defect classifier. Its job is to check whether an inspection image is usable and whether it contains visible irregularities that should be reviewed.

## Checks

1. Blur check using Laplacian variance
2. Glare check using bright low saturation pixels
3. Fog or smoke suspicion using contrast and edge density
4. Primary part region detection
5. Surface irregularity detection using blackhat and tophat morphology
6. Inspection score and decision generation

## Decision labels

- Acceptable
- Review recommended
- High defect risk
- Re-capture image

## Why this is useful

In a real inspection system, not every camera image should be sent directly to a defect classifier. Some images are too blurry, too bright, too dark, or affected by glare or haze. A quality gate can detect these issues first and request a re-capture or human review.
