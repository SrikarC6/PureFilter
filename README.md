# PureFilter

An AI-powered photo filter application that automatically detects faces and eyes using computer vision, then composites styled sunglasses and photo effects onto any uploaded image — with zero manual adjustment.

---

## Features

- **Automatic eye detection** — uses OpenCV's multi-scale Haar cascade classifiers to locate face and eye regions in any photo
- **Angle-aware placement** — computes the inter-eye angle and distance to rotate, scale, and position sunglasses precisely along the eye line
- **6 sunglass styles** — Gold (classic), Classic Black, Blue Tint, Amber, Red, Pink; all programmatically generated so no external asset file is required
- **Extra photo effects** — Pink Overlay, Purple Overlay, Vignette (NumPy radial darkening), and Portrait Blur (Gaussian background blur outside the detected face)
- **RGBA compositing** — transparent PNG layers are blended via Pillow's `alpha_composite` for clean, edge-respecting overlays
- **GUI + CLI** — a dark-themed tkinter GUI with drag-and-drop (via `tkinterdnd2`) and a file-dialog fallback; or pass an image path directly on the command line

---

## Installation

```bash
pip install opencv-python pillow numpy
pip install tkinterdnd2   # optional — enables true drag-and-drop
```

Python 3.8+ recommended.

---

## Usage

**GUI mode**
```bash
python ai_photo_filter_applier.py
```
Drop a photo onto the window (or click Browse), choose a sunglass style and optional effect, and the result is saved alongside the original.

**CLI mode**
```bash
python ai_photo_filter_applier.py /path/to/photo.jpg
```
Applies the default Gold style with no extra effect and saves `photo_sunglasses.png` next to the input file.

**Output filename format**
```
<original_name>_<style>_<effect>.png
```
Example: `portrait_gold_vignette.png`

---

## How It Works

1. **Face & eye detection** — OpenCV reads the image, converts to grayscale, and runs `haarcascade_frontalface_alt.xml` followed by `haarcascade_eye.xml` restricted to the upper 55% of the detected face region.
2. **Geometry computation** — eye centers are sorted left-to-right; inter-eye distance and angle are calculated with `atan2`.
3. **Scale & rotation** — the sunglasses asset is scaled so its anchor span matches the detected eye separation, then rotated to match the eye-line angle.
4. **Compositing** — the transformed sunglasses are placed over the base image using RGBA `alpha_composite`; extra effects (tint, vignette, portrait blur) are applied on top.

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Face & eye detection (Haar cascades) |
| `Pillow` | Image compositing, RGBA blending, effects |
| `numpy` | Vignette mask computation, array operations |
| `tkinterdnd2` | Drag-and-drop GUI support *(optional)* |
