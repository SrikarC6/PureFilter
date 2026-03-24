"""
photo_filter.py
───────────────────────────────────────────────────────────────────────────────
AI Photo Filter

    pip install opencv-python pillow numpy
    pip install tkinterdnd2          # optional — enables true drag-and-drop

Run:
    python photo_filter.py
    python photo_filter.py /path/to/photo.jpg   # skips GUI
───────────────────────────────────────────────────────────────────────────────
"""

import math
import os
import sys
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw, ImageFilter


def base_plus_lens(base, lens, dest=(0, 0), scale=1, angle=0):
    pasted = base.convert("RGBA")

    lens = lens.rotate(angle, expand=True)

    new_width = round(scale * lens.width)
    new_height = round(scale * lens.height)
    lens = lens.resize((new_width, new_height))

    pasted.alpha_composite(lens, dest=dest)
    return pasted


def detect_visualize_eyes(image_filename, eye_model='eye'):
    frame = cv.imread(image_filename)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    face_cascade = cv.CascadeClassifier(
        cv.data.haarcascades + 'haarcascade_frontalface_alt.xml'
    )
    faces = face_cascade.detectMultiScale(frame_gray)

    if eye_model == 'eye':
        cv_eye_model_file = cv.data.haarcascades + 'haarcascade_eye.xml'
    else:
        cv_eye_model_file = cv.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'

    eyes_center = None
    visualized_image = None

    if len(faces) > 0:

        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces_sorted[0]

        faceROI = frame_gray[fy: fy + int(fh * 0.55), fx: fx + fw]

        eyes_cascade = cv.CascadeClassifier(cv_eye_model_file)
        eyes_raw = eyes_cascade.detectMultiScale(
            faceROI,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(20, 20)
        )

        eyes = []
        for (ex, ey, ew, eh) in eyes_raw:
            eyes.append((fx + ex, fy + ey, ew, eh))

        vis_frame = frame.copy()
        cv.rectangle(vis_frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        for (ex, ey, ew, eh) in eyes:
            cx = ex + ew // 2
            cy = ey + eh // 2
            cv.circle(vis_frame, (cx, cy), 4, (0, 255, 0), -1)
            cv.rectangle(vis_frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        visualized_image = Image.fromarray(cv.cvtColor(vis_frame, cv.COLOR_BGR2RGB))
        eyes_center = eyes if len(eyes) > 0 else None

    else:
        print("no face detected")
        faces = None
        eyes_center = None
        visualized_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        visualized_image = Image.fromarray(visualized_image)

    return faces, eyes_center, visualized_image


def eyes_angle_degrees(eyes):
    """
    Takes the eyes list from detect_visualize_eyes — list of
    (x, y, w, h) tuples in full-image coordinates — and returns:

        (angle_eyes_degrees, left_eye_xy, right_eye_xy)

    left_eye_xy and right_eye_xy are (x, y) center points.
    angle_eyes_degrees is counter-clockwise degrees of right eye
    relative to left eye.
    """
    if eyes is None or len(eyes) < 2:
        raise ValueError("Need at least 2 detected eyes to compute angle.")

    centers = [(x + w // 2, y + h // 2) for (x, y, w, h) in eyes]

    centers.sort(key=lambda c: c[0])

    left_eye_xy  = centers[0]
    right_eye_xy = centers[1]

    dx = right_eye_xy[0] - left_eye_xy[0]
    dy = right_eye_xy[1] - left_eye_xy[1]

    angle_eyes = math.degrees(math.atan2(-dy, dx))

    return angle_eyes, left_eye_xy, right_eye_xy


def distance_2d_points(point1_xy, point2_xy):
    x1 = point1_xy[0]
    y1 = point1_xy[1]

    x2 = point2_xy[0]
    y2 = point2_xy[1]

    distance_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
    distance = np.sqrt(distance_squared)

    return distance


def rotate_img_coords(x, y, w, h, theta_radians=0):

    r = math.sqrt(x ** 2 + y ** 2)
    gamma_radians = math.atan2(y, x)

    net_rotation = gamma_radians - theta_radians

    x_bar = r * math.cos(net_rotation)
    y_bar = r * math.sin(net_rotation)

    if theta_radians <= 0:
        x_new = x_bar + h * math.sin(theta_radians)
        y_new = y_bar
    else:
        x_new = x_bar
        y_new = y_bar + w * math.sin(theta_radians)

    x_new = int(x_new)
    y_new = int(y_new)

    return x_new, y_new


def compute_rotated_lens_dest(img_left_eye_xy, sun_left_eye_xy, sun_width,
                               sun_height, angle_eyes_degrees, new_scale):

    sun_left_eye_x = sun_left_eye_xy[0]
    sun_left_eye_y = sun_left_eye_xy[1]

    ang_eyes_radians = angle_eyes_degrees * math.pi / 180

    sun_leye_new_x, sun_leye_new_y = rotate_img_coords(
        sun_left_eye_x, sun_left_eye_y,
        sun_width, sun_height,
        ang_eyes_radians
    )

    sun_leye_new_x = new_scale * sun_leye_new_x
    sun_leye_new_y = new_scale * sun_leye_new_y

    img_left_eye_x = img_left_eye_xy[0]
    img_left_eye_y = img_left_eye_xy[1]

    dest_x = img_left_eye_x - sun_leye_new_x
    dest_y = img_left_eye_y - sun_leye_new_y

    dest = (int(dest_x), int(dest_y))

    return dest
                                 

sun_left_eye_x   = 175
sun_left_eye_y   = 110
sun_left_eye_xy  = (175, 110)
sun_right_eye_xy = (485, 110)


def apply_sunglasses_to_file(image_filename, sunglasses_img):
    base_img = Image.open(image_filename)

    faces, eyes, visualized_image = detect_visualize_eyes(image_filename, 'eye')

    if eyes is None or len(eyes) < 2:
        print("Eyes not detected — returning original image.")
        return base_img

    angle_eyes_degrees, img_left_eye_xy, img_right_eye_xy = eyes_angle_degrees(eyes)

    distance_lens_eyes = distance_2d_points(sun_left_eye_xy, sun_right_eye_xy)
    distance_img_eyes  = distance_2d_points(img_left_eye_xy, img_right_eye_xy)
    estimated_scale    = distance_img_eyes / distance_lens_eyes

    print("Estimated scale    :", estimated_scale)
    print("Left eye location  :", img_left_eye_xy)
    print(f"Right eye is at {angle_eyes_degrees:.1f}° (counter-clockwise) relative to left eye.")

    sun_width, sun_height = sunglasses_img.size

    dest = compute_rotated_lens_dest(
        img_left_eye_xy,
        sun_left_eye_xy,
        sun_width, sun_height,
        angle_eyes_degrees,
        estimated_scale
    )

    print("Dest computed by AI:", dest)

    result = base_plus_lens(
        base_img, sunglasses_img,
        dest=dest,
        angle=angle_eyes_degrees,
        scale=estimated_scale
    )
    return result


def make_sunglasses_image(style="gold"):
    """
    Generates a sunglasses PNG for the given style.
    All variants keep left-eye anchor at (175,110) and right-eye at (485,110)
    so the course placement math works correctly for every style.

    Styles (added — not course code):
      "gold"    — yellow/gold circular  (matches the course screenshot)
      "black"   — classic black oval
      "blue"    — blue tinted
      "amber"   — amber/orange tinted
      "red"     — red tinted
      "pink"    — pink overlay (matches the color-circle effect in screenshot)
    """
    W, H = 660, 220
    img  = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Lens fill and frame colour per style
    styles = {
        "gold":  ((220, 180, 20,  210), (140, 100,  0, 255)),
        "black": (( 20,  20, 20,  220), ( 10,  10, 10, 255)),
        "blue":  (( 20,  80, 200, 190), (  0,  40, 140, 255)),
        "amber": ((200, 120,  20, 200), (140,  80,   0, 255)),
        "red":   ((200,  30,  30, 210), (140,   0,   0, 255)),
        "pink":  ((220,  60, 180, 200), (160,   0, 130, 255)),
    }
    lens_fill, frame_col = styles.get(style, styles["gold"])
    lw = 5

    # Left lens centred on sun_left_eye_xy  = (175, 110)
    left_lens  = [175 - 130, 110 - 80, 175 + 130, 110 + 80]
    # Right lens centred on sun_right_eye_xy = (485, 110)
    right_lens = [485 - 130, 110 - 80, 485 + 130, 110 + 80]

    draw.ellipse(left_lens,  fill=lens_fill, outline=frame_col, width=lw)
    draw.ellipse(right_lens, fill=lens_fill, outline=frame_col, width=lw)

    # Bridge
    draw.line([(175 + 130, 110), (485 - 130, 110)], fill=frame_col, width=lw)
    # Temples
    draw.line([(0, 110), (175 - 130, 110)], fill=frame_col, width=lw)
    draw.line([(485 + 130, 110), (W, 110)], fill=frame_col, width=lw)

    return img


def apply_color_tint(img, color_rgb, opacity=90):
    """
    Semi-transparent color circle over each eye region — matches the pink
    circle overlay shown in the course screenshot section 0.
    Falls back to a full-image tint if no face is detected.
    """
    from PIL import ImageDraw as _ID
    result = img.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = _ID.Draw(overlay)
    w, h = result.size
    # Draw two soft circles roughly where eyes sit (upper-centre of image)
    r = w // 8
    cx1, cx2 = w // 3, 2 * w // 3
    cy = h // 3
    fill = (*color_rgb, opacity)
    draw.ellipse([cx1 - r, cy - r, cx1 + r, cy + r], fill=fill)
    draw.ellipse([cx2 - r, cy - r, cx2 + r, cy + r], fill=fill)
    return Image.alpha_composite(result, overlay)


def apply_vignette(img, strength=0.6):
    """
    Radial darkening at the edges using NumPy — same as original generated
    version.
    """
    arr = np.array(img.convert("RGBA"), dtype=np.float32)
    h, w = arr.shape[:2]
    Y, X = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]
    dist  = np.sqrt(X ** 2 + Y ** 2)
    mask  = np.clip(1.0 - dist * strength, 0.0, 1.0)[:, :, np.newaxis]
    arr[:, :, :3] *= mask
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


def apply_portrait_blur(img, image_filename):
    """
    Blurs the background outside the detected face region.
    """
    faces, _, _ = detect_visualize_eyes(image_filename, 'eye')
    result = img.convert("RGBA")
    if faces is None or len(faces) == 0:
        return result
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    fx, fy, fw, fh = faces_sorted[0]
    blurred = result.filter(ImageFilter.GaussianBlur(radius=8))
    expand  = int(min(fw, fh) * 0.2)
    mask    = Image.new("L", result.size, 0)
    ImageDraw.Draw(mask).rectangle(
        [fx - expand, fy - expand, fx + fw + expand, fy + fh + expand],
        fill=255
    )
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1, expand // 2)))
    return Image.composite(result, blurred, mask)
  

def run_gui():

    try:
        from tkinterdnd2 import TkinterDnD, DND_FILES
        USE_DND = True
    except ImportError:
        USE_DND = False

    import tkinter as tk
    from tkinter import filedialog

    root = TkinterDnD.Tk() if USE_DND else tk.Tk()
    root.title("AI Photo Filter — Sunglasses")
    root.resizable(False, False)
    root.configure(bg="#1e1e2e")

    ZONE_W, ZONE_H, PAD = 500, 200, 16

    tk.Label(root, text="AI Photo Filter",
             font=("Helvetica", 20, "bold"),
             bg="#1e1e2e", fg="#cdd6f4").pack(pady=(PAD, 2))
    tk.Label(root, text="Detects eyes · places sunglasses · applies effects",
             font=("Helvetica", 11),
             bg="#1e1e2e", fg="#a6adc8").pack(pady=(0, PAD))

    sel_frame = tk.Frame(root, bg="#1e1e2e")
    sel_frame.pack(padx=PAD, fill="x")

    style_var = tk.StringVar(value="gold")
    style_frame = tk.LabelFrame(sel_frame, text="Sunglass Style",
                                bg="#1e1e2e", fg="#cdd6f4",
                                font=("Helvetica", 10, "bold"))
    style_frame.pack(side="left", padx=(0, 12), pady=4, fill="y")
    for label, val in [("Gold  (course)", "gold"), ("Classic Black", "black"),
                       ("Blue Tint", "blue"), ("Amber", "amber"),
                       ("Red", "red"), ("Pink", "pink")]:
        tk.Radiobutton(style_frame, text=label, variable=style_var, value=val,
                       bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
                       activebackground="#1e1e2e",
                       font=("Helvetica", 10)).pack(anchor="w", padx=8, pady=1)

    effect_var = tk.StringVar(value="none")
    effect_frame = tk.LabelFrame(sel_frame, text="Extra Effect",
                                 bg="#1e1e2e", fg="#cdd6f4",
                                 font=("Helvetica", 10, "bold"))
    effect_frame.pack(side="left", pady=4, fill="y")
    for label, val in [("None", "none"), ("Pink Overlay", "pink_overlay"),
                       ("Purple Overlay", "purple_overlay"),
                       ("Vignette", "vignette"), ("Portrait Blur", "portrait_blur")]:
        tk.Radiobutton(effect_frame, text=label, variable=effect_var, value=val,
                       bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
                       activebackground="#1e1e2e",
                       font=("Helvetica", 10)).pack(anchor="w", padx=8, pady=1)

    drop_frame = tk.Frame(root, width=ZONE_W, height=ZONE_H,
                          bg="#313244",
                          highlightbackground="#585b70",
                          highlightthickness=2)
    drop_frame.pack(padx=PAD, pady=(PAD, 0))
    drop_frame.pack_propagate(False)

    drop_label = tk.Label(
        drop_frame,
        text="\u2b07  Drop a photo here" if USE_DND else "Click here or use Browse to select a photo",
        font=("Helvetica", 14),
        bg="#313244", fg="#89b4fa",
        wraplength=ZONE_W - 40, justify="center"
    )
    drop_label.place(relx=0.5, rely=0.38, anchor="center")

    tk.Label(drop_frame, text="JPG \u00b7 PNG \u00b7 BMP \u00b7 WEBP",
             font=("Helvetica", 10),
             bg="#313244", fg="#585b70").place(relx=0.5, rely=0.65, anchor="center")

    status_var = tk.StringVar(value="Ready.")
    tk.Label(root, textvariable=status_var,
             font=("Helvetica", 10),
             bg="#1e1e2e", fg="#a6e3a1",
             anchor="w").pack(fill="x", padx=PAD, pady=(8, 0))
─
    def process_image(path):
        path = path.strip().strip("{}")
        if not os.path.isfile(path):
            status_var.set(f"File not found: {path}")
            return
        style  = style_var.get()
        effect = effect_var.get()
        drop_label.config(text="\u2699  Processing…")
        root.update()
        status_var.set(f"Processing: {os.path.basename(path)}  |  style={style}  effect={effect}")
        root.update()
        try:
            if style == "gold" and os.path.isfile("sunglasses.png"):
                sg_img = Image.open("sunglasses.png")
            else:
                sg_img = make_sunglasses_image(style=style)

            result = apply_sunglasses_to_file(path, sg_img)

            if effect == "pink_overlay":
                result = apply_color_tint(result, (220, 60, 180), opacity=90)
            elif effect == "purple_overlay":
                result = apply_color_tint(result, (130, 50, 220), opacity=90)
            elif effect == "vignette":
                result = apply_vignette(result, strength=0.65)
            elif effect == "portrait_blur":
                result = apply_portrait_blur(result, path)

            base, _ = os.path.splitext(path)
            out_path = f"{base}_{style}_{effect}.png"
            result.convert("RGB").save(out_path)
            result.show()
            drop_label.config(
                text="\u2705  Done! Drop another photo." if USE_DND else "\u2705  Done!"
            )
            status_var.set(f"Saved \u2192 {out_path}")
        except Exception as e:
            status_var.set(f"Error: {e}")
            drop_label.config(text="\u26a0  Something went wrong \u2014 see console")
            raise

    def browse():
        path = filedialog.askopenfilename(
            title="Select a photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All files", "*.*")]
        )
        if path:
            process_image(path)

    btn_frame = tk.Frame(root, bg="#1e1e2e")
    btn_frame.pack(pady=PAD)

    tk.Button(btn_frame, text="  Browse…  ",
              font=("Helvetica", 12, "bold"),
              bg="#89b4fa", fg="#1e1e2e",
              relief="flat", padx=16, pady=8,
              cursor="hand2", command=browse).pack(side="left", padx=8)

    tk.Button(btn_frame, text="  Quit  ",
              font=("Helvetica", 12),
              bg="#313244", fg="#cdd6f4",
              relief="flat", padx=16, pady=8,
              cursor="hand2", command=root.destroy).pack(side="left", padx=8)

    if USE_DND:
        def on_drop(event):
            process_image(event.data)
        for widget in (drop_frame, drop_label):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", on_drop)
    else:
        drop_frame.bind("<Button-1>", lambda e: browse())
        drop_label.bind("<Button-1>", lambda e: browse())
        tk.Label(root,
                 text="Tip: pip install tkinterdnd2  for true drag-and-drop",
                 font=("Helvetica", 9),
                 bg="#1e1e2e", fg="#45475a").pack(pady=(0, 8))

    root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command-line mode: gold style, no effect
        image_path = sys.argv[1]
        print(f"Processing: {image_path}")
        sg_img = Image.open("sunglasses.png") if os.path.isfile("sunglasses.png") \
                 else make_sunglasses_image(style="gold")
        result = apply_sunglasses_to_file(image_path, sg_img)
        if result:
            base, _ = os.path.splitext(image_path)
            out = base + "_sunglasses.png"
            result.convert("RGB").save(out)
            result.show()
            print(f"Saved -> {out}")
    else:
        run_gui()
