"""Gradio demo for the ADAS perception pipeline.

Two modes: detection-only on an arbitrary uploaded image (no LIDAR data
exists for a random upload), and the full calibrated fusion pipeline on a
handful of bundled real KITTI frames (image + calibration + LIDAR point
cloud), which is the actual point of this project.
"""

import os
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

from adas.data.calibration import parse_calib_file
from adas.detection.detector import Detector
from adas.fusion.frustum import fuse_detections_with_points
from adas.fusion.projection import load_velodyne_bin, project_velodyne_to_image

MODEL_REPO = os.environ.get("ADAS_MODEL_REPO", "mokshhere/adas-kitti-yolo11m")
SAMPLES_DIR = Path(__file__).parent / "samples"
SAMPLE_IDS = sorted(p.stem for p in SAMPLES_DIR.glob("*.png"))

_detector: Detector | None = None


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        weights = hf_hub_download(repo_id=MODEL_REPO, filename="best.onnx")
        _detector = Detector(weights, device="cpu")
    return _detector


def draw_plain_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        xmin, ymin, xmax, ymax = det["bbox"]
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, max(0, ymin - 12)), f"{det['cls']} {det['confidence']:.2f}", fill="red")
    return annotated


def draw_fused_detections(image: Image.Image, fused) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for det in fused:
        xmin, ymin, xmax, ymax = det.bbox
        label = f"{det.cls} {det.distance_m:.1f}m" if det.distance_m is not None else f"{det.cls} (no LIDAR)"
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, max(0, ymin - 12)), label, fill="red")
    return annotated


def analyze_upload(image: Image.Image):
    if image is None:
        return None, "Upload an image first."
    detections = get_detector().detect(image)
    summary = f"{len(detections)} object(s) detected. No LIDAR data for an arbitrary upload — detection only."
    return draw_plain_detections(image, detections), summary


def analyze_sample(sample_id: str):
    if not sample_id:
        return None, "Pick a sample frame first."

    image = Image.open(SAMPLES_DIR / f"{sample_id}.png")
    detections = get_detector().detect(image)

    calib = parse_calib_file(SAMPLES_DIR / f"{sample_id}_calib.txt")
    points = load_velodyne_bin(str(SAMPLES_DIR / f"{sample_id}_velodyne.bin"))
    pixels, depth = project_velodyne_to_image(points[:, :3], calib, image.width, image.height)
    fused = fuse_detections_with_points(detections, pixels, depth)

    lines = [
        f"{f.cls}: {f.distance_m:.1f}m ({f.num_points} LIDAR points)" if f.distance_m is not None
        else f"{f.cls}: no LIDAR points landed in this box"
        for f in fused
    ]
    summary = "\n".join(lines) if lines else "No objects detected in this frame."
    return draw_fused_detections(image, fused), summary


with gr.Blocks(title="ADAS Perception Demo") as demo:
    gr.Markdown(
        "# ADAS Perception Demo\n"
        "Calibrated camera-LIDAR fusion + a KITTI-fine-tuned detector. "
        f"[Model](https://huggingface.co/{MODEL_REPO}) · "
        "[Code](https://github.com/ishaannk/ADAS-Object-Detection-and-Collision-Avoidance)"
    )
    with gr.Tab("Upload your own image"):
        gr.Markdown("Detection only — there's no LIDAR sweep for a random upload to fuse against.")
        upload_input = gr.Image(type="pil", label="Image")
        upload_button = gr.Button("Detect", variant="primary")
        upload_output_image = gr.Image(label="Detections")
        upload_output_text = gr.Textbox(label="Summary")
        upload_button.click(analyze_upload, inputs=upload_input, outputs=[upload_output_image, upload_output_text])

    with gr.Tab("Real KITTI frame (full fusion)"):
        gr.Markdown(
            "These frames ship with real KITTI calibration and LIDAR point clouds, so this "
            "runs the actual calibrated fusion pipeline — distances come from real LIDAR "
            "returns projected through the camera's calibration matrices, not a guess."
        )
        sample_dropdown = gr.Dropdown(
            choices=SAMPLE_IDS, value=SAMPLE_IDS[0] if SAMPLE_IDS else None, label="KITTI sample frame"
        )
        sample_button = gr.Button("Run fusion pipeline", variant="primary")
        sample_output_image = gr.Image(label="Detections + LIDAR-fused distance")
        sample_output_text = gr.Textbox(label="Per-object distance")
        sample_button.click(analyze_sample, inputs=sample_dropdown, outputs=[sample_output_image, sample_output_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
