"""
SAM3 Detection Service Server.

This server provides an HTTP endpoint for SAM3 text-prompted segmentation.
It receives RGB images and returns detection results (masks, boxes, scores).

Usage:
    python sam3_server.py [--port PORT] [--host HOST]

Example request from client:
    POST /sam3/predict
    - image: RGB image file
    - prompt: text prompt (e.g., "knob", "handle")
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import zipfile
from typing import Optional

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from PIL import Image

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model and processor (loaded once at startup)
model: Optional[object] = None
processor: Optional[Sam3Processor] = None


def load_model():
    """Load SAM3 model and processor."""
    global model, processor
    logger.info("Loading SAM3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    logger.info("SAM3 model loaded successfully.")


def mask_to_rle(mask: np.ndarray) -> dict:
    """Convert binary mask to RLE format for efficient transmission.

    RLE format: alternating counts of [zeros, ones, zeros, ones, ...]
    Starting with count of zeros (which may be 0 if mask starts with 1).
    """
    pixels = mask.flatten().astype(np.uint8)

    # Find where values change
    diff = np.diff(pixels)
    change_indices = np.where(diff != 0)[0] + 1

    # Build run lengths
    # Start with index 0, add all change points, end with length
    boundaries = np.concatenate([[0], change_indices, [len(pixels)]])
    run_lengths = np.diff(boundaries)

    # If the first pixel is 1, prepend a 0 (zero zeros before first run of 1s)
    if pixels[0] == 1:
        run_lengths = np.concatenate([[0], run_lengths])

    return {"counts": run_lengths.tolist(), "size": list(mask.shape)}


def rle_to_mask(rle: dict) -> np.ndarray:
    """Convert RLE back to binary mask."""
    shape = rle["size"]
    counts = rle["counts"]
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    idx = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:
            mask[idx : idx + count] = 1
        idx += count
    return mask.reshape(shape)


@app.route("/sam3/predict", methods=["POST"])
def predict():
    """
    Handle detection request.

    Expects:
        - image: RGB image file (multipart/form-data)
        - prompt: text prompt for detection (form field or query param)

    Returns:
        JSON with masks (RLE), boxes, scores, and mask count.
    """
    global processor

    if processor is None:
        return jsonify({"error": "Model not loaded"}), 500

    # Get image from request
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    prompt = request.form.get("prompt") or request.args.get("prompt")

    if not prompt:
        return jsonify({"error": "No text prompt provided"}), 400

    try:
        # Load image
        image = Image.open(image_file.stream).convert("RGB")
        logger.info(f"Received image: {image.size}, prompt: '{prompt}'")

        # Run inference
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output["masks"].squeeze(1)  # (N, H, W) tensor
        boxes = output["boxes"]  # (N, 4) tensor [x1, y1, x2, y2]
        scores = output["scores"]  # (N,) tensor

        # Convert to numpy
        if torch.is_tensor(masks):
            masks = masks.cpu().numpy()
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()

        # Convert masks to RLE for efficient transmission
        masks_rle = []
        for i in range(len(masks)):
            mask = masks[i].astype(np.uint8)
            masks_rle.append(mask_to_rle(mask))

        response_data = {
            "masks_rle": masks_rle,
            "boxes": boxes.tolist() if len(boxes) > 0 else [],
            "scores": scores.tolist() if len(scores) > 0 else [],
            "num_detections": len(masks),
            "image_size": list(image.size),  # (width, height)
        }

        logger.info(f"Detected {len(masks)} objects for prompt '{prompt}'")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/sam3/predict_with_masks", methods=["POST"])
def predict_with_masks():
    """
    Handle detection request and return full binary masks as numpy arrays in a zip.

    Expects:
        - image: RGB image file (multipart/form-data)
        - prompt: text prompt for detection (form field or query param)

    Returns:
        ZIP file containing:
        - metadata.json: boxes, scores, num_detections, image_size
        - mask_0.npy, mask_1.npy, ...: binary mask arrays
    """
    global processor

    if processor is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    prompt = request.form.get("prompt") or request.args.get("prompt")

    if not prompt:
        return jsonify({"error": "No text prompt provided"}), 400

    try:
        # Load image
        image = Image.open(image_file.stream).convert("RGB")
        logger.info(f"Received image: {image.size}, prompt: '{prompt}'")

        # Run inference
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        # Convert to numpy
        if torch.is_tensor(masks):
            masks = masks.cpu().numpy()
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()

        # Create zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Save metadata
            metadata = {
                "boxes": boxes.tolist() if len(boxes) > 0 else [],
                "scores": scores.tolist() if len(scores) > 0 else [],
                "num_detections": len(masks),
                "image_size": list(image.size),
            }
            zf.writestr("metadata.json", json.dumps(metadata))

            # Save each mask as numpy array
            for i in range(len(masks)):
                mask = masks[i].astype(np.uint8)
                mask_buffer = io.BytesIO()
                np.save(mask_buffer, mask)
                mask_buffer.seek(0)
                zf.writestr(f"mask_{i}.npy", mask_buffer.read())

        zip_buffer.seek(0)
        logger.info(f"Detected {len(masks)} objects for prompt '{prompt}'")

        return send_file(
            zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name="detection_results.zip",
        )

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": processor is not None})


def main():
    parser = argparse.ArgumentParser(description="SAM3 Detection Service Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5005, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Load model before starting server
    load_model()

    logger.info(f"Starting SAM3 server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
