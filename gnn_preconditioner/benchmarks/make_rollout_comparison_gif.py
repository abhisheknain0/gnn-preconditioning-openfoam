#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence


def _parse_pair(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Each --input must be LABEL=PATH")
    label, path_str = value.split("=", 1)
    label = label.strip()
    path = Path(path_str).expanduser().resolve()
    if not label:
        raise argparse.ArgumentTypeError("Input label must be non-empty")
    return label, path


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    width = max(1, round(image.width * (height / image.height)))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def _load_gif_frames(path: Path) -> tuple[list[Image.Image], int]:
    img = Image.open(path)
    duration = int(img.info.get("duration", 100))
    frames = []
    for frame in ImageSequence.Iterator(img):
        frames.append(frame.convert("RGB"))
    if not frames:
        raise ValueError(f"No frames found in GIF: {path}")
    return frames, duration


def _build_canvas_row(images: list[Image.Image], labels: list[str], gutter: int, title_height: int) -> Image.Image:
    target_height = min(image.height for image in images)
    resized = [_resize_to_height(image, target_height) for image in images]
    width = sum(image.width for image in resized) + gutter * (len(resized) - 1)
    canvas = Image.new("RGB", (width, target_height + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    x = 0
    for image, label in zip(resized, labels):
        draw.rectangle((x, 0, x + image.width, title_height), fill=(245, 245, 245))
        draw.text((x + 8, 8), label, fill="black", font=font)
        canvas.paste(image, (x, title_height))
        x += image.width + gutter
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=_parse_pair, help="LABEL=PATH to a GIF")
    parser.add_argument("--output-gif", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--gutter", type=int, default=16)
    parser.add_argument("--title-height", type=int, default=28)
    args = parser.parse_args()

    labels = [label for label, _ in args.input]
    gif_paths = [path for _, path in args.input]
    for path in gif_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    gifs = [_load_gif_frames(path) for path in gif_paths]
    min_frames = min(len(frames) for frames, _ in gifs)
    duration = min(gif_duration for _, gif_duration in gifs)

    combined_frames: list[Image.Image] = []
    for frame_idx in range(min_frames):
        row = _build_canvas_row(
            [frames[frame_idx] for frames, _ in gifs],
            labels=labels,
            gutter=args.gutter,
            title_height=args.title_height,
        )
        combined_frames.append(row)

    args.output_gif.parent.mkdir(parents=True, exist_ok=True)
    combined_frames[0].save(
        args.output_gif,
        save_all=True,
        append_images=combined_frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )

    if args.output_png is not None:
        args.output_png.parent.mkdir(parents=True, exist_ok=True)
        combined_frames[0].save(args.output_png)

    print(f"saved_gif={args.output_gif.resolve()}")
    if args.output_png is not None:
        print(f"saved_png={args.output_png.resolve()}")
    print(f"frames={min_frames}")
    print(f"duration_ms={duration}")


if __name__ == "__main__":
    main()
