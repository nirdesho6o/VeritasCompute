"""Capture explainer_animation.html frames with Playwright and stitch into a GIF with Pillow.

Usage:
    python scripts/render_explainer_gif.py [--output explainer.gif] [--fps 10] [--width 800] [--height 450]

Requirements (install once):
    pip install playwright Pillow
    playwright install chromium
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


def capture_frames(
    html_path: Path,
    width: int = 800,
    height: int = 450,
    fps: int = 10,
    duration_ms: int = 12000,
) -> list[Image.Image]:
    """Launch headless Chromium, capture one loop of the animation as PNG frames."""
    frame_interval_ms = 1000 // fps
    num_frames = (duration_ms * fps) // 1000
    frames: list[Image.Image] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(f"file://{html_path.resolve()}")
        page.wait_for_timeout(500)

        # Inject a way to control animation time
        page.evaluate("""() => {
            window.__frames_captured = 0;
            window.__paused = false;
            const origRAF = window.requestAnimationFrame;
            window._animStart = null;
            window._forcedTime = null;
            window.setAnimTime = (ms) => { window._forcedTime = ms; };
        }""")

        # Override requestAnimationFrame to use forced time
        page.evaluate("""() => {
            const origRAF = window.requestAnimationFrame;
            let callback = null;
            window._rafCallback = null;
            // We'll just call the draw function directly with controlled timestamps
        }""")

        for i in range(num_frames):
            t = i * frame_interval_ms
            # Call draw directly with a controlled timestamp
            # The animation uses (timestamp - start) % LOOP, so we set start=0 and pass t
            page.evaluate(f"""() => {{
                start = 0;
                draw({t});
            }}""")
            png_bytes = page.screenshot(type="png")
            img = Image.open(io.BytesIO(png_bytes))
            frames.append(img.convert("RGBA"))
            if (i + 1) % 20 == 0:
                print(f"  Captured frame {i+1}/{num_frames}", flush=True)

        browser.close()

    print(f"  Captured {len(frames)} frames total", flush=True)
    return frames


def stitch_gif(
    frames: list[Image.Image],
    output: Path,
    fps: int = 10,
    max_size_mb: float = 5.0,
) -> None:
    """Stitch frames into an optimized GIF under max_size_mb."""
    frame_duration_ms = 1000 // fps

    # Convert RGBA to P (palette) mode for smaller GIF
    rgb_frames = []
    for f in frames:
        bg = Image.new("RGBA", f.size, (15, 15, 26, 255))  # #0f0f1a
        bg.paste(f, mask=f)
        rgb_frames.append(bg.convert("RGB"))

    # Quantize to common palette
    quantized = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in rgb_frames]

    quantized[0].save(
        output,
        save_all=True,
        append_images=quantized[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  GIF saved: {output} ({size_mb:.2f} MB)", flush=True)

    if size_mb > max_size_mb:
        print(f"  WARNING: GIF is {size_mb:.1f} MB (target < {max_size_mb} MB)", flush=True)
        print(f"  Try reducing --fps or colors", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Render VeritasCompute explainer GIF")
    parser.add_argument("--output", type=str, default="explainer.gif")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    args = parser.parse_args()

    html_path = Path(__file__).resolve().parent / "explainer_animation.html"
    if not html_path.exists():
        print(f"ERROR: {html_path} not found", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output)
    print(f"Rendering {html_path.name} -> {output}", flush=True)
    print(f"  Resolution: {args.width}x{args.height}, FPS: {args.fps}", flush=True)

    frames = capture_frames(html_path, args.width, args.height, args.fps)
    stitch_gif(frames, output, args.fps)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
