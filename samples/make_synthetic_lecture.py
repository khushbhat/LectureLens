"""Build a small synthetic slide lecture (offline) for development and tests.

Slides are rendered with Pillow; narration uses the OS text-to-speech
(Windows SAPI via PowerShell, `say` on macOS, `espeak` on Linux). Some slide
bullets are deliberately NOT spoken, so slide-only concepts exist (RQ1).

    python samples/make_synthetic_lecture.py  ->  samples/synthetic_gd.mp4
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SLIDES = [
    {"title": "Gradient Descent for Machine Learning",
     "bullets": ["Lecture 4: Optimization", "Introduction to Machine Learning"],
     "speech": "Welcome back. Today we study gradient descent, the workhorse optimization algorithm behind most of machine learning. "
               "We will see what it does, why it works, and how to choose its settings."},
    {"title": "The Optimization Problem",
     "bullets": ["Minimize loss L(w) over parameters w", "Mean squared error: L(w) = 1/n * sum (y_i - w x_i)^2",
                 "Convex vs non-convex objectives"],
     "speech": "Training a model means minimizing a loss function over its parameters. "
               "A common choice is the mean squared error between predictions and targets. "
               "If the loss is convex, every local minimum is also a global minimum."},
    {"title": "The Update Rule",
     "bullets": ["w <- w - eta * grad L(w)", "eta = learning rate", "Gradient points uphill: step the opposite way"],
     "speech": "The idea is simple. Compute the gradient of the loss, which points in the direction of steepest increase, "
               "and take a small step in the opposite direction. The size of the step is controlled by the learning rate."},
    {"title": "Choosing the Learning Rate",
     "bullets": ["Too large: divergence / oscillation", "Too small: slow convergence", "Learning rate schedules: step decay, cosine annealing",
                 "Adaptive methods: Adam, RMSProp"],
     "speech": "Choosing the learning rate matters a lot. If it is too large, the iterates oscillate or even diverge. "
               "If it is too small, training converges very slowly. In practice we often decrease it over time."},
    {"title": "Stochastic Gradient Descent",
     "bullets": ["Use a mini-batch instead of the full dataset", "Noisy but cheap gradient estimate", "Batch size 32 to 256 is typical",
                 "Epoch = one pass over the training data"],
     "speech": "Computing the full gradient on a large dataset is expensive. Stochastic gradient descent estimates it from a small mini-batch "
               "of examples. The estimate is noisy, but each step is much cheaper, so overall training is faster."},
    {"title": "Summary",
     "bullets": ["Gradient descent follows the negative gradient", "Learning rate controls step size", "SGD uses mini-batches",
                 "Momentum accelerates convergence"],
     "speech": "To summarize: gradient descent repeatedly steps along the negative gradient, the learning rate controls the step size, "
               "and stochastic gradient descent makes each step cheap by using mini-batches. Next time we will look at regularization."},
]


def _font(size: int, bold: bool = False):
    cands = ["C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
             "/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for c in cands:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def render_slide(slide: dict, idx: int, out: Path) -> None:
    img = Image.new("RGB", (1280, 720), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1280, 110], fill=(25, 60, 120))
    d.text((50, 30), slide["title"], font=_font(52, True), fill="white")
    y = 170
    for b in slide["bullets"]:
        d.text((70, y), "• " + b, font=_font(36), fill=(20, 20, 20))
        y += 80
    d.text((50, 670), "CS 101 · Intro to ML", font=_font(22), fill=(120, 120, 120))
    d.text((1180, 670), str(idx + 1), font=_font(22), fill=(120, 120, 120))
    img.save(out)


def tts(text: str, out_wav: Path) -> None:
    system = platform.system()
    if system == "Windows":
        ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              f"$s.Rate = 0; $s.SetOutputToWaveFile('{out_wav}'); $s.Speak([Console]::In.ReadToEnd()); $s.Dispose()")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], input=text, text=True, check=True)
    elif system == "Darwin":
        subprocess.run(["say", "-o", str(out_wav), "--data-format=LEI16@16000", text], check=True)
    elif shutil.which("espeak"):
        subprocess.run(["espeak", "-w", str(out_wav), text], check=True)
    else:
        sys.exit("No TTS engine found (needs Windows SAPI, macOS `say`, or espeak).")


def main(out: str = "samples/synthetic_gd.mp4") -> None:
    out_p = Path(out)
    tmp = Path(tempfile.mkdtemp())
    parts = []
    for i, s in enumerate(SLIDES):
        png, wav, mp4 = tmp / f"s{i}.png", tmp / f"s{i}.wav", tmp / f"s{i}.mp4"
        render_slide(s, i, png)
        tts(s["speech"], wav)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(png), "-i", str(wav),
                        "-af", "apad=pad_dur=1.5", "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                        "-r", "10", "-c:a", "aac", "-ar", "16000", "-shortest", str(mp4)], check=True)
        parts.append(mp4)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out_p)], check=True)
    print("wrote", out_p)


if __name__ == "__main__":
    main(*sys.argv[1:])
