"""Rewrite README.md for the PyPI project page (run in CI before `python -m build`).

PyPI cannot play the GitHub-hosted video, resolve repository-relative links or
render GitHub alerts, so the video becomes the timeline figure and every
relative link points at the tagged commit on GitHub.
"""
import re
import tomllib
from pathlib import Path

REPO = "KyungsuKim42/livesynth"
version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
blob = f"https://github.com/{REPO}/blob/v{version}"
raw = f"https://raw.githubusercontent.com/{REPO}/v{version}"

readme = Path("README.md")
text = readme.read_text()
video = re.compile(r"^https://github\.com/user-attachments/assets/\S+$", re.M)
assert video.search(text), "README video line not found"
text = video.sub(
    f'<p align="center"><a href="https://github.com/{REPO}">'
    f'<img alt="Timeline of a LiveSynth performance: timbre, MIDI and spectrogram" '
    f'src="{raw}/docs/assets/hero_light.png" width="100%"></a></p>', text)
text = re.sub(r"\]\((?!https?://|#)([^)]+)\)", rf"]({blob}/\1)", text)       # markdown links
text = re.sub(r'href="(?!https?://|#)([^"]+)"', rf'href="{blob}/\1"', text)  # HTML links
text = text.replace("](#", f"](https://github.com/{REPO}#")
text = re.sub(r"^> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\n",
              lambda m: f"> **{m.group(1).capitalize()}**\n>\n", text, flags=re.M)
readme.write_text(text)
