#!/usr/bin/env bash
# Apply the yolox-traffic demo to a checkout of the indah repo (idempotent).
#   gallery/pr/apply.sh /path/to/indah
# Copies the example, its assets, the thumbnail and a test; adds the catalog entry, the build-script lines
# and the CHANGELOG line. Does not commit, push or deploy anything.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
indah="${1:?usage: apply.sh /path/to/indah-checkout}"
[ -f "$indah/deploy/fly/build_gallery.sh" ] || { echo "not an indah checkout: $indah" >&2; exit 2; }

cp "$here/examples/yolox_traffic.py" "$indah/examples/yolox_traffic.py"
rm -rf "$indah/examples/yolox_traffic" && cp -r "$here/examples/yolox_traffic" "$indah/examples/yolox_traffic"
cp "$here/pr/yolox-traffic.png" "$indah/deploy/fly/thumbnails/yolox-traffic.png"
cp "$here/pr/yolox-traffic.ipynb" "$indah/examples/colab/yolox-traffic.ipynb"  # from `make colab`; regenerate if the example changes

python3 - "$here" "$indah" <<'PY'
import re, sys
from pathlib import Path

here, indah = Path(sys.argv[1]), Path(sys.argv[2])

# the test: the repo's own, pointed at the indah layout, without the part that needs the rtod package
t = (here.parent / "tests/gallery/test_yolox_traffic.py").read_text()
t = t.split("# ---- honesty: replay equals")[0] + t[t.index("# ---- mounting:"):]
t = t.replace('ROOT = Path(__file__).parents[2]\nEXAMPLE = ROOT / "gallery/examples/yolox_traffic.py"',
              'ROOT = Path(__file__).resolve().parents[2]\nEXAMPLE = ROOT / "examples" / "yolox_traffic.py"')
t = t.replace('ROOT / "gallery/examples/yolox_traffic/replay.json"', 'ROOT / "examples/yolox_traffic/replay.json"')
assert "gallery/examples" not in t
(indah / "tests/integration/test_yolox_traffic_example.py").write_text(t)

# catalog
cat = indah / "deploy/fly/gallery.toml"
text = cat.read_text()
if 'slug = "yolox-traffic"' not in text:
    cat.write_text(text.rstrip("\n") + "\n" + (here / "pr/catalog_entry.toml").read_text())

# build script: module list, asset folder, model prefetch
sh = indah / "deploy/fly/build_gallery.sh"
s = sh.read_text()
if "yolox_traffic" not in s:
    s = re.sub(r"(examples=\([^)]*?)\)", r"\1 yolox_traffic)", s, count=1)
    s = s.replace('cp -r "$repo/examples/audio_samples" "$out/"\n',
                  'cp -r "$repo/examples/audio_samples" "$out/"\n'
                  '# yolox_traffic.py reads its clip frames, recorded detections and ATTRIBUTION.txt from yolox_traffic/ beside it.\n'
                  'cp -r "$repo/examples/yolox_traffic" "$out/"\n', 1)
    s = s.replace("d('onnxmodelzoo/ssd_mobilenet_v1_12', 'ssd_mobilenet_v1_12.onnx')\"",
                  "d('onnxmodelzoo/ssd_mobilenet_v1_12', 'ssd_mobilenet_v1_12.onnx'); "
                  "d('jianlee/yolox-tiny-quantized', 'yolox_tiny_ptq_int8_calib32.onnx')\"", 1)
    assert s.count("yolox") >= 3, "build_gallery.sh no longer has the lines this patch edits"
    sh.write_text(s)

# the Colab and Space demo list (deploy/spaces/demos.tsv), which indah's tests keep in step with the catalog
tsv = indah / "deploy/spaces/demos.tsv"
row = "yolox-traffic\tyolox_traffic.py\tReal-time detection on a traffic clip\t🚦\tonnxruntime pillow huggingface_hub numpy\n"
if "yolox-traffic" not in tsv.read_text():
    tsv.write_text(tsv.read_text().rstrip("\n") + "\n" + row)

# changelog
log = indah / "CHANGELOG.md"
c = log.read_text()
line = "- A gallery demo, `examples/yolox_traffic.py`: a quantized YOLOX-Tiny on a bundled traffic clip, as a replay of recorded detections by default and a live run (one visitor at a time, up to 30 s) on request."
if "yolox_traffic" not in c:
    c = c.replace("## [Unreleased]\n", "## [Unreleased]\n\n### Added\n\n" + line + "\n", 1)
    log.write_text(c)
print("applied")
PY
