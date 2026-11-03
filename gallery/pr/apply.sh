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
t += """

def test_the_build_script_prefetches_the_pinned_model_revision():
    sh = (ROOT / "deploy/fly/build_gallery.sh").read_text()
    assert f"revision='{ex.MODEL_REVISION}'" in sh  # build and run time serve the same model
"""
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

# the model prefetch is pinned to the commit the example expects (also fixes an earlier unpinned apply)
s = sh.read_text()
pin = re.search(r'MODEL_REVISION = "([0-9a-f]{40})"', (here / "examples/yolox_traffic.py").read_text()).group(1)
old = "d('jianlee/yolox-tiny-quantized', 'yolox_tiny_ptq_int8_calib32.onnx'"
if f"revision='{pin}'" not in s:
    s = re.sub(re.escape(old) + r"(, revision='[0-9a-f]{40}')?\)", old + f", revision='{pin}')", s, count=1)
    assert f"revision='{pin}'" in s
    sh.write_text(s)

# the gallery rule, in the doc that states it
doc = indah / "docs/GALLERY-APPS-NEXT.md"
d = doc.read_text()
was = "each\napp is model-free, needs no network at run time and has bounded work per session."
now = ("each\napp has bounded work per session and needs no network at run time. A demo may run a model if it is small, "
       "fetched at image build time (the Hugging Face cache is baked into the image and pinned to a commit), "
       "bounded by a global slot, and has a recorded fallback; object_detection, upload_classify and yolox-traffic do.")
if was in d:
    doc.write_text(d.replace(was, now, 1))
else:
    assert "has a recorded fallback" in d, "the gallery rule line moved; update apply.sh"

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
