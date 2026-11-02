import json

import torch

from rtod import coco
from rtod.manifest import write_manifest
from rtod.train import Ema, seed_everything


def test_seed_everything_makes_torch_and_numpy_repeat():
    import numpy as np

    seed_everything(7)
    a = (torch.rand(3), np.random.rand(3))
    seed_everything(7)
    b = (torch.rand(3), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and (a[1] == b[1]).all()


def test_ema_has_a_fixed_decay_and_copies_buffers():
    live = torch.nn.BatchNorm1d(2)
    ema = Ema(live, decay=0.9)
    with torch.no_grad():
        live.weight.fill_(5.0)
        live.running_mean.fill_(3.0)
    ema.update(live)
    # fixed decay: one step moves 10% of the way from the start (1.0) toward 5.0, no warm-up ramp
    assert abs(float(ema.model.weight[0]) - 1.4) < 1e-6
    assert float(ema.model.running_mean[0]) == 3.0  # buffer copied, not averaged
    for _ in range(200):
        ema.update(live)
    assert abs(float(ema.model.weight[0]) - 5.0) < 1e-3


def test_lists_have_the_documented_relationships():
    dev, test = set(coco.read_ids("val_dev")), set(coco.read_ids("val_test"))
    assert not dev & test and len(dev) + len(test) == 4952  # annotated val2017 images
    t10, t40 = coco.read_ids("train10k"), coco.read_ids("train40k")
    assert t40[:10_000] == t10 and len(set(t40)) == 40_000
    assert set(coco.read_ids("val200")) <= dev | test


def test_manifest_records_what_is_needed_to_reproduce(tmp_path):
    path = write_manifest(tmp_path, {"seed": 3, "out": tmp_path}, "train10k")
    m = json.loads(path.read_text())
    assert m["args"]["seed"] == 3 and m["id_list"]["name"] == "train10k"
    assert len(m["id_list"]["sha256"]) == 64 and m["versions"]["torch"]
