from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from dccnet.datasets.torch_dataset import CellDataset, collate_fn, unletterbox
from dccnet.types import Sample

NO_AUG = {"hflip": False, "vflip": False, "rot90": False, "scale": 0.0, "translate": 0.0, "stain": False}


def _make_image(path: Path, w: int, h: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def _sample(tmp_path, name="a.png", w=640, h=480, boxes=((100, 100, 140, 140),), labels=(0,)):
    p = tmp_path / name
    _make_image(p, w, h)
    return Sample(
        image_path=p,
        source="t",
        width=w,
        height=h,
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        labels=np.asarray(labels, dtype=np.int64).reshape(-1),
    )


class FakeRng:
    """Deterministic stand-in: no flips, fixed rot90 k, uniform() returns its upper bound."""

    def __init__(self, k=0, flip=False):
        self.k = k
        self.flip = flip

    def random(self):
        return 0.0 if self.flip else 0.99

    def integers(self, low, high=None):
        return self.k

    def uniform(self, low=0.0, high=1.0, size=None):
        return high if size is None else np.full(size, high)


def _active_circles(t):
    m = t["mask"].numpy()
    return t["circle"].numpy()[m], t["cls"].numpy()[m]


def test_letterbox_center_and_unletterbox_exact(tmp_path):
    ds = CellDataset([_sample(tmp_path)])
    img, t, meta = ds[0]
    assert img.shape == (3, 640, 640) and img.dtype == torch.float32
    assert 0.0 <= img.min().item() and img.max().item() <= 1.0
    assert meta["scale"] == 1.0 and meta["pad"] == (0, 80)
    assert meta["orig_size"] == (640, 480) and meta["image"] == str(tmp_path / "a.png")
    circles, _ = _active_circles(t)
    np.testing.assert_allclose(circles, [[120.0, 200.0, 20.0]], atol=1e-5)
    back = unletterbox(circles, meta)
    assert back.dtype == np.float32
    np.testing.assert_array_equal(back, np.array([[120.0, 120.0, 20.0]], dtype=np.float32))
    # padding is 114 gray, image region is the RGB-converted resized image
    assert torch.allclose(img[:, :80], torch.full((3, 80, 640), 114 / 255), atol=1e-6)
    raw = cv2.cvtColor(cv2.imread(str(tmp_path / "a.png")), cv2.COLOR_BGR2RGB)
    np.testing.assert_allclose(img[:, 80:560].permute(1, 2, 0).numpy(), raw / 255.0, atol=1e-6)


def test_unletterbox_scaled_image_and_empty(tmp_path):
    ds = CellDataset([_sample(tmp_path, w=320, h=320, boxes=((100, 100, 140, 140),))])
    _, t, meta = ds[0]
    assert meta["scale"] == 2.0 and meta["pad"] == (0, 0)
    circles, _ = _active_circles(t)
    np.testing.assert_allclose(circles, [[240, 240, 40]], atol=1e-4)
    np.testing.assert_allclose(unletterbox(circles, meta), [[120, 120, 20]], atol=1e-4)
    out = unletterbox(np.zeros((0, 3), dtype=np.float32), meta)
    assert out.shape == (0, 3) and out.dtype == np.float32


def test_radius_valid_computed_on_original_image(tmp_path):
    # box touching the original right border is invalid; interior box valid
    s = _sample(tmp_path, boxes=((100, 100, 140, 140), (600, 100, 640, 140)), labels=(0, 1))
    _, t, _ = CellDataset([s])[0]
    assert t["r_valid"][:2].tolist() == [True, False]


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_rot90_radii_and_centers(tmp_path, k, monkeypatch):
    s = _sample(tmp_path, w=640, h=640, boxes=((100, 200, 140, 240), (300, 50, 330, 80)), labels=(0, 1))
    ds = CellDataset([s], aug={**NO_AUG, "rot90": True})
    monkeypatch.setattr(CellDataset, "_make_rng", lambda self, index: FakeRng(k))
    img, t, _ = ds[0]
    base_img, base_t, _ = CellDataset([s])[0]
    c0, _ = _active_circles(base_t)
    c1, l1 = _active_circles(t)
    exp = c0.copy()
    for _ in range(k):  # CCW rotation of a square image: (x, y) -> (y, S - x)
        exp = np.stack([exp[:, 1], 640 - exp[:, 0], exp[:, 2]], axis=1)
    np.testing.assert_allclose(c1, exp, atol=1e-4)
    np.testing.assert_allclose(c1[:, 2], c0[:, 2], atol=1e-5)
    assert l1.tolist() == [0, 1]
    np.testing.assert_allclose(img.numpy(), np.rot90(base_img.numpy(), k, axes=(1, 2)), atol=1e-6)


def test_no_aug_deterministic(tmp_path):
    ds = CellDataset([_sample(tmp_path)])
    a, ta, ma = ds[0]
    b, tb, mb = ds[0]
    assert torch.equal(a, b) and ma == mb
    for k in ta:
        assert torch.equal(ta[k], tb[k])


def test_aug_reproducible_given_torch_seed_and_varies_with_index(tmp_path):
    aug = {**NO_AUG, "hflip": True, "vflip": True, "rot90": True, "scale": 0.2, "translate": 0.1, "stain": True}
    ds = CellDataset([_sample(tmp_path, "a.png"), _sample(tmp_path, "b.png")], aug=aug)
    torch.manual_seed(7)
    a, _, _ = ds[0]
    torch.manual_seed(7)
    b, _, _ = ds[0]
    assert torch.equal(a, b)
    torch.manual_seed(8)
    c, _, _ = ds[0]
    assert not torch.equal(a, c)


def test_stain_off_leaves_pixels_unchanged(tmp_path):
    s = _sample(tmp_path)
    base, _, _ = CellDataset([s])[0]
    off, _, _ = CellDataset([s], aug=NO_AUG)[0]
    assert torch.equal(base, off)
    on, _, _ = CellDataset([s], aug={**NO_AUG, "stain": True})[0]
    assert not torch.equal(base, on)


def test_scale_affine_transforms_circles(tmp_path, monkeypatch):
    s = _sample(tmp_path, w=640, h=640, boxes=((300, 300, 340, 340),))
    ds = CellDataset([s], aug={**NO_AUG, "scale": 0.2})
    monkeypatch.setattr(CellDataset, "_make_rng", lambda self, index: FakeRng())
    _, t, _ = ds[0]
    c, _ = _active_circles(t)
    # center (320,320) is the image center -> unchanged; radius * 1.2
    np.testing.assert_allclose(c, [[320, 320, 24]], atol=1e-4)


def test_large_translate_drops_out_of_map_cells(tmp_path):
    rng = np.random.default_rng(0)
    xs = rng.uniform(20, 620, 60)
    ys = rng.uniform(20, 620, 60)
    boxes = [(x - 10, y - 10, x + 10, y + 10) for x, y in zip(xs, ys)]
    s = _sample(tmp_path, w=640, h=640, boxes=boxes, labels=[0] * 60)
    ds = CellDataset([s], aug={**NO_AUG, "translate": 0.5})
    dropped = False
    for seed in range(10):
        torch.manual_seed(seed)
        _, t, _ = ds[0]
        c, _ = _active_circles(t)
        n = int(t["mask"].sum())
        assert n == len(c)
        dropped |= n < 60
        assert ((c[:, 0] >= 0) & (c[:, 0] < 640) & (c[:, 1] >= 0) & (c[:, 1] < 640)).all()
        assert (t["ind"][t["mask"]] < 160 * 160).all()
        # r_valid False exactly for circles crossing the image edge
        rv = t["r_valid"].numpy()[t["mask"].numpy()]
        crossing = (c[:, 0] - c[:, 2] < 0) | (c[:, 1] - c[:, 2] < 0) | (c[:, 0] + c[:, 2] > 640) | (c[:, 1] + c[:, 2] > 640)
        assert not rv[crossing].any()
        assert rv[~crossing].all()
        # density mass equals number of kept cells
        assert t["density"].sum().item() == pytest.approx(n, rel=1e-4)
    assert dropped


def test_zero_boxes(tmp_path):
    s = _sample(tmp_path, boxes=(), labels=())
    for aug in (None, {**NO_AUG, "hflip": True, "scale": 0.2, "translate": 0.1, "stain": True}):
        img, t, meta = CellDataset([s], aug=aug)[0]
        assert img.shape == (3, 640, 640)
        assert int(t["mask"].sum()) == 0 and t["heatmap"].sum().item() == 0


def test_collate_stacks(tmp_path):
    ds = CellDataset([_sample(tmp_path, "a.png"), _sample(tmp_path, "b.png", w=320, h=200, boxes=(), labels=())])
    imgs, tg, metas = collate_fn([ds[0], ds[1]])
    assert imgs.shape == (2, 3, 640, 640)
    assert tg["heatmap"].shape == (2, 3, 160, 160) and tg["density"].shape == (2, 3, 160, 160)
    assert tg["ind"].shape == (2, 512) and tg["ind"].dtype == torch.int64
    assert tg["offset"].shape == (2, 512, 2) and tg["circle"].shape == (2, 512, 3)
    assert tg["mask"].dtype == torch.bool and tg["mask"].sum(dim=1).tolist() == [1, 0]
    assert isinstance(metas, list) and len(metas) == 2 and metas[1]["orig_size"] == (320, 200)
    # works through a real DataLoader
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate_fn)))
    assert batch[0].shape == (2, 3, 640, 640)


def test_epoch_changes_augmentation(tmp_path):
    aug = {**NO_AUG, "hflip": True, "vflip": True, "rot90": True, "scale": 0.2, "translate": 0.1}
    ds = CellDataset([_sample(tmp_path)], aug=aug)
    ds.set_epoch(0)
    a, _, _ = ds[0]
    a2, _, _ = ds[0]
    assert torch.equal(a, a2)
    ds.set_epoch(1)
    b, _, _ = ds[0]
    assert not torch.equal(a, b)
    ds.set_epoch(0)
    c, _, _ = ds[0]
    assert torch.equal(a, c)


def test_zero_area_box_is_dropped(tmp_path):
    s = _sample(tmp_path, boxes=((100, 100, 140, 140), (300, 300, 300, 300), (200, 200, 200.4, 200.4)), labels=(0, 1, 2))
    for aug in (None, {**NO_AUG, "scale": 0.1}):
        _, t, _ = CellDataset([s], aug=aug)[0]
        c, l = _active_circles(t)
        assert len(c) == 1 and l.tolist() == [0]


def test_warped_disc_centroid_matches_circle(tmp_path, monkeypatch):
    S = 640
    bg = 114
    cx, cy, r = 200.0, 300.0, 30.0
    yy, xx = np.mgrid[0:S, 0:S]
    disc = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r**2
    img = np.full((S, S, 3), bg, dtype=np.uint8)
    img[disc] = 255
    p = tmp_path / "disc.png"
    assert cv2.imwrite(str(p), img)
    sample = Sample(p, "t", S, S, np.array([[cx - r, cy - r, cx + r, cy + r]], np.float32), np.array([0], np.int64))
    ds = CellDataset([sample], aug={**NO_AUG, "hflip": True, "vflip": True, "scale": 0.1, "translate": 0.1})
    monkeypatch.setattr(CellDataset, "_make_rng", lambda self, index: FakeRng(0, flip=True))
    out, t, _ = ds[0]
    (circle,), _ = _active_circles(t)
    # hflip+vflip -> (440, 340); scale 1.1 about 320 -> (452, 342); translate +64 -> (516, 406); r = 33
    np.testing.assert_allclose(circle, [516.0, 406.0, 33.0], atol=1e-3)
    gray = out[0].numpy() * 255.0 - bg
    w = np.clip(gray, 0, None)
    ys, xs = np.mgrid[0:S, 0:S]
    mx = (w * (xs + 0.5)).sum() / w.sum()
    my = (w * (ys + 0.5)).sum() / w.sum()
    assert abs(mx - circle[0]) < 0.25 and abs(my - circle[1]) < 0.25


def test_heatmap_sigma_and_stride_reach_targets(tmp_path):
    # two same-class cells 12 px apart in a 640x640 image (no letterbox scaling)
    s = _sample(tmp_path, w=640, h=640, boxes=((176, 76, 224, 124), (188, 76, 236, 124)), labels=(0, 0))
    _, t_rad, _ = CellDataset([s])[0]
    _, t_ada, _ = CellDataset([s], heatmap_sigma="adaptive")[0]
    _, t_s2, _ = CellDataset([s], stride=2, heatmap_sigma="adaptive")[0]
    assert t_rad["heatmap"][0, 25, 49].item() == pytest.approx(np.exp(-1 / (2 * 2.0**2)), rel=1e-5)
    assert t_ada["heatmap"][0, 25, 49].item() == pytest.approx(np.exp(-1 / (2 * 0.75**2)), rel=1e-5)
    assert t_s2["heatmap"].shape == (3, 320, 320)
    with pytest.raises(ValueError, match="heatmap_sigma"):
        CellDataset([s], heatmap_sigma="bogus")
