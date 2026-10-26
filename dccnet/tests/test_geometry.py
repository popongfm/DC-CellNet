import math

import numpy as np
import pytest
import torch

from dccnet.datasets.geometry import (
    boxes_to_circles,
    circle_iou,
    circles_to_boxes,
    overlap_levels,
)


def test_box_to_circle_basic():
    c, v = boxes_to_circles(np.array([[10, 10, 30, 50]]), 640, 640)
    assert c.dtype == np.float32 and v.dtype == bool
    np.testing.assert_allclose(c[0], [20, 30, math.sqrt(20 * 40) / 2], atol=1e-5)
    assert c[0, 2] == pytest.approx(14.142, abs=1e-3)
    assert v[0]


def test_edge_validity():
    boxes = np.array([[0, 100, 20, 120], [618, 0, 639, 10], [100, 100, 120, 120]])
    _, v = boxes_to_circles(boxes, 640, 640)
    assert v.tolist() == [False, False, True]
    _, v = boxes_to_circles(np.array([[100, 100, 120, 638]]), 640, 640)
    assert v.tolist() == [False]  # y2 >= img_h - edge_px


def test_circles_to_boxes_roundtrip():
    circles = np.array([[20, 30, 5]], dtype=np.float32)
    np.testing.assert_allclose(circles_to_boxes(circles), [[15, 25, 25, 35]])
    assert circles_to_boxes(circles).dtype == np.float32


def test_overlap_levels():
    xs = [0, 15, 30, 100]
    circles = np.array([[x, 0, 10] for x in xs], dtype=np.float32)
    assert overlap_levels(circles).tolist() == [1, 2, 1, 0]
    five = np.array([[i, 0, 10] for i in range(5)], dtype=np.float32)
    assert overlap_levels(five).tolist() == [3] * 5
    # touching (d == r_i + r_j) is not overlap
    touch = np.array([[0, 0, 10], [20, 0, 10]], dtype=np.float32)
    assert overlap_levels(touch).tolist() == [0, 0]


def test_empty_inputs():
    c, v = boxes_to_circles(np.zeros((0, 4)), 640, 640)
    assert c.shape == (0, 3) and v.shape == (0,)
    assert c.dtype == np.float32 and v.dtype == bool
    assert circles_to_boxes(np.zeros((0, 3))).shape == (0, 4)
    o = overlap_levels(np.zeros((0, 3)))
    assert o.shape == (0,) and np.issubdtype(o.dtype, np.integer)


def _c(*v):
    return torch.tensor(v, dtype=torch.float32)


def test_circle_iou_values():
    assert circle_iou(_c(0, 0, 3), _c(0, 0, 3)).item() == pytest.approx(1.0, abs=1e-6)
    assert circle_iou(_c(0, 0, 1), _c(10, 0, 1)).item() == 0.0
    assert circle_iou(_c(0, 0, 1), _c(0, 0, 2)).item() == pytest.approx(0.25, abs=1e-6)
    # partial overlap vs. brute-force rasterisation
    a, b = _c(0, 0, 5), _c(6, 0, 4)
    g = torch.linspace(-10, 12, 2200)
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    ina = xx**2 + yy**2 <= 25
    inb = (xx - 6) ** 2 + yy**2 <= 16
    ref = (ina & inb).sum() / (ina | inb).sum()
    assert circle_iou(a, b).item() == pytest.approx(ref.item(), abs=2e-3)


def test_circle_iou_broadcast_and_symmetry():
    a = torch.rand(4, 1, 3) * 5 + 1
    b = torch.rand(1, 6, 3) * 5 + 1
    out = circle_iou(a, b)
    assert out.shape == (4, 6)
    torch.testing.assert_close(out, circle_iou(b, a))
    assert ((out >= 0) & (out <= 1 + 1e-6)).all()


@pytest.mark.parametrize(
    "a,b",
    [
        ((0, 0, 3), (0, 0, 3)),  # identical
        ((0, 0, 1), (10, 0, 1)),  # disjoint
        ((0, 0, 1), (0.5, 0, 3)),  # contained
        ((0, 0, 1), (0, 0, 2)),  # concentric
        ((0, 0, 5), (6, 0, 4)),  # partial
        ((0, 0, 2), (4, 0, 2)),  # tangent
    ],
)
def test_circle_iou_finite_grads(a, b):
    ta = _c(*a).requires_grad_()
    tb = _c(*b).requires_grad_()
    out = circle_iou(ta, tb)
    assert torch.isfinite(out)
    out.backward()
    assert torch.isfinite(ta.grad).all() and torch.isfinite(tb.grad).all()


def test_circle_iou_nonpositive_radius_no_nan():
    a = _c(0, 0, 0).requires_grad_()
    b = _c(0, 0, -1).requires_grad_()
    out = circle_iou(a, b)
    assert torch.isfinite(out)
    out.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
