"""
Stage-by-stage tests for a from-scratch CNN framework:
- Numeric gradient checks for Linear, Conv2D, MaxPool2D, CrossEntropy
- Forward-shape checks
- End-to-end minimal CNN sanity tests
- Tiny training sanity: loss should decrease

Usage:
  python -u tests_scratch.py
(or run cells in Colab)
"""

import math
import random
import torch
import torch.nn.functional as F

# ============
# Import your from-scratch layers
# ============
from cnn_from_scratch import (
    Linear, 
    ReLU, 
    Flatten, 
    Conv2D, 
    MaxPool2D, 
    CrossEntropy, 
    Sequential
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
random.seed(42)

# -------------
# Tuning knobs
# -------------
# Use None to compute full numeric gradients (slower but simple).
# Use an int (e.g., 50) to sample that many entries and compare only those.
MAX_CHECKS_LINEAR = None      # full check for small Linear tensors
MAX_CHECKS_RELU   = None
MAX_CHECKS_CE     = None
MAX_CHECKS_CONV   = 60        # keep conv checks sampled to stay fast
MAX_CHECKS_POOL   = 30

EPS_DEFAULT = 1e-3            # central-diff epsilon; OK for float32
TOL_LINEAR  = 5e-3
TOL_RELU    = 5e-3
TOL_CE      = 5e-3
TOL_CONV    = 1.5e-2
TOL_POOL    = 5e-3

# -------------------------
# Utilities
# -------------------------
def zeros_like(x):
    g = torch.zeros_like(x, device=x.device)
    g.requires_grad_(False)
    return g

def params_of(seq):
    ps = []
    for layer in getattr(seq, 'layers', []):
        for nm in ('W', 'w', 'b', 'bias'):
            if hasattr(layer, nm):
                t = getattr(layer, nm)
                if t is not None:
                    ps.append(t)
    return ps

@torch.no_grad()
def numerical_grad_tensor(f, x, eps=EPS_DEFAULT, max_checks=None, seed=0, return_indices=False):
    """
    Central-difference numeric gradient of scalar f wrt tensor x.
    If max_checks is not None, computes gradient only on a random subset of indices.
    Returns a dense tensor (zeros on un-checked coords). If return_indices=True,
    also returns a 1D LongTensor of the sampled flat indices.
    """
    rng_state = random.getstate()
    random.seed(seed)

    x_flat = x.view(-1)
    grad   = torch.zeros_like(x_flat)

    idxs = list(range(x_flat.numel()))
    if max_checks is not None and max_checks < len(idxs):
        random.shuffle(idxs)
        idxs = idxs[:max_checks]

    for i in idxs:
        orig = x_flat[i].item()
        x_flat[i] = orig + eps
        fp = f()
        x_flat[i] = orig - eps
        fm = f()
        x_flat[i] = orig
        grad[i] = (fp - fm) / (2*eps)

    random.setstate(rng_state)
    grad = grad.view_as(x)
    if return_indices:
        return grad, torch.tensor(idxs, device=x.device, dtype=torch.long)
    return grad

def masked_max_abs_diff(analytic, numeric, idxs=None):
    """
    Max |analytic - numeric|. If idxs is provided, compare only those flat indices.
    """
    a = analytic.view(-1)
    n = numeric.view(-1)
    if idxs is None:
        return (a - n).abs().max().item()
    return (a[idxs] - n[idxs]).abs().max().item()

# -------------------------
# Tests: Linear, ReLU, CE
# -------------------------
def test_linear_forward_backward():
    print("\n[Linear] forward/backward numeric grad...")
    N, Din, Dout = 4, 5, 3
    x = torch.randn(N, Din, device=DEVICE)
    layer = Linear(Din, Dout)

    y = layer(x)
    assert y.shape == (N, Dout), "Linear forward shape mismatch"

    y.g = torch.randn_like(y)
    layer.backward()

    # dL/dx
    def f_x():
        y2 = layer(x)
        return (y2 * y.g).sum().item()

    gx_num, ix = numerical_grad_tensor(
        f_x, x, eps=EPS_DEFAULT, max_checks=MAX_CHECKS_LINEAR, return_indices=True
    )
    max_diff_x = masked_max_abs_diff(x.g, gx_num, idxs=ix if MAX_CHECKS_LINEAR else None)
    print("  max |dL/dx (num - analytic)|:", max_diff_x)
    assert max_diff_x < TOL_LINEAR, "Linear backward wrt input seems off"

    # dL/dW and dL/db
    def f_wb():
        y2 = x @ layer.w + layer.b
        return (y2 * y.g).sum().item()

    gw_num, iw = numerical_grad_tensor(
        f_wb, layer.w, eps=EPS_DEFAULT, max_checks=MAX_CHECKS_LINEAR, return_indices=True
    )
    gb_num, ib = numerical_grad_tensor(
        f_wb, layer.b, eps=EPS_DEFAULT, max_checks=MAX_CHECKS_LINEAR, return_indices=True
    )

    max_diff_w = masked_max_abs_diff(layer.w.g, gw_num, idxs=iw if MAX_CHECKS_LINEAR else None)
    max_diff_b = masked_max_abs_diff(layer.b.g, gb_num, idxs=ib if MAX_CHECKS_LINEAR else None)
    print("  max |dL/dW (num - analytic)|:", max_diff_w)
    print("  max |dL/db (num - analytic)|:", max_diff_b)
    assert max_diff_w < TOL_LINEAR, "Linear backward wrt W seems off"
    assert max_diff_b < TOL_LINEAR, "Linear backward wrt b seems off"
    print("  PASS")

def test_relu_forward_backward():
    print("\n[ReLU] forward/backward numeric grad...")
    x = torch.randn(3, 4, device=DEVICE)
    relu = ReLU()
    y = relu(x)
    assert y.shape == x.shape

    y.g = torch.randn_like(y)
    relu.backward()

    def f_x():
        return (relu(x) * y.g).sum().item()

    gx_num, ix = numerical_grad_tensor(
        f_x, x, eps=EPS_DEFAULT, max_checks=MAX_CHECKS_RELU, return_indices=True
    )
    max_diff = masked_max_abs_diff(x.g, gx_num, idxs=ix if MAX_CHECKS_RELU else None)
    print("  max |dL/dx| diff:", max_diff)
    assert max_diff < TOL_RELU, "ReLU backward seems off"
    print("  PASS")

def test_cross_entropy_logits():
    print("\n[CrossEntropy] forward/backward numeric grad wrt logits...")
    N, C = 5, 7
    logits = torch.randn(N, C, device=DEVICE)
    targets = torch.randint(0, C, (N,), device=DEVICE)
    ce = CrossEntropy()
    loss = ce(logits, targets)  # scalar tensor

    loss.g = torch.tensor(1.0, device=DEVICE)   # seed upstream grad for analytic
    ce.backward()

    def f_logits():
        return ce(logits, targets).item()

    g_num, il = numerical_grad_tensor(
        f_logits, logits, eps=EPS_DEFAULT, max_checks=MAX_CHECKS_CE, return_indices=True
    )
    diff = masked_max_abs_diff(logits.g, g_num, idxs=il if MAX_CHECKS_CE else None)
    print("  max |dL/dlogits| diff:", diff)
    assert diff < TOL_CE, "CrossEntropy gradient wrt logits seems off"
    print("  PASS")

# -------------------------
# Tests: Conv2D
# -------------------------
def test_conv2d_forward_shapes():
    print("\n[Conv2D] forward shape check...")
    N, Cin, H, W = 2, 1, 7, 7
    Cout, k, stride, pad = 2, 3, 2, 1
    x = torch.randn(N, Cin, H, W, device=DEVICE)
    conv = Conv2D(Cin, Cout, k, stride=stride, padding=pad)
    y = conv(x)
    H_out = (H + 2*pad - k)//stride + 1
    W_out = (W + 2*pad - k)//stride + 1
    assert y.shape == (N, Cout, H_out, W_out), "Conv2D forward shape mismatch"
    print("  PASS")

def test_conv2d_backward_numeric_small():
    print("\n[Conv2D] backward numeric grads (sampled)...")
    N, Cin, H, W = 2, 1, 7, 7
    Cout, k, stride, pad = 2, 3, 2, 1

    x = torch.randn(N, Cin, H, W, device=DEVICE)
    conv = Conv2D(Cin, Cout, k, stride=stride, padding=pad)
    y = conv(x)
    y.g = torch.randn_like(y)
    conv.backward()

    # dL/dx
    def f_x():
        y2 = conv(x)
        return (y2 * y.g).sum().item()

    gx_num, ix = numerical_grad_tensor(
        f_x, x, eps=EPS_DEFAULT, max_checks=MAX_CHECKS_CONV, return_indices=True
    )
    max_diff_x = masked_max_abs_diff(x.g, gx_num, idxs=ix if MAX_CHECKS_CONV else None)
    print("  max |dL/dx| diff:", max_diff_x)
    assert max_diff_x < TOL_CONV, "Conv2D input grad seems off (tolerance relaxed for speed)"

    # dL/dW
    def f_w():
        y2 = conv(x)
        return (y2 * y.g).sum().item()

    gw_num, iw = numerical_grad_tensor(
        f_w, conv.W, eps=EPS_DEFAULT, max_checks=MAX_CHECKS_CONV, return_indices=True
    )
    max_diff_w = masked_max_abs_diff(conv.W.g, gw_num, idxs=iw if MAX_CHECKS_CONV else None)
    print("  max |dL/dW| diff:", max_diff_w)
    assert max_diff_w < TOL_CONV, "Conv2D weight grad seems off (tolerance relaxed for speed)"
    print("  PASS")

# -------------------------
# Tests: MaxPool2D
# -------------------------
def test_maxpool2d_forward_compare_pytorch():
    print("\n[MaxPool2D] forward vs F.max_pool2d...")
    pool = MaxPool2D(2,2)
    x = torch.randn(2, 3, 6, 6, device=DEVICE)
    y = pool(x)
    y_ref = F.max_pool2d(x, kernel_size=2, stride=2)
    diff = (y - y_ref).abs().max().item()
    print("  max |y - y_ref|:", diff)
    assert y.shape == y_ref.shape
    assert diff < 1e-6, "MaxPool2D forward mismatch vs reference"
    print("  PASS")

def test_maxpool2d_backward_numeric():
    print("\n[MaxPool2D] backward numeric grads...")
    pool = MaxPool2D(2,2)
    x = torch.randn(1, 1, 4, 4, device=DEVICE)
    y = pool(x)
    y.g = torch.randn_like(y)
    pool.backward()

    def f_x():
        y2 = pool(x)
        return (y2 * y.g).sum().item()

    gx_num, ix = numerical_grad_tensor(
        f_x, x, eps=EPS_DEFAULT, max_checks=MAX_CHECKS_POOL, return_indices=True
    )
    max_diff = masked_max_abs_diff(x.g, gx_num, idxs=ix if MAX_CHECKS_POOL else None)
    print("  max |dL/dx| diff:", max_diff)
    assert max_diff < TOL_POOL, "MaxPool2D backward seems off"
    print("  PASS")

# -------------------------
# E2E sanity tests
# -------------------------
def build_small_cnn():
    # Conv(1->4,k=3,p=1) -> ReLU -> Pool2
    # Conv(4->8,k=3,p=1) -> ReLU -> Pool2
    # Flatten -> Linear(8*7*7->10)
    return Sequential(
        Conv2D(1, 4, kernel_size=3, stride=1, padding=1),
        ReLU(),
        MaxPool2D(2,2),
        Conv2D(4, 8, kernel_size=3, stride=1, padding=1),
        ReLU(),
        MaxPool2D(2,2),
        Flatten(),
        Linear(8*7*7, 10)
    )

def one_step_sgd(params, lr):
    with torch.no_grad():
        for p in params:
            p -= lr * p.g

def test_forward_end_to_end_shapes():
    print("\n[E2E] small CNN forward shape...")
    net = build_small_cnn()
    x = torch.randn(5, 1, 28, 28, device=DEVICE)
    logits = net(x)
    assert logits.shape == (5, 10), "Final logits shape should be [N,10]"
    print("  PASS")

def test_training_step_decreases_loss():
    print("\n[E2E] tiny training sanity: loss should drop...")
    N = 128
    x = torch.randn(N, 1, 28, 28, device=DEVICE)
    scores = x.mean(dim=(2,3))
    y = (scores[:,0] > 0).long() % 10  # synthetic labels

    net = build_small_cnn()
    ce  = CrossEntropy()
    params = params_of(net)
    lr = 0.01
    batch = 64

    def epoch():
        idx = torch.randperm(N, device=DEVICE)
        total, correct, loss_sum = 0, 0, 0.0
        for i in range(0, N, batch):
            b = idx[i:i+batch]
            xb, yb = x[b], y[b]
            logits = net(xb)
            loss = ce(logits, yb)

            # zero grads
            for p in params: p.g = zeros_like(p)

            # seed and backprop
            loss.g = torch.tensor(1.0, device=DEVICE)
            ce.backward()
            net.backward(logits)

            loss_sum += loss.item() * xb.size(0)
            correct  += (logits.argmax(1) == yb).sum().item()
            total    += xb.size(0)

            one_step_sgd(params, lr)

        return loss_sum/total, correct/total

    loss1, acc1 = epoch()
    loss2, acc2 = epoch()

    print(f"  epoch1: loss={loss1:.4f}, acc={acc1:.3f}")
    print(f"  epoch2: loss={loss2:.4f}, acc={acc2:.3f}")
    assert loss2 <= loss1 + 1e-5, "Loss did not decrease on synthetic sanity check"
    print("  PASS")

# -------------------------
# Run all
# -------------------------
def run_all_tests():
    print("Device:", DEVICE)
    test_linear_forward_backward()
    test_relu_forward_backward()
    test_cross_entropy_logits()
    test_conv2d_forward_shapes()
    test_conv2d_backward_numeric_small()
    test_maxpool2d_forward_compare_pytorch()
    test_maxpool2d_backward_numeric()
    test_forward_end_to_end_shapes()
    test_training_step_decreases_loss()
    print("\nALL TESTS PASSED (within numeric tolerances)")

if __name__ == "__main__":
    run_all_tests()
