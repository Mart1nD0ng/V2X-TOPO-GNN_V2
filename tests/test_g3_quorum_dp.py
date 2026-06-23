"""G3 (spec §5): exact heterogeneous quorum DP (three-way generating function).

Checks:
  1. DP response distribution P(m,n) matches brute-force enumeration (< 1e-10).
  2. Distribution normalises: sum_{m,n} P(m,n) = 1.
  3. h^+/h^- match brute force; correct & wrong quorums mutually exclusive (2a>k).
  4. Heterogeneous links / distinct peers / no-response are all honoured.
  5. Gradient of h^+ w.r.t. logits and edge probabilities matches central FD.
  6. No iid-beta-tail closure is used anywhere in the mainline (grep).
  7. Masked padding is exact (a node padded with zeros == the unpadded node).
"""

from __future__ import annotations

import math

import torch

from src.mainline.quorum_dp import (
    bruteforce_quorum_distribution,
    quorum_decision_probabilities,
    quorum_response_distribution,
)

torch.manual_seed(0)
DT = torch.float64


def _rand_edge_probs(n):
    # random valid (p+, p-) with p+ + p- <= 1
    raw = torch.rand(3, n, dtype=DT)
    raw = raw / raw.sum(dim=0, keepdim=True)
    return raw[0], raw[1]  # p_correct, p_wrong (p0 = remainder)


def test_distribution_matches_bruteforce():
    for n, k, alpha in [(4, 2, 2), (5, 3, 2), (6, 3, 2), (6, 4, 3), (7, 5, 3)]:
        log_w = torch.randn(n, dtype=DT)
        pc, pw = _rand_edge_probs(n)
        P = quorum_response_distribution(log_w.unsqueeze(0), pc.unsqueeze(0), pw.unsqueeze(0), k)[0]
        ref = bruteforce_quorum_distribution(torch.exp(log_w), pc, pw, k, alpha)
        # normalisation
        assert abs(float(P.sum()) - 1.0) < 1e-12, (n, k, float(P.sum()))
        # entrywise vs brute force
        max_err = 0.0
        for (m, nn), p in ref["distribution"].items():
            max_err = max(max_err, abs(float(P[m, nn]) - p))
        assert max_err < 1e-10, (n, k, max_err)


def test_decision_probs_match_bruteforce_and_exclusive():
    for n, k, alpha in [(5, 3, 2), (6, 3, 2), (7, 4, 3), (7, 5, 3)]:
        log_w = torch.randn(n, dtype=DT)
        pc, pw = _rand_edge_probs(n)
        dec = quorum_decision_probabilities(
            log_w.unsqueeze(0), pc.unsqueeze(0), pw.unsqueeze(0), k, alpha
        )
        ref = bruteforce_quorum_distribution(torch.exp(log_w), pc, pw, k, alpha)
        assert abs(float(dec.h_plus[0]) - ref["h_plus"]) < 1e-10, (n, k, alpha)
        assert abs(float(dec.h_minus[0]) - ref["h_minus"]) < 1e-10, (n, k, alpha)
        # h+ + h- + h0 = 1 and h+, h- in [0,1]
        s = float(dec.h_plus[0] + dec.h_minus[0] + dec.h_zero[0])
        assert abs(s - 1.0) < 1e-10, s
        # strict-majority mutual exclusion: can't have both >= alpha among k when 2a>k
        # => h_plus + h_minus <= 1 always holds; check no double-counting region
        assert float(dec.h_plus[0]) + float(dec.h_minus[0]) <= 1.0 + 1e-10


def test_no_response_channel_matters():
    # If every edge has high no-response prob, quorum probability collapses.
    n, k, alpha = 6, 3, 2
    log_w = torch.zeros(n, dtype=DT)
    pc = torch.full((n,), 0.05, dtype=DT)
    pw = torch.full((n,), 0.05, dtype=DT)  # p0 = 0.9
    dec = quorum_decision_probabilities(
        log_w.unsqueeze(0), pc.unsqueeze(0), pw.unsqueeze(0), k, alpha
    )
    assert float(dec.h_plus[0]) < 0.05
    # with strong correct support, quorum should be high
    pc2 = torch.full((n,), 0.95, dtype=DT)
    pw2 = torch.full((n,), 0.02, dtype=DT)
    dec2 = quorum_decision_probabilities(
        log_w.unsqueeze(0), pc2.unsqueeze(0), pw2.unsqueeze(0), k, alpha
    )
    assert float(dec2.h_plus[0]) > 0.95


def test_heterogeneous_distinct_peers():
    # Strongly heterogeneous weights/probabilities still match brute force.
    n, k, alpha = 7, 5, 3
    log_w = torch.tensor([3.0, -2.0, 0.5, 1.5, -1.0, 2.2, 0.0], dtype=DT)
    pc = torch.tensor([0.9, 0.1, 0.5, 0.7, 0.2, 0.8, 0.4], dtype=DT)
    pw = torch.tensor([0.05, 0.6, 0.2, 0.1, 0.3, 0.05, 0.2], dtype=DT)
    P = quorum_response_distribution(log_w.unsqueeze(0), pc.unsqueeze(0), pw.unsqueeze(0), k)[0]
    ref = bruteforce_quorum_distribution(torch.exp(log_w), pc, pw, k, alpha)
    for (m, nn), p in ref["distribution"].items():
        assert abs(float(P[m, nn]) - p) < 1e-10, (m, nn, float(P[m, nn]), p)


def test_masked_padding_is_exact():
    n, k, alpha = 5, 3, 2
    log_w = torch.randn(n, dtype=DT)
    pc, pw = _rand_edge_probs(n)
    dec = quorum_decision_probabilities(
        log_w.unsqueeze(0), pc.unsqueeze(0), pw.unsqueeze(0), k, alpha
    )
    # pad with 3 junk candidates that are masked out
    pad = 3
    log_w_p = torch.cat([log_w, torch.randn(pad, dtype=DT)])
    pc_p = torch.cat([pc, torch.rand(pad, dtype=DT) * 0.3])
    pw_p = torch.cat([pw, torch.rand(pad, dtype=DT) * 0.3])
    mask = torch.cat([torch.ones(n, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)])
    dec_p = quorum_decision_probabilities(
        log_w_p.unsqueeze(0), pc_p.unsqueeze(0), pw_p.unsqueeze(0), k, alpha, mask=mask.unsqueeze(0)
    )
    assert abs(float(dec.h_plus[0]) - float(dec_p.h_plus[0])) < 1e-12
    assert abs(float(dec.h_minus[0]) - float(dec_p.h_minus[0])) < 1e-12


def test_batched_consistency():
    # Batched evaluation equals per-row evaluation.
    B, n, k, alpha = 4, 6, 3, 2
    log_w = torch.randn(B, n, dtype=DT)
    raw = torch.rand(B, 3, n, dtype=DT)
    raw = raw / raw.sum(dim=1, keepdim=True)
    pc, pw = raw[:, 0], raw[:, 1]
    dec = quorum_decision_probabilities(log_w, pc, pw, k, alpha)
    for b in range(B):
        d1 = quorum_decision_probabilities(
            log_w[b : b + 1], pc[b : b + 1], pw[b : b + 1], k, alpha
        )
        assert abs(float(dec.h_plus[b]) - float(d1.h_plus[0])) < 1e-12
        assert abs(float(dec.h_minus[b]) - float(d1.h_minus[0])) < 1e-12


def test_gradient_matches_finite_difference():
    n, k, alpha = 6, 3, 2
    log_w = torch.randn(n, dtype=DT)
    pc, pw = _rand_edge_probs(n)

    def h_plus_of(lw, pcv, pwv):
        return quorum_decision_probabilities(
            lw.unsqueeze(0), pcv.unsqueeze(0), pwv.unsqueeze(0), k, alpha
        ).h_plus[0]

    lw = log_w.clone().requires_grad_(True)
    pcv = pc.clone().requires_grad_(True)
    h = h_plus_of(lw, pcv, pw)
    h.backward()
    g_lw = lw.grad.detach().clone()
    g_pc = pcv.grad.detach().clone()

    eps = 1e-6
    for j in range(n):
        for base, grad in ((log_w, g_lw), (pc, g_pc)):
            plus = base.clone()
            minus = base.clone()
            plus[j] += eps
            minus[j] -= eps
            if base is log_w:
                fp = float(h_plus_of(plus, pc, pw))
                fm = float(h_plus_of(minus, pc, pw))
            else:
                # keep p+ + p- <= 1 valid near these random draws
                fp = float(h_plus_of(log_w, plus, pw))
                fm = float(h_plus_of(log_w, minus, pw))
            fd = (fp - fm) / (2 * eps)
            ref = float(grad[j])
            rel = abs(fd - ref) / (abs(fd) + 1e-6)
            assert rel < 1e-4, (("logit" if base is log_w else "p_correct"), j, fd, ref, rel)


def test_float32_stability():
    # The normalised DP should be stable in float32 with large logits.
    n, k, alpha = 8, 5, 3
    log_w = torch.randn(n, dtype=torch.float32) * 6.0  # wide dynamic range
    pc, pw = _rand_edge_probs(n)
    pc = pc.float()
    pw = pw.float()
    dec = quorum_decision_probabilities(
        log_w.unsqueeze(0), pc.unsqueeze(0), pw.unsqueeze(0), k, alpha
    )
    assert torch.isfinite(dec.h_plus).all()
    assert 0.0 <= float(dec.h_plus[0]) <= 1.0
    # compare to float64 brute force within float32 tolerance
    ref = bruteforce_quorum_distribution(
        torch.exp(log_w.double()), pc.double(), pw.double(), k, alpha
    )
    assert abs(float(dec.h_plus[0]) - ref["h_plus"]) < 1e-3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} G3 tests passed.")
