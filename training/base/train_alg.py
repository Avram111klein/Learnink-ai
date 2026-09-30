"""Algebra handwriting classifier: digits + + x / ( ) + letters a b c d k m n p t y s z + ÷.

Same network shape, input and int8 export as train.py (784-512-256-C MLP), so hw.js runs it unchanged.
"×" is folded into "x" (the same shape by hand; context decides in the reader).
"-", "=", ".", ":" stay geometric (hw.js), as before.

Data (NEVER the frozen test set):
  MNIST train+val (raw, augmented, skeleton-redrawn as pen strokes) – as train.py
  synthetic strokes: glyphs.py STYLE_TRAIN (r1) or glyphs2.py STYLE_TRAIN2 incl. pen digits (r2+), seed 7
  CROHME train split ONLY for the reference run r1. CROHME is CC BY-NC-SA 4.0 (non-commercial):
  a model trained with it must NOT ship in the app. App candidates are trained with crohme = "-".
  CROHME is otherwise used for evaluation only (frozen test set).
usage: OMP_NUM_THREADS=1 python3 train_alg.py mnist.pkl.gz <crohme_train.json | -> out_model.json [n_syn] [gen1|gen2]
"""
import gzip, pickle, json, base64, time, sys
import numpy as np
from sklearn.neural_network import MLPClassifier
import train as T
from glyphs import Gen, STYLE_TRAIN

CLASSES = list("0123456789") + ["+", "x", "/", "(", ")"] + list("abcdkmnptysz") + ["÷"]
CI = {c: i for i, c in enumerate(CLASSES)}
FOLD = {"×": "x"}
rng = T.rng
N_SYN = int(sys.argv[4]) if len(sys.argv) > 4 else 6000


def real_symbols(path, reps_letter=12, reps_other=3):
    X, y = [], []
    for s in json.load(open(path))["sym"]:
        c = FOLD.get(s["c"], s["c"])
        if c not in CI:
            continue
        st = [np.array(p, float) for p in s["s"] if len(p)]
        if not st:
            continue
        reps = reps_letter if c in "abcdkmnptysz÷" else reps_other
        for _ in range(reps):
            X.append(T.render(T.affine(st), T.u(1.6, 3.2)).ravel()); y.append(CI[c])
    return np.array(X, np.float32), np.array(y)


GEN = sys.argv[5] if len(sys.argv) > 5 else "gen1"


def synth(n_per, seed=7, n_digit=2000):
    if GEN == "gen2":
        from glyphs2 import Gen2, STYLE_TRAIN2
        g = Gen2(seed, STYLE_TRAIN2)
    else:
        g = Gen(seed, STYLE_TRAIN)
    X, y = [], []
    for c in (CLASSES if GEN == "gen2" else CLASSES[10:]):
        for _ in range(n_per if c in CLASSES[10:] else n_digit):
            X.append(T.render(g.strokes(c), g.lw()).ravel()); y.append(CI[c])
    return np.array(X, np.float32), np.array(y)


def main():
    t0 = time.time()
    tr, va, te = pickle.load(gzip.open(sys.argv[1]), encoding="latin1")
    Xd = np.vstack([tr[0], va[0]]).astype(np.float32); yd = np.concatenate([tr[1], va[1]])
    Xt, yt = te[0].astype(np.float32), te[1]
    idx = rng.permutation(len(Xd))
    Xa = np.array([T.aug_img(Xd[i]) for i in idx[:30000]], np.float32); ya = yd[idx[:30000]]
    print("aug", time.time() - t0, flush=True)
    sk, sy = [], []
    for i in np.concatenate([idx, idx[:30000]]):
        r = T.skeleton_render(Xd[i])
        if r is not None:
            sk.append(r); sy.append(yd[i])
    Xs, ys_ = np.array(sk, np.float32), np.array(sy)
    print("skeleton", len(Xs), time.time() - t0, flush=True)
    if sys.argv[2] == "-":
        Xr, yr = np.zeros((0, 784), np.float32), np.zeros(0, int)
    else:
        Xr, yr = real_symbols(sys.argv[2])   # r1 reference only (CC BY-NC-SA) – NOT for the app
    print("crohme train", len(Xr), np.bincount(yr.astype(int), minlength=len(CLASSES)).tolist(), time.time() - t0, flush=True)
    Xy, yy = synth(N_SYN)
    print("synthetic", len(Xy), time.time() - t0, flush=True)
    X = np.vstack([Xd, Xa, Xs, Xr, Xy]); y = np.concatenate([yd, ya, ys_, yr, yy])
    del Xa, Xs, Xy
    clf = MLPClassifier(hidden_layer_sizes=(512, 256), batch_size=256, learning_rate_init=1e-3, alpha=2e-4,
                        max_iter=40, early_stopping=True, n_iter_no_change=4, validation_fraction=0.05,
                        random_state=1, verbose=True)
    clf.fit(X, y)
    print("train", time.time() - t0, flush=True)
    print("mnist test acc", (clf.predict(Xt) == yt).mean(), flush=True)
    layers = []
    for W, b in zip(clf.coefs_, clf.intercepts_):
        s = float(np.abs(W).max() / 127.0)
        q = np.clip(np.round(W / s), -127, 127).astype(np.int8)
        layers.append({"in": W.shape[0], "out": W.shape[1], "s": s,
                       "w": base64.b64encode(q.T.copy().tobytes()).decode(),
                       "b": [round(float(v), 5) for v in b]})
    json.dump({"classes": CLASSES, "layers": layers}, open(sys.argv[3], "w"))
    print("saved", sys.argv[3], time.time() - t0, flush=True)


def export(clf, T_=1.0):
    layers = []
    for li, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        if li == len(clf.coefs_) - 1:          # temperature folded into the last layer: logits / T
            W, b = W / T_, b / T_
        s = float(np.abs(W).max() / 127.0)
        q = np.clip(np.round(W / s), -127, 127).astype(np.int8)
        layers.append({"in": W.shape[0], "out": W.shape[1], "s": s,
                       "w": base64.b64encode(q.T.copy().tobytes()).decode(),
                       "b": [round(float(v), 5) for v in b]})
    return layers


def fwd_int8(layers, X):
    h = X
    for i, L in enumerate(layers):
        W = np.frombuffer(base64.b64decode(L["w"]), np.int8).reshape(L["out"], L["in"]).astype(np.float32) * L["s"]
        h = h @ W.T + np.array(L["b"], np.float32)
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return h


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def ece(P, y, bins=15):
    conf, pred = P.max(1), P.argmax(1); e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def main3():
    """r3: rebalanced. MNIST (5000 train digits held out for calibration), extra skeleton pass,
    today's old-style drawn + x / ( ) (train.py symbols, 8000 each) PLUS glyphs2 for them (2000 each),
    glyphs2 letters/÷ (N_SYN each), no synthetic digits. Temperature fitted on a held-out synthetic set
    (held-out MNIST digits skeleton-redrawn + glyphs2 seed 777 + old-style symbols seed 778) and folded into
    the last layer, so hw.js needs no change."""
    t0 = time.time()
    tr, va, te = pickle.load(gzip.open(sys.argv[1]), encoding="latin1")
    Xall = np.vstack([tr[0], va[0]]).astype(np.float32); yall = np.concatenate([tr[1], va[1]])
    perm = rng.permutation(len(Xall)); hold, keep = perm[:5000], perm[5000:]
    Xd, yd = Xall[keep], yall[keep]
    Xt, yt = te[0].astype(np.float32), te[1]
    idx = rng.permutation(len(Xd))
    Xa = np.array([T.aug_img(Xd[i]) for i in idx[:30000]], np.float32); ya = yd[idx[:30000]]
    sk, sy = [], []
    for i in np.concatenate([idx, idx[:60000]]):          # one extra skeleton pass: pen-like digits
        r = T.skeleton_render(Xd[i])
        if r is not None:
            sk.append(r); sy.append(yd[i])
    Xs, ys_ = np.array(sk, np.float32), np.array(sy)
    print("mnist+skeleton", len(Xd) + len(Xa) + len(Xs), time.time() - t0, flush=True)
    Xo, yo = T.symbols(8000)                               # today's drawn + x / ( ), class ids 10..14 match
    from glyphs2 import Gen2, STYLE_TRAIN2
    g = Gen2(7, STYLE_TRAIN2)
    Xg, yg = [], []
    for c in CLASSES[10:]:
        for _ in range(2000 if c in "+x/()" else N_SYN):
            Xg.append(T.render(g.strokes(c), g.lw()).ravel()); yg.append(CI[c])
    Xg, yg = np.array(Xg, np.float32), np.array(yg)
    print("synthetic", len(Xo) + len(Xg), time.time() - t0, flush=True)
    X = np.vstack([Xd, Xa, Xs, Xo, Xg]); y = np.concatenate([yd, ya, ys_, yo, yg])
    del Xa, Xs, Xo, Xg
    print("counts", np.bincount(y, minlength=len(CLASSES)).tolist(), flush=True)
    clf = MLPClassifier(hidden_layer_sizes=(512, 256), batch_size=256, learning_rate_init=1e-3, alpha=2e-4,
                        max_iter=30, early_stopping=True, n_iter_no_change=4, validation_fraction=0.05,
                        random_state=1, verbose=True)
    clf.fit(X, y); del X
    print("train", time.time() - t0, "mnist test acc", (clf.predict(Xt) == yt).mean(), flush=True)
    # ---- held-out calibration set (never trained on, not the frozen set) ----
    Xc, yc = [], []
    for i in hold[:3000]:
        r = T.skeleton_render(Xall[i])
        if r is not None:
            Xc.append(r); yc.append(yall[i])
    Xc += [a for a in Xall[hold[3000:]]]; yc += list(yall[hold[3000:]])
    gc = Gen2(777, STYLE_TRAIN2)
    for c in CLASSES[10:]:
        for _ in range(300):
            Xc.append(T.render(gc.strokes(c), gc.lw()).ravel()); yc.append(CI[c])
    T.rng = np.random.default_rng(778); Xs2, ys2 = T.symbols(300)
    Xc = np.vstack([np.array(Xc, np.float32), Xs2]); yc = np.concatenate([np.array(yc), ys2])
    L1 = export(clf, 1.0); Z = fwd_int8(L1, Xc)
    nll = lambda t: -np.log(softmax(Z / t)[np.arange(len(yc)), yc] + 1e-12).mean()
    ts = np.exp(np.linspace(np.log(0.5), np.log(5), 60)); tbest = float(ts[np.argmin([nll(t) for t in ts])])
    print(f"calibration n={len(yc)} acc={(Z.argmax(1) == yc).mean():.4f} T={tbest:.3f} "
          f"ECE before={ece(softmax(Z), yc):.4f} after={ece(softmax(Z / tbest), yc):.4f} "
          f"NLL before={nll(1.0):.4f} after={nll(tbest):.4f}", flush=True)
    json.dump({"classes": CLASSES, "layers": L1}, open(sys.argv[3].replace(".json", "_uncal.json"), "w"))
    json.dump({"classes": CLASSES, "layers": export(clf, tbest)}, open(sys.argv[3], "w"))
    print("saved", sys.argv[3], time.time() - t0, flush=True)


if __name__ == "__main__":
    main3() if GEN == "gen3" else main()
