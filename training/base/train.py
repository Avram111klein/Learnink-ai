"""Train a small on-device handwriting classifier for the math notebook.

Classes: digits 0-9 and + x / ( )
Minus, equals, dots and colons are recognised by geometry in JS, not here.
Input: 28x28 image rendered from pen strokes exactly like the JS renderer.
"""
import gzip, pickle, json, base64, time, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.morphology import skeletonize
from sklearn.neural_network import MLPClassifier

rng = np.random.default_rng(7)
CLASSES = list("0123456789") + ["+", "x", "/", "(", ")"]
F = 4

def center(a):
    m = a.sum()
    if m <= 0:
        return a
    cy, cx = ndimage.center_of_mass(a)
    dy, dx = int(round(14 - cy)), int(round(14 - cx))
    out = np.zeros_like(a)
    ys, yd = (slice(0, 28 - dy), slice(dy, 28)) if dy >= 0 else (slice(-dy, 28), slice(0, 28 + dy))
    xs, xd = (slice(0, 28 - dx), slice(dx, 28)) if dx >= 0 else (slice(-dx, 28), slice(0, 28 + dx))
    out[yd, xd] = a[ys, xs]
    return out

def render(strokes, lw):
    """strokes: list of (n,2) arrays in any coordinates, y down. Mirrors JS preprocess()."""
    pts = np.concatenate(strokes)
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    w, h = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
    s = 20.0 / max(w, h)
    ox, oy = (28 - w * s) / 2, (28 - h * s) / 2
    img = Image.new("L", (28 * F, 28 * F), 0)
    d = ImageDraw.Draw(img)
    r = lw * F / 2
    for st in strokes:
        P = [(((p[0] - x0) * s + ox) * F, ((p[1] - y0) * s + oy) * F) for p in st]
        if len(P) > 1:
            d.line(P, fill=255, width=max(1, int(round(lw * F))))
        for px, py in P:
            d.ellipse([px - r, py - r, px + r, py + r], fill=255)
    a = np.asarray(img, dtype=np.float32).reshape(28, F, 28, F).mean((1, 3)) / 255.0
    return center(a)

# ---------- synthetic symbol strokes ----------
def seg(p, q, n=14):
    t = np.linspace(0, 1, n)[:, None]
    return np.array(p, float) + (np.array(q, float) - np.array(p, float)) * t

def wobble(st, amt):
    n = len(st)
    noise = np.cumsum(rng.normal(0, amt, (n, 2)), 0)
    noise -= np.linspace(0, 1, n)[:, None] * noise[-1]  # keep endpoints roughly
    return st + noise

def affine(strokes):
    a = np.deg2rad(rng.uniform(-12, 12)); sh = rng.uniform(-0.2, 0.2)
    sx, sy = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
    M = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.array([[1, sh], [0, 1]]) @ np.diag([sx, sy])
    return [st @ M.T for st in strokes]

def u(a, b):
    return rng.uniform(a, b)

def gen_symbol(c):
    if c == "+":
        cx, cy = u(.4, .6), u(.4, .6)
        v = seg((cx + u(-.05, .05), u(0, .2)), (cx + u(-.05, .05), u(.8, 1)))
        h = seg((u(0, .2), cy + u(-.05, .05)), (u(.8, 1), cy + u(-.05, .05)))
        S = [v, h] if rng.random() < .5 else [h, v]
    elif c == "x":
        a = seg((u(0, .15), u(0, .15)), (u(.85, 1), u(.85, 1)))
        b = seg((u(.85, 1), u(0, .15)), (u(0, .15), u(.85, 1)))
        if rng.random() < .25:  # curly x: two arcs back to back
            t = np.linspace(-1.1, 1.1, 16)
            a = np.stack([.45 - .35 * np.cos(t), .5 + .45 * np.sin(t)], 1)
            b = np.stack([.55 + .35 * np.cos(t), .5 + .45 * np.sin(t)], 1)
        S = [a, b]
    elif c == "/":
        S = [seg((u(.1, .35), u(.9, 1)), (u(.65, .9), u(0, .1)))]
    elif c in "()":
        t = np.linspace(np.deg2rad(u(110, 130)), np.deg2rad(u(230, 250)), 18)
        rx, ry = u(.25, .5), u(.5, .6)
        x = .7 + rx * np.cos(t); y = .5 + ry * np.sin(t)
        if c == ")":
            x = 1 - x
        S = [np.stack([x, y], 1)]
    S = [wobble(st, u(0.002, 0.012)) for st in S]
    return affine(S)

def symbols(n_per):
    X, y = [], []
    for ci, c in enumerate(CLASSES[10:], start=10):
        for _ in range(n_per):
            X.append(render(gen_symbol(c), u(1.6, 3.2)).ravel()); y.append(ci)
    return np.array(X, np.float32), np.array(y)

# ---------- digits ----------
def aug_img(a):
    a = a.reshape(28, 28)
    ang = np.deg2rad(rng.uniform(-12, 12)); sh = rng.uniform(-.2, .2); sc = rng.uniform(.85, 1.1)
    M = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) @ np.array([[1, sh], [0, 1]]) / sc
    c = np.array([14, 14])
    out = ndimage.affine_transform(a, M, offset=c - M @ c, order=1)
    if rng.random() < .25:
        out = ndimage.grey_dilation(out, size=(2, 2))
    return center(np.clip(out, 0, 1)).ravel()

def skeleton_render(a):
    """Re-draw an MNIST digit the way pen strokes are drawn: skeleton + uniform round pen."""
    big = ndimage.zoom(a.reshape(28, 28), 4, order=1) > 0.35
    sk = skeletonize(big)
    ys, xs = np.nonzero(sk)
    if len(xs) < 5:
        return None
    pts = np.stack([xs, ys], 1).astype(float)
    pts = [p[None, :] for p in pts]            # dots only
    pts = affine(pts)
    return render(pts, u(1.6, 3.2)).ravel()

def main():
    t0 = time.time()
    tr, va, te = pickle.load(gzip.open(sys.argv[1]), encoding="latin1")
    Xd = np.vstack([tr[0], va[0]]).astype(np.float32); yd = np.concatenate([tr[1], va[1]])
    Xt, yt = te[0].astype(np.float32), te[1]
    idx = rng.permutation(len(Xd))
    Xa = np.array([aug_img(Xd[i]) for i in idx[:30000]], np.float32); ya = yd[idx[:30000]]
    print("aug", time.time() - t0, flush=True)
    sk, sy = [], []
    for i in np.concatenate([idx, idx[:30000]]):
        r = skeleton_render(Xd[i])
        if r is not None:
            sk.append(r); sy.append(yd[i])
    Xs, ys_ = np.array(sk, np.float32), np.array(sy)
    print("skeleton", len(Xs), time.time() - t0, flush=True)
    Xy, yy = symbols(8000)
    print("symbols", time.time() - t0, flush=True)
    X = np.vstack([Xd, Xa, Xs, Xy]); y = np.concatenate([yd, ya, ys_, yy])
    clf = MLPClassifier(hidden_layer_sizes=(512, 256), batch_size=256, learning_rate_init=1e-3, alpha=2e-4,
                        max_iter=40, early_stopping=True, n_iter_no_change=4, validation_fraction=0.06,
                        random_state=1, verbose=True)
    clf.fit(X, y)
    print("train", time.time() - t0, flush=True)
    print("mnist test acc", (clf.predict(Xt) == yt).mean())
    sk_t = [skeleton_render(a) for a in Xt[:3000]]
    ok = [(r, yt[i]) for i, r in enumerate(sk_t) if r is not None]
    print("skeleton-rendered test acc", (clf.predict(np.array([r for r, _ in ok])) == np.array([l for _, l in ok])).mean())
    Xyt, yyt = symbols(400)
    print("symbol test acc", (clf.predict(Xyt) == yyt).mean())
    # export int8
    layers = []
    for W, b in zip(clf.coefs_, clf.intercepts_):
        s = float(np.abs(W).max() / 127.0)
        q = np.clip(np.round(W / s), -127, 127).astype(np.int8)
        layers.append({"in": W.shape[0], "out": W.shape[1], "s": s,
                       "w": base64.b64encode(q.T.copy().tobytes()).decode(),  # row = output neuron
                       "b": [round(float(v), 5) for v in b]})
    json.dump({"classes": CLASSES, "layers": layers}, open("model.json", "w"))
    # quantised accuracy check
    def fwd(Xq):
        h = Xq
        for i, L in enumerate(layers):
            W = np.frombuffer(base64.b64decode(L["w"]), np.int8).reshape(L["out"], L["in"]).astype(np.float32) * L["s"]
            h = h @ W.T + np.array(L["b"], np.float32)
            if i < len(layers) - 1:
                h = np.maximum(h, 0)
        return h.argmax(1)
    print("int8 mnist acc", (fwd(Xt) == yt).mean(), "int8 symbol acc", (fwd(Xyt) == yyt).mean())
    # a few reference vectors for JS parity testing
    json.dump({"x": [list(map(float, np.round(Xyt[i], 4))) for i in range(3)], "y": [int(v) for v in fwd(Xyt[:3])]}, open("parity.json", "w"))

if __name__ == "__main__":
    main()
