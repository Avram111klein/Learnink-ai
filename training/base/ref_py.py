"""No-regression reference in Python: int8 model on MNIST test (10k) + skeleton-redrawn MNIST + synthetic old symbols.
usage: python3 ref_py.py model.json mnist.pkl.gz"""
import sys, json, base64, gzip, pickle, numpy as np
import train as T
M = json.load(open(sys.argv[1])); C = M["classes"]
def fwd(X):
    h = X
    for i, L in enumerate(M["layers"]):
        W = np.frombuffer(base64.b64decode(L["w"]), np.int8).reshape(L["out"], L["in"]).astype(np.float32) * L["s"]
        h = h @ W.T + np.array(L["b"], np.float32)
        if i < len(M["layers"]) - 1: h = np.maximum(h, 0)
    return h
def pred(X, allowed):
    h = fwd(X); mask = np.array([c in allowed for c in C]); h[:, ~mask] = -1e9; return np.array(C)[h.argmax(1)]
_, _, te = pickle.load(gzip.open(sys.argv[2]), encoding="latin1")
Xt, yt = te[0].astype(np.float32), te[1].astype(str)
D = list("0123456789")
print("mnist10k(digits-only)", (pred(Xt, D) == yt).mean(), " mnist10k(all classes)", (pred(Xt, C) == yt).mean())
T.rng = np.random.default_rng(12345)
sk = [(T.skeleton_render(a), yt[i]) for i, a in enumerate(Xt[:3000])]; sk = [(r, l) for r, l in sk if r is not None]
print("skeleton-mnist3k", (pred(np.array([r for r, _ in sk]), D) == np.array([l for _, l in sk])).mean())
T.rng = np.random.default_rng(999)
Xs, ys = T.symbols(400)
print("old synthetic symbols (+x/() , 400 each)", (pred(Xs, C) == np.array(T.CLASSES)[ys]).mean())
