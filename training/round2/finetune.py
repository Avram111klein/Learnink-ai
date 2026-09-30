"""r5 (round 2) = fine-tune of r4 (TIN=1.379 undoes its temperature); round-1 docstring follows: fine-tune of r3 on the owner's labelled TRAIN-FIT symbols (heavily augmented) + replay of the r3 mix
(MNIST + old-style drawn symbols + glyphs2; NO CROHME). Same 28 classes / 784-512-256 / int8 export.
Temperature is fitted on TRAIN-SIDE data only: owner train_val symbols + held-out synthetic.
usage: python3 finetune.py r3_uncal.json mnist.pkl.gz out.json
"""
import sys, json, gzip, pickle, time, base64
import numpy as np
sys.path.insert(0, '__WORKDIR__/hw/training/handwriting')
import train as T
from glyphs2 import Gen2, STYLE_TRAIN2
from train_alg import CLASSES, CI, export, fwd_int8, softmax, ece

t0 = time.time()
rng = np.random.default_rng(4); T.rng = np.random.default_rng(5)
M = json.load(open(sys.argv[1])); assert M['classes'] == CLASSES
Ws = [np.frombuffer(base64.b64decode(L['w']), np.int8).reshape(L['out'], L['in']).astype(np.float32).T * L['s'] for L in M['layers']]
bs = [np.array(L['b'], np.float32) for L in M['layers']]
import os
TIN = float(os.environ.get('TIN', '1'))   # undo the temperature folded into the start model's last layer
Ws[-1] *= TIN; bs[-1] *= TIN

sy = json.load(open('owner_symbols_train.json'))
def aug(st, k):
    out = []
    for _ in range(k):
        S = [np.array(s, float) for s in st]
        S = [s + np.cumsum(rng.normal(0, rng.uniform(0, 0.6), s.shape), 0) * (len(s) > 2) for s in S]   # px jitter
        out.append(T.render(T.affine(S), rng.uniform(1.6, 3.2)).ravel())
    return out
Xo, yo = [], []
for s in sy:
    if s['part'] != 'fit': continue
    k = 60 if s['c'] in 'abcdkmnptysz÷/x' else 30
    Xo += aug(s['s'], k); yo += [CI[s['c']]] * k
Xv = np.array([T.render([np.array(q, float) for q in s['s']], 2.4).ravel() for s in sy if s['part'] == 'val'], np.float32)
yv = np.array([CI[s['c']] for s in sy if s['part'] == 'val'])
print('owner fit aug', len(Xo), 'val', len(Xv), time.time() - t0, flush=True)

tr, va, te = pickle.load(gzip.open(sys.argv[2]), encoding='latin1')
Xm = np.vstack([tr[0], va[0]]).astype(np.float32); ym = np.concatenate([tr[1], va[1]])
idx = rng.permutation(len(Xm)); rep, calm = idx[:25000], idx[25000:28000]
Xr = list(Xm[rep[:12000]]); yr = list(ym[rep[:12000]])
for i in rep[12000:]:
    r = T.skeleton_render(Xm[i])
    if r is not None: Xr.append(r); yr.append(ym[i])
Xs, ys = T.symbols(2000); Xr += list(Xs); yr += list(ys)
g = Gen2(11, STYLE_TRAIN2)
for c in CLASSES[10:]:
    for _ in range(600 if c in '+x/()' else 2500):
        Xr.append(T.render(g.strokes(c), g.lw()).ravel()); yr.append(CI[c])
print('replay', len(Xr), time.time() - t0, flush=True)
X = np.vstack([np.array(Xo, np.float32), np.array(Xr, np.float32)]); y = np.concatenate([np.array(yo), np.array(yr)])
del Xo, Xr

def forward(X, W, b):
    h1 = np.maximum(X @ W[0] + b[0], 0); h2 = np.maximum(h1 @ W[1] + b[1], 0); return h1, h2, h2 @ W[2] + b[2]
def acc(X, y, W, b): return (forward(X, W, b)[2].argmax(1) == y).mean()
W, b = [w.copy() for w in Ws], [x.copy() for x in bs]
mW = [np.zeros_like(w) for w in W]; vW = [np.zeros_like(w) for w in W]; mb = [np.zeros_like(x) for x in b]; vb = [np.zeros_like(x) for x in b]
lr, lam, step, B = 2e-4, 2e-4, 0, 256
best = (acc(Xv, yv, W, b), -1, [w.copy() for w in W], [x.copy() for x in b]); print('epoch -1 owner val acc', round(best[0], 4), flush=True)
for ep in range(6):
    perm = rng.permutation(len(X))
    for s0 in range(0, len(X), B):
        ix = perm[s0:s0 + B]; xb, yb = X[ix], y[ix]
        h1, h2, z = forward(xb, W, b); p = softmax(z); p[np.arange(len(yb)), yb] -= 1; d3 = p / len(yb)
        gW2 = h2.T @ d3; gb2 = d3.sum(0); d2 = (d3 @ W[2].T) * (h2 > 0)
        gW1 = h1.T @ d2; gb1 = d2.sum(0); d1 = (d2 @ W[1].T) * (h1 > 0)
        gW0 = xb.T @ d1; gb0 = d1.sum(0)
        step += 1
        for k, (gw, gb) in enumerate(((gW0, gb0), (gW1, gb1), (gW2, gb2))):
            gw = gw + lam * W[k]
            mW[k] = .9 * mW[k] + .1 * gw; vW[k] = .999 * vW[k] + .001 * gw * gw
            mb[k] = .9 * mb[k] + .1 * gb; vb[k] = .999 * vb[k] + .001 * gb * gb
            c1, c2 = 1 - .9 ** step, 1 - .999 ** step
            W[k] -= lr * (mW[k] / c1) / (np.sqrt(vW[k] / c2) + 1e-8); b[k] -= lr * (mb[k] / c1) / (np.sqrt(vb[k] / c2) + 1e-8)
    a = acc(Xv, yv, W, b); print('epoch', ep, 'owner val acc', round(a, 4), 'mnist test', round(acc(te[0].astype(np.float32), te[1], W, b), 4), time.time() - t0, flush=True)
    if a >= best[0]: best = (a, ep, [w.copy() for w in W], [x.copy() for x in b])
print('best epoch', best[1], 'val', best[0], flush=True)

class Clf: pass
clf = Clf(); clf.coefs_, clf.intercepts_ = best[2], best[3]
# calibration: train-side only (owner val + held-out synthetic + held-out MNIST skeletons)
Xc, yc = [Xv], [yv]
sk = [(T.skeleton_render(Xm[i]), ym[i]) for i in calm]; sk = [(r, l) for r, l in sk if r is not None]
Xc.append(np.array([r for r, _ in sk], np.float32)); yc.append(np.array([l for _, l in sk]))
gc = Gen2(9090, STYLE_TRAIN2); Xg, yg = [], []
for c in CLASSES[10:]:
    for _ in range(200): Xg.append(T.render(gc.strokes(c), gc.lw()).ravel()); yg.append(CI[c])
Xc.append(np.array(Xg, np.float32)); yc.append(np.array(yg))
w_owner = 10   # owner val symbols weighted up (it is the writer we calibrate for)
Xc = np.vstack(Xc); yc = np.concatenate(yc); wc = np.concatenate([np.full(len(yv), w_owner), np.ones(len(yc) - len(yv))])
Z = fwd_int8(export(clf, 1.0), Xc)
nll = lambda t: -(wc * np.log(softmax(Z / t)[np.arange(len(yc)), yc] + 1e-12)).sum() / wc.sum()
ts = np.exp(np.linspace(np.log(0.5), np.log(5), 60)); tb = float(ts[np.argmin([nll(t) for t in ts])])
ov = slice(0, len(yv))
print(f"T={tb:.3f} ECE(all) {ece(softmax(Z), yc):.4f}->{ece(softmax(Z / tb), yc):.4f}  ECE(owner val) {ece(softmax(Z[ov]), yv):.4f}->{ece(softmax(Z[ov] / tb), yv):.4f}", flush=True)
json.dump({'classes': CLASSES, 'layers': export(clf, tb)}, open(sys.argv[3], 'w'))
print('saved', sys.argv[3], time.time() - t0, flush=True)
