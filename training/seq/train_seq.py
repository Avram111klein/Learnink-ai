"""Train the stroke-sequence shadow reader (reader/seq.js) → int8 JSON model (models/seq_v1.json format).
Inputs are produced by node scripts that call reader/seq.js features() (same code as runtime):
  __WORKDIR__/seq/data/feat_train.json  {classes, rows:[{t, sess, f}]}  real aligned train groups
  __WORKDIR__/seq/data/aug_train.json   {rows:[{t, sess, f}]}           augmented copies of the same groups
Tuning uses ONLY the training items: leave-one-session-out folds over the train sessions (the test items are never
read here). Temperature is fitted on the out-of-fold predictions.
usage: python3 train_seq.py cv  [grid]          → prints CV accuracy per setting
       python3 train_seq.py fit <H> <wd> <ep> <aug 0|1> <out.json>
"""
import sys, json, base64, numpy as np
W = '__WORKDIR__/seq/data'
FT = json.load(open(f'{W}/feat_train.json')); CL = FT['classes']; CI = {c: i for i, c in enumerate(CL)}
R = FT['rows']; X = np.array([r['f'] for r in R], np.float32); Y = np.array([CI[r['t']] for r in R]); SS = np.array([r['sess'] for r in R])
AG = json.load(open(f'{W}/aug_train.json'))['rows']; XA = np.array([r['f'] for r in AG], np.float32); YA = np.array([CI[r['t']] for r in AG]); SA = np.array([r['sess'] for r in AG])
K = len(CL)

def train(Xt, Yt, H, wd, ep, seed=0, lr=2e-3, bs=128):
    rng = np.random.default_rng(seed); mu = Xt.mean(0); sd = Xt.std(0) + 1e-3; Z = (Xt - mu) / sd
    dims = [Z.shape[1]] + list(H) + [K]
    Ws = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in zip(dims, dims[1:])]; bs_ = [np.zeros(b, np.float32) for b in dims[1:]]
    P = Ws + bs_; M1 = [np.zeros_like(p) for p in P]; M2 = [np.zeros_like(p) for p in P]; step = 0
    cw = np.bincount(Yt, minlength=K).astype(np.float32); cw = (cw.sum() / np.maximum(cw, 1) / K) ** 0.5   # soft class balancing
    for e in range(ep):
        idx = rng.permutation(len(Z)); a = lr * 0.5 * (1 + np.cos(np.pi * e / ep))
        for k in range(0, len(Z), bs):
            b = idx[k:k + bs]; h = [Z[b]]
            for i in range(len(Ws)):
                z = h[-1] @ Ws[i] + bs_[i]
                if i < len(Ws) - 1:
                    z = np.maximum(z, 0)
                    z = z * (rng.random(z.shape) > 0.2) / 0.8   # dropout
                h.append(z)
            p = np.exp(h[-1] - h[-1].max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            g = p; g[np.arange(len(b)), Yt[b]] -= 1; g *= cw[Yt[b]][:, None] / len(b)
            gW = [None] * len(Ws); gb = [None] * len(Ws)
            for i in range(len(Ws) - 1, -1, -1):
                gW[i] = h[i].T @ g + wd * Ws[i]; gb[i] = g.sum(0)
                if i: g = (g @ Ws[i].T) * (h[i] > 0)
            step += 1
            for j, (p_, g_) in enumerate(zip(P, gW + gb)):
                M1[j] = 0.9 * M1[j] + 0.1 * g_; M2[j] = 0.999 * M2[j] + 0.001 * g_ * g_
                p_ -= a * (M1[j] / (1 - 0.9 ** step)) / (np.sqrt(M2[j] / (1 - 0.999 ** step)) + 1e-8)
    return dict(mu=mu, sd=sd, Ws=Ws, bs=bs_)

def quant(m):   # int8 per layer, exactly what the JS will run
    q = dict(m); q['Ws'] = []; q['sc'] = []
    for Wl in m['Ws']:
        s = float(np.abs(Wl).max() / 127) or 1.0; q['Ws'].append(np.clip(np.round(Wl / s), -127, 127).astype(np.int8).astype(np.float32) * s); q['sc'].append(s)
    return q

def logits(m, Xe):
    h = (Xe - m['mu']) / m['sd']
    for i, (Wl, b) in enumerate(zip(m['Ws'], m['bs'])):
        h = h @ Wl + b
        if i < len(m['Ws']) - 1: h = np.maximum(h, 0)
    return h

def sm(z, T=1.0):
    z = z / T; p = np.exp(z - z.max(1, keepdims=True)); return p / p.sum(1, keepdims=True)

def folds():   # leave one train session out; sessions with < 100 groups always stay in training
    big = [s for s in np.unique(SS) if (SS == s).sum() >= 100]
    for s in big: yield s, SS != s, SA != s, SS == s

def cv(H, wd, ep, aug):
    Z = np.zeros((len(X), K), np.float32); m_ = np.zeros(len(X), bool)
    for s, tr, tra, va in folds():
        Xt, Yt = (np.vstack([X[tr], XA[tra]]), np.concatenate([Y[tr], YA[tra]])) if aug else (X[tr], Y[tr])
        m = quant(train(Xt, Yt, H, wd, ep)); Z[va] = logits(m, X[va]); m_[va] = True
    return Z, m_

def fitT(Z, Yv):
    best = (1e9, 1.0)
    for T in np.arange(0.5, 3.01, 0.05):
        nll = -np.log(sm(Z, T)[np.arange(len(Yv)), Yv] + 1e-9).mean(); best = min(best, (nll, T))
    return float(best[1])

if __name__ == '__main__':
    if sys.argv[1] == 'cv':
        grid = json.loads(sys.argv[2]) if len(sys.argv) > 2 else [[[128], 1e-4, 30, 1]]
        for H, wd, ep, aug in grid:
            Z, m_ = cv(H, wd, ep, aug); Yv = Y[m_]; P = sm(Z[m_]); T = fitT(Z[m_], Yv); Pt = sm(Z[m_], T)
            acc = (P.argmax(1) == Yv).mean(); sure = Pt.max(1) >= 0.5; wr = (Pt.argmax(1) != Yv)[sure].mean()
            per = {s: round(float((P.argmax(1) == Yv)[SS[m_] == s].mean()), 3) for s in np.unique(SS[m_])}
            print(json.dumps(dict(H=H, wd=wd, ep=ep, aug=aug, n=int(m_.sum()), acc=round(float(acc), 4), T=round(T, 2), sureWrong=round(float(wr), 4), perSess=per)), flush=True)
    else:
        H, wd, ep, aug, out = json.loads(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
        Z, m_ = cv(H, wd, ep, aug); T = fitT(Z[m_], Y[m_]); print('cv acc', round(float((Z[m_].argmax(1) == Y[m_]).mean()), 4), 'T', round(T, 2))
        Xt, Yt = (np.vstack([X, XA]), np.concatenate([Y, YA])) if aug else (X, Y)
        m = train(Xt, Yt, H, wd, ep); q = quant(m)
        L = []
        for i, (Wl, b) in enumerate(zip(m['Ws'], m['bs'])):
            s = q['sc'][i]; Wi = np.clip(np.round(Wl / s), -127, 127).astype(np.int8)
            bb = b.astype(np.float64); ss = s
            if i == len(m['Ws']) - 1: ss = s / T; bb = bb / T          # temperature folded into the last layer
            L.append(dict(**{'in': int(Wl.shape[0]), 'out': int(Wl.shape[1]), 's': ss, 'w': base64.b64encode(Wi.T.copy().tobytes()).decode(), 'b': [round(float(v), 6) for v in bb]}))
        json.dump(dict(name='seq_v1', classes=CL, N=32, dim=int(X.shape[1]), T=round(T, 3), mean=[round(float(v), 5) for v in m['mu']], std=[round(float(v), 5) for v in m['sd']], layers=L), open(out, 'w'))
        print('wrote', out)
