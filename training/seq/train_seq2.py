"""seq_v2: the sequence reader trained on aligned symbol groups (as v1) PLUS the groups segment() really forms in the
training items, where a group that is not exactly one aligned symbol (fragment / several symbols) is class "∅".
29 outputs = the 28 symbols + "∅". Same features (reader/seq.js), same 328→128→K MLP, int8.
Only training items (modelVersion 1/2). Folds = leave one training session out (sessions with <100 groups always train).
Inputs (__WORKDIR__/seq): data/feat_train.json, data/aug_train.json (aligned, as v1); v2/feat_null_aug.json (∅ groups,
real + augmented); v2/feat_seg_train.json + v2/seg_train.json (all segment groups: features + pixel dist, for W tuning).
usage: python3 train_seq2.py cv '<grid>'      → OOF accuracy (symbols / ∅) per setting
       python3 train_seq2.py fit H wd ep out.json oof.json  → final model + out-of-fold seq probs on the segment groups
       python3 train_seq2.py tune oof.json    → W per kind (digits/letters/operators) from {0.25,0.5,0.75} ∪ {0} on OOF"""
import sys, json, base64, numpy as np
W = '__WORKDIR__/seq'
FT = json.load(open(f'{W}/data/feat_train.json')); CL = FT['classes'] + ['∅']; CI = {c: i for i, c in enumerate(CL)}; K = len(CL)
def load(rows): return np.array([r['f'] for r in rows], np.float32), np.array([CI[r['t']] for r in rows]), np.array([r['sess'] for r in rows])
X1, Y1, S1 = load(FT['rows']); XA, YA, SA = load(json.load(open(f'{W}/data/aug_train.json'))['rows'])
NR = json.load(open(f'{W}/v2/feat_null_aug.json'))['rows']
XN, YN, SN = load([r for r in NR if not r['aug']]); XNA, YNA, SNA = load([r for r in NR if r['aug']])
SG = json.load(open(f'{W}/v2/feat_seg_train.json'))['rows']; XG, YG, SGs = load(SG)
def train(Xt, Yt, H, wd, ep, seed=0, lr=2e-3, bs=128):
    rng = np.random.default_rng(seed); mu = Xt.mean(0); sd = Xt.std(0) + 1e-3; Z = (Xt - mu) / sd
    dims = [Z.shape[1]] + list(H) + [K]
    Ws = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in zip(dims, dims[1:])]; bs_ = [np.zeros(b, np.float32) for b in dims[1:]]
    P = Ws + bs_; M1 = [np.zeros_like(p) for p in P]; M2 = [np.zeros_like(p) for p in P]; step = 0
    cw = np.bincount(Yt, minlength=K).astype(np.float32); cw = (cw.sum() / np.maximum(cw, 1) / K) ** 0.5
    for e in range(ep):
        idx = rng.permutation(len(Z)); a = lr * 0.5 * (1 + np.cos(np.pi * e / ep))
        for k in range(0, len(Z), bs):
            b = idx[k:k + bs]; h = [Z[b]]
            for i in range(len(Ws)):
                z = h[-1] @ Ws[i] + bs_[i]
                if i < len(Ws) - 1: z = np.maximum(z, 0) * (rng.random(z.shape) > 0.2) / 0.8
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
def qW(Wl): s = float(np.abs(Wl).max() / 127) or 1.0; return np.clip(np.round(Wl / s), -127, 127).astype(np.int8), s
def logits(m, Xe):
    h = (Xe - m['mu']) / m['sd']
    for i, (Wl, b) in enumerate(zip(m['Ws'], m['bs'])):
        q, s = qW(Wl); h = h @ (q.astype(np.float32) * s) + b
        if i < len(m['Ws']) - 1: h = np.maximum(h, 0)
    return h
def sm(z, T=1.0): z = z / T; p = np.exp(z - z.max(1, keepdims=True)); return p / p.sum(1, keepdims=True)
def fitT(Z, Yv): return float(min((-np.log(sm(Z, T)[np.arange(len(Yv)), Yv] + 1e-9).mean(), T) for T in np.arange(0.5, 3.01, 0.05))[1])
BIG = [s for s in np.unique(S1) if (S1 == s).sum() >= 100]
def trainset(keep):   # keep(sessions array) → mask
    return np.vstack([X1[keep(S1)], XA[keep(SA)], XN[keep(SN)], XNA[keep(SNA)]]), np.concatenate([Y1[keep(S1)], YA[keep(SA)], YN[keep(SN)], YNA[keep(SNA)]])
def cv(H, wd, ep):
    Z1 = np.full((len(X1), K), np.nan, np.float32); ZG = np.full((len(XG), K), np.nan, np.float32)
    for s in BIG:
        m = train(*trainset(lambda S: S != s), H, wd, ep); Z1[S1 == s] = logits(m, X1[S1 == s]); ZG[SGs == s] = logits(m, XG[SGs == s])
    return Z1, ZG
def report(Z1, ZG, T):
    o1 = ~np.isnan(Z1[:, 0]); oG = ~np.isnan(ZG[:, 0]); P1 = sm(Z1[o1], T); PG = sm(ZG[oG], T); yG = YG[oG]; nul = CI['∅']
    sym = yG != nul
    return dict(alignedAcc=round(float((P1.argmax(1) == Y1[o1]).mean()), 4), segSymAcc=round(float((PG[sym].argmax(1) == yG[sym]).mean()), 4),
                segSymAcc28=round(float((PG[sym][:, :nul].argmax(1) == yG[sym]).mean()), 4), nullRecall=round(float((PG[~sym].argmax(1) == nul).mean()), 4),
                nullFalse=round(float((PG[sym].argmax(1) == nul).mean()), 4), meanPnull_sym=round(float(PG[sym][:, nul].mean()), 3), meanPnull_null=round(float(PG[~sym][:, nul].mean()), 3))
if __name__ == '__main__':
    if sys.argv[1] == 'cv':
        for H, wd, ep in json.loads(sys.argv[2]):
            Z1, ZG = cv(H, wd, ep); o = ~np.isnan(np.vstack([Z1, ZG])[:, 0]); T = fitT(np.vstack([Z1, ZG])[o], np.concatenate([Y1, YG])[o])
            print(json.dumps(dict(H=H, wd=wd, ep=ep, T=round(T, 2), **report(Z1, ZG, T))), flush=True)
    elif sys.argv[1] == 'fit':
        H, wd, ep, out, oofp = json.loads(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
        Z1, ZG = cv(H, wd, ep); o = ~np.isnan(np.vstack([Z1, ZG])[:, 0]); T = fitT(np.vstack([Z1, ZG])[o], np.concatenate([Y1, YG])[o])
        print('OOF', report(Z1, ZG, T), 'T', round(T, 2))
        json.dump(dict(classes=CL, T=T, rows=[dict(id=r['id'], sess=r['sess'], t=r['t'], p=(None if np.isnan(z[0]) else [round(float(v), 5) for v in sm(z[None], T)[0]])) for r, z in zip(SG, ZG)]), open(oofp, 'w'))
        m = train(*trainset(lambda S: np.ones(len(S), bool)), H, wd, ep); L = []
        for i, (Wl, b) in enumerate(zip(m['Ws'], m['bs'])):
            q, s = qW(Wl); bb = b.astype(np.float64)
            if i == len(m['Ws']) - 1: s, bb = s / T, bb / T
            L.append({'in': int(Wl.shape[0]), 'out': int(Wl.shape[1]), 's': s, 'w': base64.b64encode(q.T.copy().tobytes()).decode(), 'b': [round(float(v), 6) for v in bb]})
        json.dump(dict(name='seq_v2', classes=CL, N=32, dim=int(X1.shape[1]), T=round(T, 3), mean=[round(float(v), 5) for v in m['mu']], std=[round(float(v), 5) for v in m['sd']], layers=L), open(out, 'w'))
        print('wrote', out)

def fuse_label(dist, CX, ps_seq, seq_classes, allowed, Wk):
    """Python mirror of hw_fuse.js fuse(): dist = pixel allowed-normalised dist over CX; ps_seq = seq probs."""
    pc = CX[int(np.argmax(dist))]; kind = 'd' if pc.isdigit() else 'l' if pc.isalpha() and len(pc) == 1 and pc.islower() else 'o'
    pn = ps_seq[seq_classes.index('∅')] if '∅' in seq_classes else 0.0
    ps = np.array([0.0 if (c not in seq_classes or c not in allowed) else ps_seq[seq_classes.index(c)] for c in CX]); z = ps.sum()
    w = Wk[kind] * (1 - pn)
    if not z > 0 or not w > 0: return pc
    m = (1 - w) * np.array(dist) + w * ps / z
    best = max((i for i, c in enumerate(CX) if c in allowed), key=lambda i: m[i]); return CX[best]
if __name__ == '__main__' and sys.argv[1] == 'tune':
    O = json.load(open(sys.argv[2])); SGr = json.load(open(f'{W}/v2/seg_train.json')); CX = None
    HWC = SGr['classes']; CX = HWC + [l for l in 'abcdkmnpty' if l not in HWC]
    items = {x['id']: x for x in json.load(open(f'{W}/data/items_train.json'))}
    import re
    def allowed_of(it):
        src = it['promptText']; t = re.sub(r'[\[\]>\s]', '', src); m = []; d = 0; T = []
        i = 0
        while i < len(src):
            c = src[i]
            if c == '[': d += 1
            elif c == ']': d -= 1
            elif c == '>' or c.isspace(): pass
            elif c == '^':
                j = i + 1
                while j < len(src) and src[j].isdigit(): T.append(src[j]); m.append(False); j += 1
                i = j; continue
            else: T.append(c); m.append(c == '/' and d > 0)
            i += 1
        v = {x for x, b in zip(T, m) if re.fullmatch('[a-z]', x) or x == '÷' or (x == '/' and not b)}
        return set('0123456789()+') | v
    R = [(g, o) for g, o in zip(SGr['rows'], O['rows']) if o['p'] is not None and g['t'] != '∅']
    assert all(g['id'] == o['id'] and g['t'] == o['t'] for g, o in zip(SGr['rows'], O['rows']))
    kindT = lambda t: 'd' if t.isdigit() else 'l' if t.isalpha() else 'o'
    import itertools
    pre = [(g, o, allowed_of(items[g['id']])) for g, o in R]
    def run(Wk):
        ok = {k: 0 for k in 'dlo'}; n = {k: 0 for k in 'dlo'}; pairs_f = []; pairs_b = []
        for g, o, al in pre:
            b = g['ch']   # the reader's final label; rule-labelled groups (':', '.', '-', '=') never reach the fusion
            f = fuse_label(g['dist'], CX, o['p'], O['classes'], al, Wk) if b in CX else b; k = kindT(g['t'])
            if f in ('+', 't') and '+' in al and 't' in al: f = g['ch'] if g['ch'] in ('+', 't') else f   # plusOrT() re-decides +/t after the label (mirror: keep reader's)
            n[k] += 1; ok[k] += f == g['t']
            if f != b: (pairs_f if f == g['t'] else pairs_b if b == g['t'] else []).append(f"{b}->{f}")
        return ok, n, pairs_f, pairs_b
    ok0, n0, _, _ = run(dict(d=0, l=0, o=0))
    print('reader label (pixel in-sample!) per true kind:', {k: f'{ok0[k]}/{n0[k]}' for k in 'dlo'}, 'total', sum(ok0.values()), '/', sum(n0.values()))
    GR = [0.25, 0.5, 0.75]; best = None
    for wd_, wl_, wo_ in itertools.product(GR, GR, GR):
        ok, n, pf, pb = run(dict(d=wd_, l=wl_, o=wo_)); tot = sum(ok.values())
        print(f'W d={wd_} l={wl_} o={wo_}: digits {ok["d"]}/{n["d"]} letters {ok["l"]}/{n["l"]} ops {ok["o"]}/{n["o"]} total {tot} (fixed {len(pf)} broke {len(pb)})')
        cand = (ok['d'] >= ok0['d'], tot, -(wd_ + wl_ + wo_))
        if best is None or cand > best[0]: best = (cand, (wd_, wl_, wo_), pf, pb)
    print('SELECTED (digits not below pixel, max total, ties -> smaller weights):', best[1], 'fixed', sorted(best[2]), 'broke', sorted(best[3]))
