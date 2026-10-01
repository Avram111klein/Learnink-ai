"""Stroke-texture metrics, real TRAIN vs generator versions: turning angle per point, point spacing, and the reader's
isFlat/barConf (ported from the trainer page) on '-', '=' and fraction-bar strokes."""
import json, gzip, glob, sys, math, random, importlib
import numpy as np
from ml import roles
SP = '__WORKDIR__'
def bx(P): P = np.asarray(P, float); return P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()
def cH_of(strokes):
    hs = sorted(b[3] - b[1] for b in map(bx, strokes) if (b[3] - b[1]) >= 0.5 * (b[2] - b[0]))
    return max(8, hs[int(len(hs) * 0.75)] if hs else 20)
def is_flat(P, cH):   # reader's isFlat
    x0, y0, x1, y1 = bx(P); w, h = x1 - x0, y1 - y0
    if w <= 0.28 * cH: return False
    if h < 0.34 * max(w, 1): return True
    a, b = P[0], P[-1]; dx, dy = b[0] - a[0], b[1] - a[1]; L = math.hypot(dx, dy)
    if not (L >= 0.85 * w and abs(dy) < 0.62 * abs(dx)): return False
    dev = max(abs((x - a[0]) * dy - (y - a[1]) * dx) / L for x, y in P)
    return dev <= max(1.2, 0.16 * L) and h < 0.62 * w
def bar_conf(P, cH):  # reader's barConf
    a, b = P[0], P[-1]; dx, dy = b[0] - a[0], b[1] - a[1]; L = math.hypot(dx, dy) or 1
    dev = max(abs((x - a[0]) * dy - (y - a[1]) * dx) / L for x, y in P); w = bx(P)[2] - bx(P)[0]
    straight = max(0, 1 - dev / (0.25 * L)); level = max(0, 1 - abs(dy) / (0.6 * abs(dx) + 1e-6)); ln = min(1, w / (0.45 * cH))
    return max(0.3, min(0.995, 0.5 + 0.5 * min(straight, level, ln)))
def stats(exs):
    """exs: list of (strokes, {stroke_index: label}) ; strokes [[x,y,...]]"""
    turn, sp, zero, dash = [], [], [], {'-': [], '=': [], 'bar': []}
    for S, lab in exs:
        S = [[p[:2] for p in s] for s in S if s]; cH = cH_of(S)
        for k, P in enumerate(S):
            A = np.array(P, float); d = np.hypot(*np.diff(A, axis=0).T)
            sp += list(d[d > 0]); zero += list(d == 0)
            B = A[np.r_[True, d > 0]]
            if len(B) >= 3:
                v = np.diff(B, axis=0); ang = np.arctan2(v[:, 1], v[:, 0]); t = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi); turn += list(t)
            if lab.get(k) in dash and len(P) >= 2: dash[lab[k]].append((is_flat(P, cH), bar_conf(P, cH)))
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [10, 50, 90])]
    out = dict(turn_mean_deg=round(float(np.degrees(np.mean(turn))), 2), spacing_p10_50_90=q(sp), spacing_mean=round(float(np.mean(sp)), 3), zero_step_share=round(float(np.mean(zero)), 3))
    for k, v in dash.items():
        if v: out[k] = dict(n=len(v), isFlat=round(float(np.mean([a for a, _ in v])), 3), barConf=round(float(np.mean([b for _, b in v])), 3))
    return out
def real():
    sp = json.load(open(SP + '/hwdata/round2/split.json')); TR = set(sp['train'])
    items = {x['id']: x for f in glob.glob(SP + '/trainer_data2/samples/__WRITER__/ex/*.json') for x in json.load(open(f))['items'] if x['id'] in TR and not x['discarded'] and not x['warmup']}
    out = []
    for r in json.load(open('align_train.json')):
        if not r['ok']: continue
        it = items[r['id']]; rl, _ = roles(it['promptText']); ne = [i for i, s in enumerate(it['strokes']) if s]; m = {o: n for n, o in enumerate(ne)}
        lab = {}
        for (tok, role), q in zip(rl, r['per']):
            L = 'bar' if role == 'bar' else tok
            for k in q['strokes']: lab[m[k]] = L
        out.append(([s for s in it['strokes'] if s], lab))
    return out
def synth(modname, n=300, seed=9):
    G = importlib.import_module(modname); r = random.Random(seed); out = []
    for i in range(n):
        src, tpl = G.prompt(r); e = G.Ex(r).build(src); S = e.finish()
        toks = [('bar' if ro == 'bar' else t.lstrip('^')) for t, ro in roles(src)[0]]; arrows = roles(src)[1]
        # parts are written in token order, with an arrow part (if drawn) where '>' sits
        labs = []; drawn = len(e.parts) - len(toks); ti = 0
        for pi in range(len(e.parts)):
            if drawn and ti in arrows and (not labs or labs[-1] != '>') and len(e.parts) - pi > len(toks) - ti: labs.append('>'); drawn -= 1; continue
            labs.append(toks[ti]); ti += 1
        lab = {}; k = 0
        for part, L in zip(e.parts, labs):
            for _ in part: lab[k] = L; k += 1
        out.append((S, lab))
    return out
if __name__ == '__main__':
    res = {'real_TRAIN': stats(real())}
    for m in sys.argv[1:]: res[m] = stats(synth(m))
    print(json.dumps(res, indent=0)); json.dump(res, open('texture_metrics.json', 'w'), indent=1)
