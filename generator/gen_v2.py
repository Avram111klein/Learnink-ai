"""v2 (after the blind look-test): powers anchored top-right of the base box with the owner's raise/scale and no overlap;
light touching only (no deep overlaps); every sample normalised to its class height and baseline (operators mid-line);
one sample per letter per exercise; malformed bank samples dropped by an outlier check against the class centroid.

Synthetic exercises in the owner's handwriting, composed from real TRAIN-only symbol samples (bank.json.gz) with the layout
statistics of style.json. Output = the trainer's item format + sources/style.

usage: python3 gen.py N out.jsonl.gz [seed]
"""
import sys, json, gzip, math, random, time
import numpy as np
from ml import roles, target, tokens

HERE = __file__.rsplit('/', 1)[0] if '/' in __file__ else '.'
BANK = json.load(gzip.open(HERE + '/bank.json.gz', 'rt'))
STY = json.load(open(HERE + '/style.json'))
SYM = BANK['symbols']
OPS = set('+-=÷')

def _raster(s, n=20):
    """sample -> n x n bitmap (bbox-fitted, aspect kept), for the outlier check"""
    from PIL import Image, ImageDraw
    P = [np.array([p[:2] for p in st], float) for st in s['s']]; A = np.vstack(P); lo = A.min(0); sz = max(np.ptp(A[:, 0]), np.ptp(A[:, 1]), 1e-3)
    im = Image.new('L', (n * 2, n * 2), 0); d = ImageDraw.Draw(im); off = (n * 2 - 2) * (1 - np.ptp(A, 0) / sz) / 2
    for q in P:
        q = (q - lo) / sz * (n * 2 - 2) + 1 + off; pts = [tuple(v) for v in q]
        d.line(pts if len(pts) > 1 else pts * 2, fill=255, width=3)
    return np.asarray(im.resize((n, n), Image.BILINEAR), float).ravel() / 255

def _clean(v):
    """drop outliers: bitmap distance to the class centroid or aspect ratio beyond median + 3.5 MAD (classes with >= 8 samples)"""
    if len(v) < 8: return v, []
    X = np.array([_raster(s) for s in v]); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9; c = X.mean(0); d = np.linalg.norm(X - c, axis=1)
    ar = np.log(np.array([(s['w'] + 0.02) / (s['h'] + 0.02) for s in v]))
    bad = lambda z: np.abs(z - np.median(z)) > 3.5 * (np.median(np.abs(z - np.median(z))) * 1.4826 + 1e-6)
    out = (d > np.median(d) + 3.5 * np.median(np.abs(d - np.median(d))) * 1.4826) | bad(ar)
    return [s for s, o in zip(v, out) if not o], [s for s, o in zip(v, out) if o]
DROPPED = {}
for _c in list(SYM):
    if _c == 'bar': continue
    SYM[_c], _d = _clean(SYM[_c])
    if _d: DROPPED[_c] = len(_d)
MAIN = {c: [s for s in v if s['role'] == 'main'] for c, v in SYM.items()}
POW = {c: [s for s in v if s['role'] == 'pow'] for c, v in SYM.items()}
BARS = SYM['bar']; ARROWS = BANK['arrow']
GAPS = {k: v['list'] for k, v in STY['gaps'].items()}
PAIR = STY['pair_gaps']
GALL = [x for k, v in GAPS.items() if k != 'arrow' for x in v]
FRACS = STY['fraction']['list']; FGAPS = [g for f in FRACS for g in f['num_gaps'] + f['den_gaps']] or [0.2]
POWS = STY['power']['list']; PAUSES = STY['timing']['pause_list']
DL, SL, RL = STY['D_list'], STY['baseline']['slope_list'], STY['baseline']['resid_list']
XY = STY['start']['xy_list']; CANV = [c for c, n in STY['canvas'] for _ in range(n)]
KROT = STY['glyph_rotation_follows_baseline']['k'] * float(__import__('os').environ.get('GEN_KROT', '1'))
TOUCH = float(__import__('os').environ.get('GEN_TOUCH', '0'))
HARD_RATE = float(__import__('os').environ.get('GEN_HARD', '0.3'))
TRAIN = set(STY['train_ids_used'])
# class norms (from the kept main-row samples): height, baseline offset of the bottom, centre (operators), width (flat marks)
NORM = {}
for _c, _v in MAIN.items():
    if len(_v) >= 3:
        NORM[_c] = dict(h=float(np.median([s['h'] for s in _v])), bot=float(np.median([s['bot'] for s in _v])),
                        cy=float(np.median([(s['top'] + s['bot']) / 2 for s in _v])), w=float(np.median([s['w'] for s in _v])))
POW2 = [p for p in POWS if p.get('rel_top') is not None]
PSCALE = float(np.median([p['scale'] for p in POWS]))

def kind(c):
    return 'f' if c == 'frac' else 'd' if c.isdigit() else 'l' if c.isalpha() else 'o' if c == '(' else 'c' if c == ')' else 'p' if c == '.' else 'op'

class Ex:
    def __init__(self, rng):
        self.r = rng; r = rng
        self.D = r.choice(DL) * r.uniform(0.95, 1.05)
        self.slope = r.choice(SL) + r.gauss(0, 0.02)
        self.x0, self.y0 = [v + r.gauss(0, 8) for v in r.choice(XY)]
        self.cw, self.ch = r.choice(CANV)
        self.speed = r.uniform(0.85, 1.15); self.pk = r.uniform(0.9, 1.1)
        self.pen_w = round(r.uniform(1.8, 3.2), 2)
        self.rot0 = r.uniform(-1.5, 1.5); self.shear0 = r.uniform(-0.04, 0.04)
        self.parts = []; self.sources = set(); self.t = 0.0; self.fixed = {}; self.last_ink = None; self.touch = False

    def base(self, x): return self.y0 + self.slope * (x - self.x0)

    def pick(self, pool):
        s = self.r.choice(pool); self.sources.add(s['src']); return s

    def xform(self, s, scale, xscale=1.0, norm=None, pow_h=None):
        """sample strokes (D units, x from left edge, y from its baseline) -> augmented, in px relative to (left, baseline).
        norm=class: rescale to the class's median height (width for flat marks) and move to its median baseline offset
        (centre for operators), then only small jitter. pow_h: rescale to this height, bottom at 0."""
        r = self.r; D = self.D * scale
        ang = math.radians(r.uniform(-4, 4) * 0.6 + self.rot0) + KROT * (math.atan(self.slope) - math.atan(s['slope']))
        sh = self.shear0 + r.uniform(-0.06, 0.06); k = min(1.08, max(0.92, 1 + r.gauss(0, 0.04)))
        f, dy = 1.0, 0.0
        if pow_h is not None:
            f = pow_h / max(s['h'], 0.05); dy = -s['bot'] * f
        elif norm in NORM and norm != '.':
            N = NORM[norm]
            f = N['w'] / max(s['w'], 0.05) if norm in '-=' else N['h'] / max(s['h'], 0.05)
            f = min(2.0, max(0.5, f))
            if norm in OPS: dy = N['cy'] + r.gauss(0, 0.03) - (s['top'] + s['bot']) / 2 * f
            else: dy = N['bot'] + r.gauss(0, 0.03) - s['bot'] * f
        P = [(np.array([[p[0], p[1]] for p in st], float) * f + [0, dy]) * [k * xscale, k] for st in s['s']]
        allp = np.vstack(P); c = allp.mean(0)
        ca, sa = math.cos(ang), math.sin(ang)
        out = []
        for st, raw in zip(P, s['s']):
            q = st - c; q = np.c_[q[:, 0] + sh * q[:, 1], q[:, 1]]
            q = np.c_[ca * q[:, 0] - sa * q[:, 1], sa * q[:, 0] + ca * q[:, 1]] + c
            n = len(q)
            if n > 3:   # smooth low-frequency wobble + tiny tremor
                u = np.linspace(0, 1, n); amp = r.uniform(0, 0.012)
                q = q + amp * np.c_[np.sin(2 * math.pi * (u * r.uniform(0.5, 1.5) + r.random())), np.sin(2 * math.pi * (u * r.uniform(0.5, 1.5) + r.random()))]
            q = q * D
            out.append([[q[i, 0], q[i, 1], raw[i][2], raw[i][3]] for i in range(n)])
        return out

    def box(self, strokes):
        a = np.array([p[:2] for st in strokes for p in st]); return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()

    def put(self, strokes, dx, dy):
        """shift into place, stamp times (writing order = call order), record"""
        r = self.r; out = []
        if self.parts: self.t += max(40, r.choice(PAUSES)) * self.speed
        t0 = self.t
        for st in strokes:
            out.append([[p[0] + dx, p[1] + dy, t0 + p[2] * self.speed, p[3]] for p in st])
        self.t = max(p[2] for st in out for p in st)
        self.parts.append(out); return self.box(out)

    def glyph(self, c, role='main', scale=1.0):
        """a sample for token c -> strokes rel. to left edge/baseline (normalised). Letters: one sample per exercise."""
        if role == 'pow':
            pool = POW.get(c) if len(POW.get(c, [])) >= 3 and self.r.random() < 0.7 else (MAIN.get(c) or SYM[c])
            ph = PSCALE * self.r.uniform(0.92, 1.08)
            return self.xform(self.pick(pool), scale, pow_h=ph), 'pow'
        pool = MAIN.get(c) or SYM[c]
        if c.isalpha():
            if c not in self.fixed: self.fixed[c] = self.pick(pool)
            s = self.fixed[c]
        else: s = self.pick(pool)
        return self.xform(s, scale, norm=c), 'main'

    def gap(self, a, b):
        r = self.r
        self.touch = False
        if r.random() < TOUCH: self.touch = True; return r.uniform(-0.03, 0.03)
        L = PAIR.get(a + '|' + b) if len(PAIR.get(a + '|' + b, [])) >= 8 else GAPS.get(kind(a.rstrip('^')) + '>' + kind(b)) or GALL
        g = r.choice(L if len(L) >= 5 else GALL)
        # the owner's touching neighbours (gap < 0.03 D, ~15 %) are kept as light touching only; never a deep overlap
        if g < 0.03: self.touch = True; return r.uniform(-0.03, 0.03)
        return g

    def ink_shift(self, placed, touch):
        """how far to push a unit right so its ink does not run into the previous unit's ink: light touching
        (a few points within 0.025 D) only where the owner's gap said 'touch', otherwise a clear 0.05 D"""
        if self.last_ink is None: return 0.0
        A = np.array([[p[0] + dx, p[1] + dy] for st, dx, dy in placed for s in st for p in s]); Bk = self.last_ink
        u = self.D; sh = 0.0
        Bk = Bk[Bk[:, 0] > A[:, 0].min() - 0.2 * u]
        if not len(Bk): return 0.0
        for _ in range(60):
            d = np.sqrt(((A[:, None, :] + [sh, 0] - Bk[None, :, :]) ** 2).sum(-1)).min(1)
            if (touch and (d < 0.025 * u).sum() <= 4 and d.min() > 0.0) or (not touch and d.min() >= 0.05 * u): break
            sh += 0.02 * u
        return sh

    def emit(self, placed, touch):
        sh = self.ink_shift(placed, touch)
        for st, dx, dy in placed: self.put(st, dx + sh, dy)
        self.last_ink = np.array([[p[0] + dx + sh, p[1] + dy] for st, dx, dy in placed for s in st for p in s])
        return sh

    def row(self, toks, x, scale, place):
        """lay out a flat token list (with ^ powers) starting at x; place(strokes)->(dy) decides vertical; returns x1, list of (strokes, dx)"""
        items, prev = [], None
        for tok, role in toks:
            if tok.startswith('^'):
                # anchored to the top-right of the base's box: the owner's (dx, raise) pair; no overlap with the base
                c = tok[1:]; pe = self.r.choice(POW2); u = self.D * scale
                st, _ = self.glyph(c, 'pow', scale); b = self.box(st)
                bi = max(i for i, it in enumerate(items) if it[2][0] != 'rel'); bst, bdx, (bm, boff) = items[bi]; B = self.box(bst)
                Bx0, By0, Bx1, By1 = B[0] + bdx, B[1] + boff, B[2] + bdx, B[3] + boff
                px0 = Bx1 + min(0.15, max(-0.35, pe['dx'])) * u; py1 = By0 + min(0.15, max(-0.35, pe['rel_top'])) * u
                pw, ph = b[2] - b[0], b[3] - b[1]; m = 0.04 * u
                if px0 < Bx1 + m and py1 > By0 - m:          # would overlap the base: lift it, or move right if that is shorter
                    if py1 - (By0 - m) <= (Bx1 + m) - px0 + 0.1 * u: py1 = By0 - m
                    else: px0 = Bx1 + m
                dx = px0 - b[0]; items.append((st, dx, ('rel', bi, boff + py1 - b[3])))
                x = max(x, dx + b[2]); continue
            c = tok
            st, _ = self.glyph(c, 'main', scale); b = self.box(st)
            if prev is not None: x += self.gap(prev, c) * self.D * scale
            dx = x - b[0]; items.append((st, dx, ('own', 0.0))); x = dx + b[2]; prev = c
        return x, items

    def build(self, src):
        r = self.r; rl, arrows = roles(src)
        # group into units: plain tokens, powers attach to the unit before, fractions are one unit
        units, i = [], 0
        while i < len(rl):
            tok, role = rl[i]
            if role in ('num', 'bar', 'den', 'numpow', 'denpow'):
                j = i
                while j < len(rl) and rl[j][1] in ('num', 'bar', 'den', 'numpow', 'denpow'): j += 1
                units.append(('frac', rl[i:j], i)); i = j; continue
            if role == 'pow' and units: units[-1][1].append((tok, role)); i += 1; continue
            units.append((tok, [(tok, role)], i)); i += 1
        x = self.x0; prevc = None
        for c, toks, at in units:
            if prevc is not None:
                if at in arrows:
                    g = r.choice(GAPS['arrow']) * self.D
                    if ARROWS and r.random() < STY['arrow']['n_drawn'] / max(1, STY['arrow']['n_prompts_with_arrow']):
                        a = self.pick(ARROWS); st = self.xform(dict(a, slope=self.slope), 1.0); b = self.box(st)
                        w = b[2] - b[0]; g = max(g, w + 0.5 * self.D); ax = x + (g - w) / 2 - b[0]
                        self.put(st, ax, self.base(ax + w / 2))
                    x += g
                else:
                    x += self.gap(prevc, 'frac' if c == 'frac' else c) * self.D
            touch = self.touch if prevc is not None and at not in arrows else False
            if c == 'frac':
                x1, placed = self.frac(toks, x)
            else:
                x1, items = self.row(toks, x, 1.0, None); placed = []
                for st, dx, (m, *rest) in items:
                    b = self.box(st); cx = dx + (b[0] + b[2]) / 2
                    dy = placed[rest[0]][2] + rest[1] if m == 'rel' else self.base(cx) + r.gauss(0, 0.02) * self.D
                    placed.append((st, dx, dy))
            x = x1 + self.emit(placed, touch)
            prevc = 'frac' if c == 'frac' else c + ('^' if any(t.startswith('^') for t, _ in toks) else '')
        return self

    def frac(self, toks, x):
        r = self.r; f = r.choice(FRACS); fs = f['scale'] * r.uniform(0.95, 1.05)
        nu = [(t, 'main' if ro == 'num' else 'pow') for t, ro in toks if ro in ('num', 'numpow')]
        de = [(t, 'main' if ro == 'den' else 'pow') for t, ro in toks if ro in ('den', 'denpow')]
        global FGAPS
        def fgap(a, b):
            v = r.choice(FGAPS); return r.uniform(-0.03, 0.03) if v < 0.03 else v
        saveG = self.gap; self.gap = fgap
        nx1, ni = self.row(nu, 0.0, fs, None); dx1, di = self.row(de, 0.0, fs, None); self.gap = saveG
        nw = nx1 - min(dx + self.box(st)[0] for st, dx, _ in ni); dw = dx1 - min(dx + self.box(st)[0] for st, dx, _ in di)
        over = max(-0.05, f['over'] + r.gauss(0, 0.05)) * self.D
        W = max(nw, dw) + over
        bs = self.pick(BARS); bst = self.xform(bs, 1.0); bb = self.box(bst); bw = max(bb[2] - bb[0], 1e-3)
        bst = [[[bb[0] + (p[0] - bb[0]) * W / bw, p[1], p[2], p[3]] for p in s] for s in bst]
        bcx = x + W / 2; bcy = self.base(bcx) + f['bar_y'] * self.D
        def place(items, width, cxo, anchor, gap):
            # row centred on the bar (+ real offset); anchored by its lowest bottom (numerator) or highest top (denominator)
            x0 = min(dx + self.box(st)[0] for st, dx, _ in items); shift = bcx + cxo * self.D - width / 2 - x0
            offs = []
            for st, dx, (m, *rest) in items:
                off = rest[1] if m == 'rel' else 0.0
                b = self.box(st); offs.append(b[3] + off if anchor == 'bottom' else b[1] + off)
            ref = max(offs) if anchor == 'bottom' else min(offs)
            rb = bcy - gap * self.D - ref if anchor == 'bottom' else bcy + gap * self.D - ref
            res = []
            for st, dx, (m, *rest) in items:
                dy = res[rest[0]][2] + rest[1] if m == 'rel' else rb + r.gauss(0, 0.015) * self.D
                res.append((st, dx + shift, dy))
            return res
        N = place(ni, nw, f['num_dx'], 'bottom', f['num_gap']); Dn = place(di, dw, f['den_dx'], 'top', f['den_gap'])
        bb = self.box(bst); placed = N + [(bst, bcx - (bb[0] + bb[2]) / 2, bcy - (bb[1] + bb[3]) / 2)] + Dn
        return max(p[0] + dx for st, dx, dy in placed for s in st for p in s), placed

    def finish(self):
        S = [st for part in self.parts for st in part]
        A = np.array([p[:2] for st in S for p in st])
        x0, y0, x1, y1 = A[:, 0].min(), A[:, 1].min(), A[:, 0].max(), A[:, 1].max()
        f = min(1.0, (self.cw - 12 - x0) / max(1, x1 - x0) if x1 > self.cw - 12 else 1.0)
        out = []
        for st in S:
            out.append([[round(x0 + (p[0] - x0) * f), round(y0 + (p[1] - y0) * f), round(p[2]), int(min(100, max(1, round(p[3] * self.pk))))] for p in st])
        A = np.array([p[:2] for st in out for p in st]); dy = 0
        if A[:, 1].min() < 6: dy = 6 - A[:, 1].min()
        elif A[:, 1].max() > self.ch - 6: dy = max(6 - A[:, 1].min(), self.ch - 6 - A[:, 1].max())
        if dy: out = [[[p[0], p[1] + int(dy), p[2], p[3]] for p in st] for st in out]
        t0 = out[0][0][2]; out = [[[p[0], p[1], p[2] - t0, p[3]] for p in st] for st in out]
        self.fit = round(f, 3); return out

# ---------------- exercise templates (trainer TPL, mirrored) + harder mixes ----------------
LET = list('xabcdkmnptysz')
def rnd(r, a, b): return r.randint(a, b)
def coef(k, v): return ('' if k == 1 else '-' if k == -1 else str(k)) + v
def fmtH(h): s = ('%.2f' % (h / 100)).rstrip('0').rstrip('.'); return s
def T_add(r): a, b = rnd(r, 12, 89), rnd(r, 11, 89); return f'{a}+{b}={a+b}'
def T_sub(r): a = rnd(r, 30, 99); b = rnd(r, 11, a - 3); return f'{a}-{b}={a-b}'
def T_neg(r): a = rnd(r, 2, 19); b = rnd(r, a + 1, 40); return f'-{a}+{b}={b-a}'
def T_div(r): b, q = rnd(r, 2, 9), rnd(r, 3, 15); return f'{b*q}÷{b}={q}'
def T_slash(r): b, q = rnd(r, 2, 9), rnd(r, 2, 12); return f'{b*q}/{b}={q}'
def T_dec(r):
    while True:
        A = rnd(r, 12, 95) * 10
        if A % 100: break
    B = rnd(r, 105, 495)
    if r.random() < .5: return f'{fmtH(A)}+{fmtH(B)}={fmtH(A+B)}'
    return f'{fmtH(A)}-{fmtH(B)}={fmtH(A-B)}' if A > B else f'{fmtH(B)}-{fmtH(A)}={fmtH(B-A)}'
def T_frac(r):
    while True:
        D = [2, 3, 4, 5, 6, 8, 10]; b, d = r.choice(D), r.choice(D); a, c = rnd(r, 1, b - 1), rnd(r, 1, d - 1)
        if math.gcd(a, b) != 1 or math.gcd(c, d) != 1: continue
        n, m, op = a * d + c * b, b * d, '+'
        if n >= m:
            n, op = a * d - c * b, '-'
            if n <= 0: continue
        g = math.gcd(n, m); n //= g; m //= g
        if m == 1: continue
        return f'[{a}/{b}]{op}[{c}/{d}]=[{n}/{m}]'
def T_pow(r): a, b = rnd(r, 2, 9), rnd(r, 2, 9); return f'{a}^2+{b}^2={a*a+b*b}'
def T_eqParen(r, v=None):
    v = v or r.choice(LET); k, c = rnd(r, 2, 6), rnd(r, 1, 9); s = rnd(r, c + 1, c + 9)
    return f'{k}({v}+{c})={k*(s+c)} > {v}={s}' if r.random() < .4 else f'{k}({v}-{c})={k*(s-c)} > {v}={s}'
def T_lin(r, v=None):
    v = v or r.choice(LET); a, b, s = rnd(r, 2, 9), rnd(r, 1, 20), rnd(r, 1, 12)
    if r.random() < .5 or a * s <= b: return f'{a}{v}+{b}={a*s+b} > {v}={s}'
    return f'{a}{v}-{b}={a*s-b} > {v}={s}'
def T_two(r, v=None):
    v1 = v or r.choice(LET); v2 = r.choice([l for l in LET if l != v1]); a, b, x, y = [rnd(r, *z) for z in ((2, 9), (2, 9), (1, 9), (1, 9))]
    return f'{a}{v1}+{b}{v2}={a*x+b*y}'
def T_diffsq(r, v=None): v = v or r.choice(LET); c = rnd(r, 1, 9); return f'{v}^2-{c*c}=({v}-{c})({v}+{c})'
def T_trinom(r, v=None): v = v or r.choice(LET); p, q = rnd(r, 1, 9), rnd(r, 1, 9); return f'({v}+{p})({v}+{q})={v}^2+{coef(p+q, v)}+{p*q}'
def T_expand(r, v=None):
    v = v or r.choice(LET); k, c = rnd(r, 2, 9), rnd(r, 1, 9)
    return f'{k}({v}+{c})={k}{v}+{k*c}' if r.random() < .5 else f'{k}({v}-{c})={k}{v}-{k*c}'
def T_factor(r, v=None): v = v or r.choice(LET); c = rnd(r, 2, 9); return f'{v}^2+{c}{v}={v}({v}+{c})'
def T_fracEq(r, v=None):
    v = v or r.choice(LET); d, q = rnd(r, 2, 9), rnd(r, 2, 9)
    return f'[{v}/{d}]={q} > {v}={d*q}' if r.random() < .5 else f'[{v}+{q}/{d}]={q+1} > {v}={d*(q+1)-q}'
def T_divLet(r, v=None): v = v or r.choice(LET); k, c = rnd(r, 2, 9), rnd(r, 2, 9); return f'{k*c}{v}÷{k}={c}{v}'
def T_mono(r, v=None):
    v1 = v or r.choice(LET); v2 = r.choice([l for l in LET if l != v1]); a = rnd(r, 2, 9)
    return f'{a}{v1}^2{v2}-{v1}^2{v2}={coef(a-1, v1+"^2"+v2)}'
TPL = dict(add=T_add, sub=T_sub, neg=T_neg, div=T_div, slash=T_slash, dec=T_dec, frac=T_frac, pow=T_pow, eqParen=T_eqParen, lin=T_lin, two=T_two,
           diffsq=T_diffsq, trinom=T_trinom, expand=T_expand, factor=T_factor, fracEq=T_fracEq, divLet=T_divLet, mono=T_mono)
# harder mixes (not in the trainer): longer, nested parens, letter fractions, powers after letters/parens, ÷, decimals
def H_long(r):
    v = r.choice(LET); a, b, c, d = rnd(r, 2, 9), rnd(r, 1, 9), rnd(r, 2, 9), rnd(r, 1, 9)
    return f'{a}({v}+{b})-{c}({v}-{d})={coef(a-c, v) if a != c else ""}+{a*b+c*d}'.replace('=+', '=')
def H_nested(r):
    v = r.choice(LET); a, b, c, d = rnd(r, 2, 5), rnd(r, 2, 5), rnd(r, 1, 9), rnd(r, 1, 9)
    return f'{a}({b}({v}+{c})-{d})={a*b}{v}+{a*b*c-a*d}' if a * b * c - a * d >= 0 else f'{a}({b}({v}+{c})-{d})={a*b}{v}{a*b*c-a*d}'
def H_letfrac(r):
    v, w = r.sample(LET, 2); d = rnd(r, 2, 9); k = r.randint(0, 2)
    if k == 0: return f'[{v}/{d}]+[{w}/{d}]=[{v}+{w}/{d}]'
    if k == 1: a, b = rnd(r, 1, 9), rnd(r, 1, 9); return f'[{v}+{a}/{v}-{b}]=2 > {v}={a+2*b}'
    q = rnd(r, 2, 9); return f'[{d}{v}/{q}]={d} > {v}={q}'
def H_powpar(r):
    v = r.choice(LET); c = rnd(r, 1, 9); k = r.randint(0, 2)
    if k == 0: return f'({v}+{c})^2={v}^2+{2*c}{v}+{c*c}'
    if k == 1: return f'({v}-{c})^2={v}^2-{2*c}{v}+{c*c}'
    w = r.choice([l for l in LET if l != v]); a = rnd(r, 2, 6); return f'{a}{v}^3{w}^2÷{v}{w}={a}{v}^2{w}'
def H_divmix(r):
    v = r.choice(LET); k, c, e = rnd(r, 2, 9), rnd(r, 2, 9), rnd(r, 1, 9)
    return f'({k*c}{v}+{k*e})÷{k}={c}{v}+{e}'
def H_decalg(r):
    v = r.choice(LET); a = rnd(r, 11, 49) / 10; s = rnd(r, 2, 9); b = rnd(r, 1, 30) / 10
    g = lambda z: ('%.1f' % z).rstrip('0').rstrip('.')
    return f'{g(a)}{v}+{g(b)}={g(a*s+b)} > {v}={s}' if r.random() < .5 else f'{g(a)}({v}+{g(b)})={g(a*(s+b))} > {v}={s}'
HARD = dict(long=H_long, nested=H_nested, letfrac=H_letfrac, powpar=H_powpar, divmix=H_divmix, decalg=H_decalg)
TPL_W = {'trinom': 34, 'factor': 33, 'diffsq': 33, 'divLet': 31, 'eqParen': 23, 'expand': 21, 'div': 21, 'two': 18, 'slash': 12, 'fracEq': 11, 'add': 10,
         'mono': 9, 'pow': 8, 'lin': 7, 'frac': 6, 'sub': 4, 'dec': 3, 'neg': 2}   # the owner's TRAIN template mix
for k in TPL_W: TPL_W[k] = max(TPL_W[k], 4)   # keep rare templates visible

def prompt(r, hard_rate=None):
    hard_rate = HARD_RATE if hard_rate is None else hard_rate
    if r.random() < hard_rate:
        name = r.choice(list(HARD)); return HARD[name](r), 'hard:' + name
    names = list(TPL_W); name = r.choices(names, [TPL_W[n] for n in names])[0]
    return TPL[name](r), name

def make(i, r, src=None, tpl=None):
    if src is None: src, tpl = prompt(r)
    e = Ex(r).build(src); strokes = e.finish()
    assert e.sources <= TRAIN, 'non-TRAIN source'
    tt = target(src)
    return dict(id=f'SYN-{i:06d}', promptText=src, targetText=tt, targetTokens=tokens(tt), template=tpl, strokes=strokes, sources=sorted(e.sources),
                style=dict(D=round(e.D, 2), slope=round(e.slope, 4), pen_w=e.pen_w, speed=round(e.speed, 3), fit=e.fit, canvas=[e.cw, e.ch]),
                discarded=False, warmup=False, synthetic=True)

if __name__ == '__main__':
    N, out = int(sys.argv[1]), sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    r = random.Random(seed); t = time.time()
    with gzip.open(out, 'wt') as f:
        for i in range(N): f.write(json.dumps(make(i, r), separators=(',', ':')) + '\n')
    dt = time.time() - t; print(f'{N} exercises in {dt:.1f}s = {N/dt:.1f}/s')
