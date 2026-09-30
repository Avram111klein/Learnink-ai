"""Build the FROZEN algebra test set (never used for training).

LICENCE: CROHME is CC BY-NC-SA 4.0 (non-commercial). Use it for EVALUATION ONLY; a model trained on it
must not ship in the paid app (the r1 run that used CROHME train is a reference, NOT for the app).

Sources
  real_symbols.json   symbols cut from CROHME 2014/2016/2019 TEST splits (other writers than CROHME train)
  real_exprs.json     the whole linear CROHME test expressions those symbols came from
  synth_symbols.json  glyphs.py with STYLE_TEST and seed 90210 (training uses STYLE_TRAIN, other seeds)
  synth_exprs.json    composed algebra answers ("2x+3=7", "x^2-1", "(a+b)/2", ...): digits/operators are
                      real CROHME-test glyphs, letters are STYLE_TEST glyphs; layout jitter seed 4242
  synth_exprs_pen.json (added in part 2, PEN=<digits_strokes.json>): the same answer templates but with
                      continuous pen-like strokes only — digits = MNIST *test* digits traced to strokes (the
                      400 of digits_strokes.json, never trained on), letters/operators = STYLE_TEST glyphs;
                      layout seed 4343. Added because CROHME bitmap->skeleton strokes are fragmented at
                      junctions, which hw.js segment() (made for live S-Pen strokes) mis-groups.
usage: python3 make_frozen.py <hwdata dir with crohme_20xx.json> <out dir>
       PEN=digits_strokes.json python3 make_frozen.py <hwdata> <out dir>   # writes only synth_exprs_pen.json
"""
import sys, json, os, hashlib
import numpy as np
from glyphs import Gen, STYLE_TEST

NEW = list("abcdkmnptysz") + ["÷"]
OLD_SYN = ["+", "x", "/", "(", ")"]
PER_CLASS = 320

src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
real = {"sym": [], "expr": []}
for sp in ("2014", "2016", "2019"):
    d = json.load(open(f"{src}/crohme_{sp}.json"))
    for s in d["sym"]:
        s["split"] = sp
    for e in d["expr"]:
        e["split"] = sp
    real["sym"] += d["sym"]; real["expr"] += d["expr"]

g = Gen(90210, STYLE_TEST)
r4 = lambda st: [[[round(float(x), 4), round(float(y), 4)] for x, y in s] for s in st]
synth = [{"c": c, "s": r4(g.strokes(c)), "lw": round(g.lw(), 3)} for c in NEW + OLD_SYN for _ in range(PER_CLASS)]

# ---- composed expressions ----
PEN = os.environ.get("PEN")
R = np.random.default_rng(4343 if PEN else 4242)
pool = {}
if PEN:
    for d in json.load(open(PEN)):
        pool.setdefault(str(d["d"]), []).append(d["s"])
else:
    for s in real["sym"]:
        pool.setdefault(s["c"], []).append(s["s"])
TALL, DESC = set("bdkt0123456789()"), set("py")


def glyph(c):
    if c in pool and (c not in "abcdkmnptysz" or R.random() < 0.5):
        st = pool[c][R.integers(len(pool[c]))]      # real handwriting (letters: half real, half synthetic)
    else:
        st = g.strokes(c)
    return [np.array(s, float) for s in st]


def place(st, x, H, kind):
    P = np.concatenate(st); x0, y0 = P.min(0); x1, y1 = P.max(0)
    w, h = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
    if kind == "op":
        tgt = 0.45 * H; sc = tgt / max(w, h); top = 0.5 * H - h * sc / 2
    elif kind == "flat":
        sc = 0.45 * H / w; top = 0.5 * H - h * sc / 2
    elif kind == "sup":
        sc = 0.5 * H / h; top = -0.25 * H
    elif kind == "paren":
        sc = 1.15 * H / h; top = -0.08 * H
    elif kind == "desc":
        sc = 0.95 * H / h; top = 0.3 * H
    elif kind == "small":
        sc = 0.62 * H / h; top = 0.38 * H
    else:
        sc = H / h; top = 0.0
    sc *= R.uniform(0.9, 1.1)
    top += R.uniform(-0.06, 0.06) * H
    return [np.stack([(s[:, 0] - x0) * sc + x, (s[:, 1] - y0) * sc + top], 1) for s in st], x + w * sc


def dash(x, H, y=0.5):
    L = 0.45 * H * R.uniform(0.8, 1.2)
    return [np.stack([np.linspace(x, x + L, 8), y * H + np.linspace(0, R.uniform(-.06, .06) * H, 8)], 1)], x + L


def layout(tokens, H=40.0, x=0.0):
    """tokens: list of chars; '^' makes the next token a superscript. returns strokes, width."""
    S = []; sup = False
    for t in tokens:
        if t == "^":
            sup = True; continue
        gap = H * R.uniform(0.12, 0.3)
        if t == "-":
            st, x2 = dash(x + gap, H)
        elif t == "=":
            a, _ = dash(x + gap, H, 0.38); b, x2 = dash(x + gap, H, 0.66); st = a + b
        elif t == ".":
            st = [np.array([[x + gap, 0.95 * H], [x + gap + 1.5, 0.97 * H]])]; x2 = x + gap + 2
        else:
            kind = "sup" if sup else "op" if t in "+÷×" else "paren" if t in "()" else "desc" if t in DESC else \
                "small" if t in "acemnrsuvwxz" else "tall"
            st, x2 = place(glyph(t), x + gap, H, kind)
        S += st; x = x2; sup = False
    return S, x


def frac(num, den, H=40.0):
    n, wn = layout(num, H); d, wd = layout(den, H)
    W = max(wn, wd) + 0.3 * H
    n = [s + [(W - wn) / 2, -1.25 * H] for s in n]
    d = [s + [(W - wd) / 2, 0.45 * H] for s in d]
    bar = [np.stack([np.linspace(0, W, 10), 0.1 * H + np.linspace(0, R.uniform(-.05, .05) * H, 10)], 1)]
    return n + bar + d, W


def num():
    return str(R.integers(1, 10)) if R.random() < .6 else str(R.integers(10, 100))


def rv():
    return str(R.choice(list("xyabnktcmpdsz"), p=np.array([8, 5, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1]) / 31))


def make():
    v, w = rv(), rv()
    while w == v:
        w = rv()
    k = R.integers(12)
    if k == 0: e = f"{num()}{v}+{num()}={num()}"
    elif k == 1: e = f"{v}^2-{num()}"
    elif k == 2: return f"({v}+{w})/{num()}", ("frac", f"({v}+{w})", num())
    elif k == 3: e = f"{num()}({v}-{num()})"
    elif k == 4: e = f"{v}^2+{num()}{v}+{num()}"
    elif k == 5: e = f"{v}={num()}"
    elif k == 6: e = f"-{num()}{v}"
    elif k == 7: e = f"{v}^2{w}"
    elif k == 8: e = f"{num()}{v}-{num()}{w}"
    elif k == 9: e = f"{v}={num()}.{R.integers(1, 10)}"
    elif k == 10: e = f"({v}-{num()})({v}+{num()})"
    else: return None, None
    return e, None


exprs = []
while len(exprs) < 600:
    e, fr = make()
    if e is None:
        n, d = num(), num()
        v = rv()
        st, _ = frac(list(f"{n}{v}"), list(d)); e = f"{n}{v}/{d}"
    elif fr:
        _, a, b = fr
        st, _ = frac(list(a), list(b)); e = f"{a}/{b}"
    else:
        st, _ = layout(list(e))
    vars_ = sorted(set(c for c in e if c.isalpha()))
    exprs.append({"text": e, "vars": vars_, "s": r4(st)})

if PEN:
    json.dump(exprs, open(f"{out}/synth_exprs_pen.json", "w"))
    print("pen exprs", len(exprs)); sys.exit(0)
json.dump(real["sym"], open(f"{out}/real_symbols.json", "w"))
json.dump(real["expr"], open(f"{out}/real_exprs.json", "w"))
json.dump(synth, open(f"{out}/synth_symbols.json", "w"))
json.dump(exprs, open(f"{out}/synth_exprs.json", "w"))
with open(f"{out}/SHA256SUMS", "w") as f:
    for n in sorted(os.listdir(out)):
        if n.endswith(".json"):
            f.write(hashlib.sha256(open(f"{out}/{n}", "rb").read()).hexdigest() + "  " + n + "\n")
from collections import Counter
print("real sym", len(real["sym"]), "real expr", len(real["expr"]), "synth sym", len(synth), "synth expr", len(exprs))
cnt = Counter(s["c"] for s in real["sym"]) + Counter(s["c"] for s in synth)
print({c: cnt[c] for c in NEW + ["x", "×"]})
