"""Cut real handwritten symbols out of CROHME expression bitmaps (CoMER's data.zip layout).

LICENCE: CROHME is CC BY-NC-SA 4.0 (non-commercial). Use it for EVALUATION ONLY; a model trained on it
must not ship in the paid app (the r1 run that used CROHME train is a reference, NOT for the app).

Only "linear" expressions are used (no \\frac, \\sqrt, sums…): their symbols read left to right, so
connected components sorted by x can be matched 1:1 to the caption tokens. Components that overlap
in x are merged first (the two bars of "=", the dots of "÷", the dot of "i", a 2-stroke "+" or "x").
An expression is kept only when the number of merged groups equals the number of tokens.

Each symbol is turned into pen-like strokes (skeleton → traced paths), so it goes through exactly the
same renderer as live S-Pen ink (train.py render() / hw.js pre()).

usage: python3 extract_crohme.py <crohme data dir> <split: train|2014|2016|2019> <out.json>
out: {"sym":[{"c":label,"s":strokes,"w":writer/file id}], "expr":[{"id","tex","s":strokes,"toks"}]}
"""
import sys, json, re
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

MAP = {"\\times": "×", "\\div": "÷", "\\cdot": "·"}
OK = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-=()/.,") | set(MAP)
SKIP = {"{", "}", "^", "_"}


def tokens(tex):
    t = [w for w in tex.split() if w not in SKIP]
    if not t or any(w not in OK for w in t):
        return None
    return [MAP.get(w, w) for w in t]


def trace(mask):
    """binary skeleton → list of polylines [[x,y],...] (greedy walk, as in mkstrokes.py)."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        py, px = np.nonzero(lab == k)
        pts = list(zip(px.tolist(), py.tolist()))
        if len(pts) == 1:
            out.append([pts[0]]); continue
        pset = set(pts)
        def deg(p):
            return sum((p[0] + dx, p[1] + dy) in pset for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)
        start = min(pts, key=deg)
        path = [start]; left = set(pts); left.remove(start); cur = start
        while left:
            nb = [(cur[0] + dx, cur[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx or dy) and (cur[0] + dx, cur[1] + dy) in left]
            if nb:
                nxt = nb[0]
            else:
                nxt = min(left, key=lambda q: (q[0] - cur[0]) ** 2 + (q[1] - cur[1]) ** 2)
                if (nxt[0] - cur[0]) ** 2 + (nxt[1] - cur[1]) ** 2 > 8:
                    out.append(path); path = []
            path.append(nxt); left.remove(nxt); cur = nxt
        out.append(path)
    res = []
    for p in out:
        if len(p) > 2:
            p = p[::2] + ([p[-1]] if len(p) % 2 == 0 else [])
        res.append([[float(x), float(y)] for x, y in p])
    return res


def groups_of(a):
    lab, n = ndimage.label(a > 0, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(lab)
    comps = [dict(ids=[i + 1], y0=s[0].start, y1=s[0].stop, x0=s[1].start, x1=s[1].stop) for i, s in enumerate(objs)]
    comps.sort(key=lambda c: c["x0"])
    G = []
    for c in comps:
        hit = None
        for g in G:
            ov = min(g["x1"], c["x1"]) - max(g["x0"], c["x0"])
            if ov >= 0.5 * max(1, min(g["x1"] - g["x0"], c["x1"] - c["x0"])):
                hit = g; break
        if hit:
            hit["ids"] += c["ids"]
            for k, f in (("x0", min), ("y0", min), ("x1", max), ("y1", max)):
                hit[k] = f(hit[k], c[k])
        else:
            G.append(dict(c))
    G.sort(key=lambda g: g["x0"])
    return lab, G


def main():
    root, split, outp = sys.argv[1:4]
    caps = [l.rstrip("\n").split("\t") for l in open(f"{root}/{split}/caption.txt", encoding="utf-8") if "\t" in l]
    syms, exprs, kept = [], [], 0
    for fid, tex in caps:
        toks = tokens(tex)
        if not toks:
            continue
        a = np.array(Image.open(f"{root}/{split}/img/{fid}.bmp").convert("L"))
        lab, G = groups_of(a)
        if len(G) != len(toks):
            continue
        # sanity: a "-" must be flat, "=" two flat bars
        bad = False
        for g, t in zip(G, toks):
            h, w = g["y1"] - g["y0"], g["x1"] - g["x0"]
            if t == "-" and h > 0.5 * w: bad = True
            if t == "=" and len(g["ids"]) != 2: bad = True
        if bad:
            continue
        kept += 1
        big = ndimage.zoom((a > 0).astype(np.float32), 2, order=1) > 0.5
        sk_all = skeletonize(big)
        labz = ndimage.zoom(lab, 2, order=0)
        all_st = []
        for g, t in zip(G, toks):
            m = sk_all & np.isin(labz, g["ids"])
            if m.sum() < 2:
                m = big & np.isin(labz, g["ids"])
            st = trace(m)
            st = [[[x / 2, y / 2] for x, y in s] for s in st]
            all_st += st
            syms.append({"c": t, "s": st, "w": fid})
        exprs.append({"id": fid, "tex": tex, "toks": toks, "s": all_st})
    json.dump({"sym": syms, "expr": exprs}, open(outp, "w"))
    from collections import Counter
    print(split, "captions", len(caps), "kept expr", kept, "symbols", len(syms))
    print(sorted(Counter(s["c"] for s in syms).items(), key=lambda x: -x[1]))


if __name__ == "__main__":
    main()
