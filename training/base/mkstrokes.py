"""Turn MNIST test digits into pen-like strokes (for end-to-end testing of the JS reader)."""
import gzip, pickle, json, sys
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

tr, va, te = pickle.load(gzip.open(sys.argv[1]), encoding="latin1")
X, Y = te

def strokes_of(a, size=22.0):
    big = ndimage.zoom(a.reshape(28, 28), 4, order=1) > 0.35
    sk = skeletonize(big)
    lab, n = ndimage.label(sk, structure=np.ones((3, 3)))
    ys, xs = np.nonzero(sk)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    s = size / max(y1 - y0, 1)
    out = []
    for k in range(1, n + 1):
        py, px = np.nonzero(lab == k)
        pts = list(zip(px.tolist(), py.tolist()))
        if len(pts) < 3:
            continue
        # greedy path from an endpoint
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
                if (nxt[0] - cur[0]) ** 2 + (nxt[1] - cur[1]) ** 2 > 16:  # jump -> new stroke
                    out.append(path); path = []
            path.append(nxt); left.remove(nxt); cur = nxt
        out.append(path)
    res = []
    for p in out:
        if len(p) < 2:
            continue
        p = p[::2] + ([p[-1]] if len(p) % 2 == 0 else [])
        res.append([[round((x - x0) * s, 1), round((y - y0) * s, 1)] for x, y in p])
    return res

rng = np.random.default_rng(3)
samples = []
for i in rng.choice(len(X), 400, replace=False):
    st = strokes_of(X[i])
    if st:
        samples.append({"d": int(Y[i]), "s": st})
json.dump(samples, open("digits_strokes.json", "w"))
print(len(samples))
