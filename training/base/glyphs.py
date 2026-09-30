"""Synthetic pen strokes for algebra symbols (letters, ÷, and the old + x / ( ) set).

Every glyph is a few control-point polylines in a unit box (y down), smoothed with Catmull-Rom,
then distorted by a STYLE: slant, rotation, aspect, per-point jitter, pen width.
Training and the frozen test set use different STYLE dicts and different seeds
(see STYLE_TRAIN / STYLE_TEST), so the test measures generalisation to other hands, not memory.
"""
import numpy as np

STYLE_TRAIN = dict(rot=12, slant=(-0.22, 0.22), sx=(0.8, 1.2), sy=(0.8, 1.2), jit=(0.002, 0.012), smooth=False,
                   lw=(1.6, 3.2), cp=0.035, variant_bias=0.0)
# frozen-test style: stronger right slant, rounder/noisier strokes, thinner & thicker pens, other variant mix
STYLE_TEST = dict(rot=8, slant=(-0.1, 0.4), sx=(0.7, 1.3), sy=(0.75, 1.15), jit=(0.008, 0.02), smooth=True,
                  lw=(1.2, 3.8), cp=0.05, variant_bias=0.35)


def cr(P, n=None):
    """Catmull-Rom through control points."""
    P = np.asarray(P, float)
    if len(P) < 3:
        n = n or 14
        t = np.linspace(0, 1, n)[:, None]
        return P[0] + (P[-1] - P[0]) * t
    Q = np.vstack([P[0] * 2 - P[1], P, P[-1] * 2 - P[-2]])
    out = []
    k = n or 6
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        for t in np.linspace(0, 1, k, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    return np.array(out)


def arc(cx, cy, rx, ry, a0, a1, n=8):
    t = np.deg2rad(np.linspace(a0, a1, n))
    return [(cx + rx * np.cos(t_), cy - ry * np.sin(t_)) for t_ in t]   # y down, angles ccw


class Gen:
    def __init__(self, seed, style):
        self.r = np.random.default_rng(seed)
        self.S = style

    def u(self, a, b):
        return self.r.uniform(a, b)

    def pick(self, weights):
        w = np.array(weights, float)
        if self.S["variant_bias"]:
            w = w + self.S["variant_bias"] * w.mean()  # flatten the mix → rarer variants more common
        return self.r.choice(len(w), p=w / w.sum())

    # ---- templates: list of control-point lists; unit box, y down ----
    def tmpl(self, c):
        u, J = self.u, lambda: self.u(-1, 1) * self.S["cp"]
        if c == "a":
            v = self.pick([6, 3, 1])
            r = u(.3, .42)
            bowl = arc(.45, .55, r, u(.35, .45), u(20, 60), u(330, 370), 9)
            if v == 0:   # bowl + stem in one stroke
                top = (.45 + r + J(), .15 + J()); return [bowl + [top, (.45 + r + J(), .95), (.45 + r + .1, 1.0)]]
            if v == 1:   # bowl, then separate stem
                return [bowl, [(.45 + r + J(), .12), (.45 + r + J(), .98)]]
            return [[(.2, .15), (.5, .05), (.75, .2), (.75, .95), (.85, 1.0)], arc(.5, .7, .3, .25, 20, 330, 8)]  # double-story
        if c == "b":
            x = u(.2, .3); r = u(.28, .38)
            if self.pick([7, 3]) == 0:
                return [[(x + J(), 0.0), (x, .5), (x, 1.0)] + arc(x + r, .75, r, .25, 190, -170, 9)[1:]]
            return [[(x + J(), 0.0), (x, 1.0)], arc(x + r, .75, r, .25, 170, -170, 9)]
        if c == "c":
            return [arc(.5, .5, u(.3, .45), u(.4, .5), u(35, 70), u(290, 325), 9)]
        if c == "d":
            x = u(.7, .8); r = u(.28, .38)
            if self.pick([6, 4]) == 0:
                return [arc(x - r, .72, r, .27, u(10, 40), u(340, 370), 9) + [(x + J(), 0.0), (x, 1.0)]]
            return [arc(x - r, .72, r, .27, 20, 370, 9), [(x + J(), 0.0), (x, 1.0)]]
        if c == "k":
            x = u(.2, .3); m = u(.55, .7)
            if self.pick([6, 4]) == 0:
                return [[(x, 0.0), (x, 1.0)], [(u(.75, .9), u(.3, .45)), (x + .02, m), (u(.75, .9), 1.0)]]
            return [[(x, 0.0), (x, 1.0)], [(u(.75, .9), u(.3, .45)), (x + .02, m)], [(x + .15, m - .05), (u(.75, .95), 1.0)]]
        if c == "m":
            h = u(.05, .2)
            P = [(0.05, h + .05), (.07, 1.0), (.1, .45), (.25, h), (.4, .35), (.42, 1.0), (.45, .45), (.62, h), (.78, .35), (.8, 1.0)]
            return [[(x * 1.2, y * .75) for x, y in P]]
        if c == "n":
            h = u(.05, .25)
            return [[(0.1, h + .05), (.12, 1.0), (.2, .45), (.45, h), (.75, .3), (.8, 1.0)]]
        if c == "p":
            x = u(.2, .3); r = u(.25, .35)
            if self.pick([6, 4]) == 0:
                return [[(x, 0.0), (x, 1.0), (x, .6)] + arc(x + r, .27, r, .25, 200, -160, 9)[1:]]
            return [[(x, 0.0), (x, 1.0)], arc(x + r, .27, r, .25, 170, -170, 9)]
        if c == "t":
            x = u(.4, .55); yb = u(.25, .4)
            stem = [(x + J(), 0.0), (x, .8), (x + .1, 1.0), (x + u(.2, .35), .92)] if self.pick([6, 4]) == 0 else [(x + J(), 0.0), (x, 1.0)]
            return [stem, [(u(.05, .2), yb), (u(.8, .95), yb + J())]]
        if c == "y":
            v = self.pick([5, 4, 2])
            if v == 0:   # two straight strokes
                return [[(u(.05, .2), 0.0), (u(.45, .55), .5)], [(u(.8, .95), 0.0), (.5, .55), (u(.2, .35), 1.0)]]
            if v == 1:   # one stroke: u-shape + descender
                return [[(.1, 0.0), (.12, .3), (.3, .48), (.6, .4), (.75, 0.0), (.75, .6), (.65, .92), (.4, 1.0), (.25, .88)]]
            return [[(.1, 0.0), (.5, .5)], [(.9, 0.0), (.5, .5), (.45, 1.0)]]  # y with straight tail
        if c == "s":
            return [[(.8, .12), (.5, 0.0), (.2, .15), (.3, .42), (.7, .58), (.8, .85), (.5, 1.0), (.15, .88)]]
        if c == "z":
            v = self.pick([6, 3, 1])
            Z = [(u(.05, .15), u(0, .08)), (u(.85, .95), u(0, .08)), (u(.05, .15), u(.92, 1)), (u(.85, .95), u(.92, 1))]
            if v == 0:
                return [Z]
            if v == 1:   # with a middle cross bar
                return [Z, [(.25, .5), (.75, .5)]]
            return [[(.1, .1), (.45, 0.0), (.8, .1), (.1, 1.0), (.9, .95)]]  # rounded top
        if c == "÷":
            y = u(.45, .55); d = u(.05, .09)
            dot = lambda cy: [(.5 + J() - d / 2, cy), (.5 + J() + d / 2, cy + d / 2)]
            return [[(0.0, y), (1.0, y + J())], dot(u(0, .15)), dot(u(.85, 1.0))]
        if c == "+":
            cx, cy = u(.4, .6), u(.4, .6)
            v = [(cx + J(), u(0, .2)), (cx + J(), u(.8, 1))]; h = [(u(0, .2), cy + J()), (u(.8, 1), cy + J())]
            return [v, h] if self.r.random() < .5 else [h, v]
        if c == "x":
            if self.pick([7, 3]) == 0:
                return [[(u(0, .15), u(0, .15)), (u(.85, 1), u(.85, 1))], [(u(.85, 1), u(0, .15)), (u(0, .15), u(.85, 1))]]
            return [arc(.1, .5, .38, .48, 70, -70, 8), arc(.9, .5, .38, .48, 110, 250, 8)]
        if c == "/":
            return [[(u(.1, .35), u(.9, 1)), (u(.65, .9), u(0, .1))]]
        if c in "()":
            a0, a1 = u(110, 130), u(230, 250); rx, ry = u(.25, .5), u(.5, .6)
            P = arc(.7, .5, rx, ry, a0, a1, 9)
            if c == ")":
                P = [(1 - x, y) for x, y in P]
            return [P]
        raise KeyError(c)

    def strokes(self, c):
        S = self.S
        out = []
        for P in self.tmpl(c):
            st = cr(P) if len(P) >= 2 else np.array(P, float)
            n = len(st)
            if n > 1:
                amt = self.u(*S["jit"])
                noise = np.cumsum(self.r.normal(0, amt, (n, 2)), 0)
                if S["smooth"]:
                    k = np.ones(5) / 5
                    noise = np.stack([np.convolve(noise[:, i], k, "same") for i in (0, 1)], 1)
                noise -= np.linspace(0, 1, n)[:, None] * noise[-1]
                st = st + noise
            out.append(st)
        a = np.deg2rad(self.u(-S["rot"], S["rot"])); sh = self.u(*S["slant"])
        sx, sy = self.u(*S["sx"]), self.u(*S["sy"])
        M = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.array([[1, -sh], [0, 1]]) @ np.diag([sx, sy])
        return [st @ M.T for st in out]

    def lw(self):
        return self.u(*self.S["lw"])
