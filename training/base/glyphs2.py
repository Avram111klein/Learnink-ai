"""Richer synthetic handwriting for TRAINING (r2+). glyphs.py stays frozen: it generates the frozen test set.

Adds on top of glyphs.Gen:
  * pen-stroke digits 0-9 (to complement MNIST) with the confusable variants:
    1 plain / flag / flag+base, 7 with/without bar, 2 round / looped, 4 open/closed, 5, 6, 9 round/straight tail
  * more letter variants: open/closed a bowl, b with open/closed bowl, x as ")(" arcs / one-stroke loop,
    z sharp / with bar, s smooth, y cursive/straight, t hooked
  * realism: smooth elastic warp (low-frequency displacement field), start/end hooks, overshoot,
    occasional broken stroke (pen lifted mid-line), per-glyph pen width
"""
import numpy as np
from glyphs import Gen, arc, cr

STYLE_TRAIN2 = dict(rot=12, slant=(-0.25, 0.4), sx=(0.7, 1.3), sy=(0.75, 1.2), jit=(0.002, 0.018), smooth=None,
                    lw=(1.2, 3.8), cp=0.05, variant_bias=0.3, warp=(0.0, 0.06), hook=0.35, gap=0.08)


class Gen2(Gen):
    def tmpl(self, c):
        u, J, pk = self.u, lambda: self.u(-1, 1) * self.S["cp"], self.pick
        if c == "0":
            return [arc(.5, .5, u(.3, .45), .5, u(80, 100), u(80, 100) + u(340, 380), 12)]
        if c == "1":
            v = pk([5, 4, 2])
            s = [(.5 + J(), 0.0), (.5 + J(), 1.0)]
            if v == 0: return [s]
            if v == 1: return [[(u(.2, .35), u(.15, .3)), (.5, 0.0), (.5 + J(), 1.0)]]
            return [[(u(.2, .35), u(.15, .3)), (.5, 0.0), (.5 + J(), 1.0)], [(u(.2, .3), 1.0), (u(.7, .8), 1.0)]]
        if c == "2":
            v = pk([6, 3])
            top = arc(.5, .3, u(.3, .4), .28, u(150, 180), u(-20, 0) - 40, 7)
            if v == 0: return [top + [(u(.05, .15), 1.0), (u(.85, 1), u(.95, 1))]]
            return [top + [(.15, .95), (.1, .85), (.25, .8), (.35, .95), (u(.85, 1), u(.95, 1))]]   # loop at the bottom
        if c == "3":
            return [arc(.45, .27, .3, .25, 160, -90, 7) + arc(.45, .73, .33, .27, 90, -160, 7)]
        if c == "4":
            if pk([5, 5]) == 0:   # open 4: two strokes
                return [[(u(.3, .45), 0.0), (u(.05, .15), .65), (u(.85, 1), .65)], [(u(.65, .75), u(.1, .35)), (u(.65, .75), 1.0)]]
            return [[(u(.65, .75), 1.0), (u(.65, .75), 0.0), (.05, .65), (u(.85, 1), .65)]]   # closed 4, one stroke
        if c == "5":
            bar = [(u(.8, .95), u(0, .05)), (u(.25, .35), 0.0)]
            body = [(u(.25, .35), 0.0), (u(.2, .3), .45)] + arc(.5, .68, .33, .3, 140, -150, 8)
            return [body, bar] if pk([6, 4]) == 0 else [bar + body[1:]]
        if c == "6":
            return [[(u(.65, .8), 0.0), (.35, .25), (.2, .6)] + arc(.5, .7, .3, .28, 190, -170, 9)]
        if c == "7":
            s = [(u(.05, .15), u(0, .08)), (u(.85, 1), 0.0), (u(.35, .55), 1.0)]
            return [s] if pk([6, 4]) == 0 else [s, [(.3, .5), (.8, .5)]]
        if c == "8":
            return [arc(.5, .25, u(.22, .3), .24, -90, 270, 10) + arc(.5, .73, u(.28, .36), .26, 90, -270, 11)[1:]]
        if c == "9":
            bowl = arc(.5, .3, u(.28, .38), .28, u(-20, 20), u(-20, 20) + 360, 10)
            if pk([6, 4]) == 0: return [bowl + [(bowl[0][0] + J(), 1.0)]]                       # straight tail
            return [bowl + [(bowl[0][0], .7), (.6, .95), (.35, 1.0)]]                          # curved tail
        if c == "a":
            v = pk([5, 3, 2, 1])
            r = u(.3, .42)
            if v == 2:   # open bowl (gap at the top right), stem separate
                return [arc(.45, .55, r, u(.35, .45), u(60, 90), 330, 9), [(.45 + r + J(), .15), (.45 + r, .98)]]
            if v == 3:   # "o" with a tiny tail — closed loop
                return [arc(.45, .55, r, .42, 10, 370, 12) + [(.45 + r + .12, .98)]]
        if c == "b":
            x = u(.2, .3); r = u(.28, .38)
            if pk([7, 3]) == 1:   # open bowl (does not return to the stem)
                return [[(x + J(), 0.0), (x, 1.0), (x, .6)] + arc(x + r, .75, r, .25, 170, -120, 8)[1:]]
        if c == "x":
            v = pk([4, 4, 2])
            if v == 1:   # ")(" back-to-back arcs, sometimes touching
                d = u(-.05, .12)
                return [arc(.1 - d, .5, .38, .48, 70, -70, 8), arc(.9 + d, .5, .38, .48, 110, 250, 8)]
            if v == 2:   # one-stroke loop x (like a script x)
                return [[(0.0, .1), (.35, .2), (.5, .5), (.35, .8), (.1, .75), (.3, .6), (.5, .5), (.7, .2), (1.0, .05)], [(.55, .6), (.95, 1.0)]]
        if c == "z":
            v = pk([5, 3, 2])
            Z = [(u(.05, .15), u(0, .08)), (u(.85, .95), u(0, .08)), (u(.05, .15), u(.92, 1)), (u(.85, .95), u(.92, 1))]
            if v == 1: return [Z, [(.25, .5), (.75, .5)]]
            if v == 2: return [[(.1, .05), (.9, .05), (.1, .95), (.5, .92), (.9, 1.0)]]
            return [Z]
        if c == "s":
            k = u(.1, .25)
            return [[(.85, .12), (.5, 0.0), (.15, .1 + k * .3), (.3, .42), (.7, .58), (.85, .85), (.5, 1.0), (.1, .88)]]
        if c == "y" and pk([6, 4]) == 1:
            return [[(.1, 0.0), (.2, .4), (.5, .5), (.8, 0.0)], [(.8, 0.0), (.7, .7), (.5, 1.0), (.3, .9)]]
        return super().tmpl(c)

    def strokes(self, c):
        S = self.S
        out = []
        for P in self.tmpl(c):
            P = [tuple(p) for p in P]
            if self.r.random() < 0.5:
                P = P[::-1]
            st = cr(P) if len(P) >= 2 else np.array(P, float)
            n = len(st)
            if n > 1:
                amt = self.u(*S["jit"])
                noise = np.cumsum(self.r.normal(0, amt, (n, 2)), 0)
                if self.r.random() < 0.5:
                    k = np.ones(5) / 5
                    noise = np.stack([np.convolve(noise[:, i], k, "same") for i in (0, 1)], 1)
                noise -= np.linspace(0, 1, n)[:, None] * noise[-1]
                st = st + noise
                if self.r.random() < S["hook"]:          # tiny hook / overshoot at an end
                    d = st[-1] - st[-2]; nd = np.linalg.norm(d) + 1e-6
                    perp = np.array([-d[1], d[0]]) / nd * self.u(-1, 1)
                    L = self.u(0.03, 0.08)
                    st = np.vstack([st, st[-1] + d / nd * L * .5 + perp * L])
                if self.r.random() < S["gap"] and n > 8:  # pen lifted mid-line
                    k = self.r.integers(3, n - 3)
                    out.append(st[:k]); st = st[k + 1:]
            out.append(st)
        # smooth elastic warp: low-frequency displacement field
        w = self.u(*S["warp"])
        if w > 0:
            fx, fy, ph = self.r.uniform(1, 3, 2), self.r.uniform(1, 3, 2), self.r.uniform(0, 6.3, 4)
            def warp(p):
                return p + w * np.stack([np.sin(fx[0] * p[:, 1] * 3.1 + ph[0]) + .5 * np.sin(fx[1] * p[:, 0] * 3.1 + ph[1]),
                                         np.sin(fy[0] * p[:, 0] * 3.1 + ph[2]) + .5 * np.sin(fy[1] * p[:, 1] * 3.1 + ph[3])], 1)
            out = [warp(s) for s in out]
        a = np.deg2rad(self.u(-S["rot"], S["rot"])); sh = self.u(*S["slant"])
        sx, sy = self.u(*S["sx"]), self.u(*S["sy"])
        M = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.array([[1, -sh], [0, 1]]) @ np.diag([sx, sy])
        return [s @ M.T for s in out if len(s)]
