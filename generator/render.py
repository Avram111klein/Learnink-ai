"""Same rendering for real and synthetic items: strokes drawn black on white, cropped to the ink + margin, fitted to a fixed 900x300 canvas."""
from PIL import Image, ImageDraw
W, H, M = 900, 300, 24
def render(strokes, pen=2.4):
    P = [p for st in strokes for p in st]
    x0, y0 = min(p[0] for p in P), min(p[1] for p in P); x1, y1 = max(p[0] for p in P), max(p[1] for p in P)
    s = min((W - 2 * M) / max(1, x1 - x0), (H - 2 * M) / max(1, y1 - y0), 2.0)
    ox, oy = (W - (x1 - x0) * s) / 2, (H - (y1 - y0) * s) / 2
    F = 3; im = Image.new('L', (W * F, H * F), 255); d = ImageDraw.Draw(im); lw = max(1, round(pen * s * F))
    for st in strokes:
        q = [((p[0] - x0) * s * F + ox * F, (p[1] - y0) * s * F + oy * F) for p in st]
        if len(q) == 1: q = q * 2
        d.line(q, fill=20, width=lw, joint='curve')
        for x, y in (q[0], q[-1]): d.ellipse([x - lw / 2, y - lw / 2, x + lw / 2, y + lw / 2], fill=20)
    return im.resize((W, H), Image.LANCZOS)
