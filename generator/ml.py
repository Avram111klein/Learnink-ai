"""Mini-language helpers (Python port of the trainer's target/tokens/barMask) + structural roles."""
import re
def target(src): return re.sub(r'[\[\]>\s]', '', src)
def tokens(text):
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c.isspace(): i += 1; continue
        if c == '^':
            j = i + 1
            while j < len(text) and text[j].isdigit(): out.append('^' + text[j]); j += 1
            i = j; continue
        out.append(c); i += 1
    return out
def roles(src):
    """per written token: (tok, role) with role in main/num/den/bar/pow/numpow/denpow; plus the index of tokens after which an arrow '>' sits"""
    out, arrows, depth, side, i = [], [], 0, None, 0
    while i < len(src):
        c = src[i]
        if c == '[':
            depth += 1; side = 'num'; j = src.index(']', i); bar_at = i + src[i:j].rfind('/'); i += 1; continue
        if c == ']': depth -= 1; side = None; i += 1; continue
        if c == '>': arrows.append(len(out)); i += 1; continue
        if c.isspace(): i += 1; continue
        if c == '^':
            j = i + 1
            while j < len(src) and src[j].isdigit(): out.append(('^' + src[j], (side + 'pow') if side else 'pow')); j += 1
            i = j; continue
        if depth and c == '/' and i == bar_at: out.append(('/', 'bar')); side = 'den'; i += 1; continue
        out.append((c, side or 'main')); i += 1
    assert [t for t, _ in out] == tokens(target(src)), src
    return out, arrows
