"""Score fuse_eval.js output: baseline (hw_exp.js) vs fusion (hw_fuse.js FUSE on), end to end with readAnswer.
answerAcc = exact text; symbolAcc = trainer judge (Levenshtein token alignment, ok tokens / target tokens);
falseX = sure (p>=0.5), wrong, well-formed, same shape, |len diff|<=1, target not in alts (training/round2/score.py);
digits = target digit tokens read right (same alignment); '?' = answer not sure (p<0.5).
Gate (fixed in advance): answerAcc, symbolAcc not lower; falseX not higher; digits drop <= 1 pt; '?' rise <= 10 pts.
usage: python3 fuse_score.py <fe.json> <items.json> [--ids]   (--ids lists differing item ids, not texts)"""
import json, sys, math, re
sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/round2')
from score import lev_ok, tokens, wellformed, shape
def target(s): return re.sub(r'[\[\]>\s]', '', s)
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f'{k}/{n} = {100*p:.1f}% [{100*(c-h):.0f}–{100*(c+h):.0f}]'
def align_ok(T, R):   # per-target-token ok flags, same DP/backtrack as the trainer's align()
    n, m = len(T), len(R); D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1): D[i][j] = min(D[i-1][j-1] + (T[i-1] != R[j-1]), D[i-1][j] + 1, D[i][j-1] + 1)
    ok = [False] * n; i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i-1][j-1] + (T[i-1] != R[j-1]): ok[i-1] = T[i-1] == R[j-1]; i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i-1][j] + 1: i -= 1
        else: j -= 1
    return ok
def per_item(tgt, r):
    text = (r or {}).get('text', '') or ''; p = (r or {}).get('p', 0) or 0; alts = (r or {}).get('alts', []) or []
    T = tokens(tgt); ok = align_ok(T, tokens(text)); exact = text == tgt
    fx = (not exact) and p >= 0.5 and tgt not in alts and wellformed(text) and shape(text) == shape(tgt) and abs(len(text) - len(tgt)) <= 1
    dig = [o for t, o in zip(T, ok) if t.isdigit()]
    return dict(exact=exact, nOk=sum(ok), nT=len(T), fx=fx, q=p < 0.5, dOk=sum(dig), dN=len(dig), text=text, p=p)
def binom2(b, c):
    n = b + c; k = min(b, c); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
if __name__ == '__main__':
    F = json.load(open(sys.argv[1])); I = {x['id']: x for x in json.load(open(sys.argv[2]))}
    print(f"n={F['n']}  FUSE-off identical to base: {F['offIdentical']}  groups identical: {F['segIdentical']} ({F['nGroups']} groups, unresolved {F['unresolved']})  errors: {F['errs'] or 'none'}")
    print(f"ms/answer: {F['ms']}   params: {F['info']}")
    S = {}
    for e in ['base', 'fuse']:
        S[e] = [per_item(target(I[r['id']]['promptText']), r[e]) for r in F['rows']]
    n = len(F['rows']); M = {}
    for e in ['base', 'fuse']:
        X = S[e]; M[e] = dict(ans=sum(x['exact'] for x in X) / n, sym=sum(x['nOk'] for x in X) / sum(x['nT'] for x in X),
                              fx=sum(x['fx'] for x in X) / n, dig=sum(x['dOk'] for x in X) / sum(x['dN'] for x in X), q=sum(x['q'] for x in X) / n)
        print(f"{e:5s} answerAcc {wil(sum(x['exact'] for x in X), n)} | symbolAcc {wil(sum(x['nOk'] for x in X), sum(x['nT'] for x in X))} | "
              f"falseX {wil(sum(x['fx'] for x in X), n)} | digits {wil(sum(x['dOk'] for x in X), sum(x['dN'] for x in X))} | '?' {wil(sum(x['q'] for x in X), n)}")
    B, Fu = M['base'], M['fuse']
    gate = [('answerAcc not lower', Fu['ans'] >= B['ans'] - 1e-12, Fu['ans'] - B['ans']), ('symbolAcc not lower', Fu['sym'] >= B['sym'] - 1e-12, Fu['sym'] - B['sym']),
            ('falseX not higher', Fu['fx'] <= B['fx'] + 1e-12, Fu['fx'] - B['fx']), ('digits drop <= 1 pt', Fu['dig'] >= B['dig'] - 0.01 - 1e-12, Fu['dig'] - B['dig']),
            ("'?' rise <= 10 pts", Fu['q'] <= B['q'] + 0.10 + 1e-12, Fu['q'] - B['q'])]
    for name, ok, d in gate: print(f"  GATE {name:22s} {'PASS' if ok else 'FAIL'}  (delta {100*d:+.1f} pts)")
    print('  GATE overall:', 'PASS' if all(g[1] for g in gate) else 'FAIL')
    a, b = S['base'], S['fuse']
    txt = sum(x['text'] != y['text'] for x, y in zip(a, b)); pch = sum(x['text'] == y['text'] and x['p'] != y['p'] for x, y in zip(a, b))
    print(f"items with different text: {txt}; same text, different p: {pch}; sure->unsure: {sum((not x['q']) and y['q'] for x,y in zip(a,b))}; unsure->sure: {sum(x['q'] and not y['q'] for x,y in zip(a,b))}")
    ex_b = sum(x['exact'] and not y['exact'] for x, y in zip(a, b)); ex_f = sum(y['exact'] and not x['exact'] for x, y in zip(a, b))
    print(f"exact: base-only {ex_b}, fuse-only {ex_f} (exact binomial p={binom2(ex_b, ex_f):.3f}); falseX: base-only {sum(x['fx'] and not y['fx'] for x,y in zip(a,b))}, fuse-only {sum(y['fx'] and not x['fx'] for x,y in zip(a,b))}")
    print(f"symbol ok per item: fuse better in {sum(y['nOk']>x['nOk'] for x,y in zip(a,b))}, worse in {sum(y['nOk']<x['nOk'] for x,y in zip(a,b))}; tokens gained {sum(max(0,y['nOk']-x['nOk']) for x,y in zip(a,b))}, lost {sum(max(0,x['nOk']-y['nOk']) for x,y in zip(a,b))}")
    if '--ids' in sys.argv:
        for r, x, y in zip(F['rows'], a, b):
            if x['text'] != y['text'] or x['p'] != y['p']:
                print(f"   {r['id']}  ok {x['nOk']}->{y['nOk']}/{x['nT']}  exact {int(x['exact'])}->{int(y['exact'])}  p {x['p']:.2f}->{y['p']:.2f}  fx {int(x['fx'])}->{int(y['fx'])}" + ('  [text]' if x['text'] != y['text'] else ''))
