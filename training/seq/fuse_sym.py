"""Stats for fuse_sym.js: base vs fusion per symbol (seg / aligned groups), by kind, Wilson CI, exact McNemar,
fixed/broken pairs, and the gate (digits not lower at all; total symbol accuracy higher).  usage: python3 fuse_sym.py out.json [...]"""
import json, sys, math, collections
def wil(k, n, z=1.96):
    if not n: return '–'
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f'{k}/{n} = {100*p:.1f}% [{100*(c-h):.0f}–{100*(c+h):.0f}]'
def mcn(b, c): n = b + c; k = min(b, c); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
kind = lambda t: 'digits' if t.isdigit() else 'letters' if t.isalpha() else 'operators'
for f in sys.argv[1:]:
    D = json.load(open(f)); print(f'=== {f}  W={D["W"]}  errors: {D["errs"] or "none"}')
    for K in ['seg', 'aligned']:
        R = [r for r in D['rows'] if r['kind'] == K]; print(f'-- {K} groups: {len(R)} symbols, {len({r["id"] for r in R})} exercises')
        for name, S in [('all', R)] + [(k, [r for r in R if kind(r['t']) == k]) for k in ['digits', 'letters', 'operators']]:
            b = sum(r['base'] == r['t'] for r in S); fu = sum(r['fuse'] == r['t'] for r in S)
            fx = sum(r['fuse'] == r['t'] and r['base'] != r['t'] for r in S); br = sum(r['base'] == r['t'] and r['fuse'] != r['t'] for r in S)
            print(f'   {name:9s} base {wil(b, len(S))} | fused {wil(fu, len(S))} | fixed {fx} broken {br} | McNemar p={mcn(fx, br):.3f}')
        fx = collections.Counter(f"{r['t']}: {r['base']}→{r['fuse']}" for r in R if r['fuse'] == r['t'] and r['base'] != r['t'])
        br = collections.Counter(f"{r['t']}: {r['base']}→{r['fuse']}" for r in R if r['base'] == r['t'] and r['fuse'] != r['t'])
        ww = collections.Counter(f"{r['t']}: {r['base']}→{r['fuse']}" for r in R if r['base'] != r['t'] and r['fuse'] != r['t'] and r['base'] != r['fuse'])
        print('   fixed :', dict(fx.most_common())); print('   broken:', dict(br.most_common())); print('   wrong→other wrong:', dict(ww.most_common()))
        dg = [r for r in R if kind(r['t']) == 'digits']
        g1 = sum(r['fuse'] == r['t'] for r in dg) >= sum(r['base'] == r['t'] for r in dg); g2 = sum(r['fuse'] == r['t'] for r in R) > sum(r['base'] == r['t'] for r in R)
        print(f'   GATE digits not lower: {"PASS" if g1 else "FAIL"} | total higher: {"PASS" if g2 else "FAIL"} | overall {"PASS" if g1 and g2 else "FAIL"}')
