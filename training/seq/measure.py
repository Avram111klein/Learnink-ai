"""Stats for measure.js output: per-symbol accuracy (pixel vs sequence), 2x2, combination ceiling, by kind,
confusions, calibration at p>=0.5, 2x2 restricted to pixel-sure symbols, exercise level (all aligned symbols right).
usage: python3 measure.py <meas.json> [...]"""
import json, sys, math, collections
def wil(k, n, z=1.96):
    if not n: return '–'
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f'{k}/{n} = {100*p:.1f}% [{100*(c-h):.0f}–{100*(c+h):.0f}]'
kind = lambda t: 'digits' if t.isdigit() else 'operators' if t in '+/()÷' else 'letters'
def two(R, title):
    a = sum(r['pix'] == r['t'] and r['seq'] == r['t'] for r in R); p = sum(r['pix'] == r['t'] and r['seq'] != r['t'] for r in R)
    s = sum(r['pix'] != r['t'] and r['seq'] == r['t'] for r in R); w = sum(r['pix'] != r['t'] and r['seq'] != r['t'] for r in R); n = len(R) or 1
    print(f'  2x2 {title} (n={len(R)}): both {a} ({100*a/n:.1f}%) | pixel only {p} ({100*p/n:.1f}%) | seq only {s} ({100*s/n:.1f}%) | both wrong {w} ({100*w/n:.1f}%)')
    print(f'  ceiling (at least one right): {wil(a+p+s, len(R))}')
def mcnemar(b, c):   # exact two-sided
    n = b + c; k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
for f in sys.argv[1:]:
    M = json.load(open(f)); R = M['rows']; print(f'=== {f}: {len(R)} symbols, {len({r["id"] for r in R})} exercises')
    for e in ['pix', 'seq']: print(f'  {e} acc: {wil(sum(r[e]==r["t"] for r in R), len(R))}')
    b = sum(r['pix'] == r['t'] and r['seq'] != r['t'] for r in R); c = sum(r['pix'] != r['t'] and r['seq'] == r['t'] for r in R)
    print(f'  McNemar exact p (pixel vs seq) = {mcnemar(b, c):.3f}')
    two(R, 'all')
    for k in ['digits', 'letters', 'operators']:
        K = [r for r in R if kind(r['t']) == k]; print(f'  {k:9s} pix {wil(sum(r["pix"]==r["t"] for r in K), len(K))} | seq {wil(sum(r["seq"]==r["t"] for r in K), len(K))}')
    for e in ['pix', 'seq']: print(f'  {e} confusions:', collections.Counter(r['t'] + '→' + r[e] for r in R if r[e] != r['t']).most_common(10))
    for e in ['pix', 'seq']:
        S = [r for r in R if r[e + 'p'] >= 0.5]; wr = sum(r[e] != r['t'] for r in S)
        print(f'  {e} sure (p>=0.5): {len(S)}/{len(R)} symbols, wrong {wr} ({100*wr/max(1,len(S)):.1f}%); not sure: {len(R)-len(S)}, wrong {sum(r[e]!=r["t"] for r in R if r[e+"p"]<0.5)}')
    PS = [r for r in R if r['pixp'] >= 0.5]; two(PS, 'pixel-sure only')
    SW = [r for r in PS if r['pix'] != r['t']]
    print(f'  pixel sure-wrong {len(SW)}: seq right on {sum(r["seq"]==r["t"] for r in SW)} (of those seq sure: {sum(r["seq"]==r["t"] and r["seqp"]>=0.5 for r in SW)}); detail:', [(r['t'], r['pix'], round(r['pixp'], 2), r['seq'], round(r['seqp'], 2)) for r in SW])
    PR = [r for r in PS if r['pix'] == r['t']]; print(f'  pixel sure-right {len(PR)}: seq wrong on {sum(r["seq"]!=r["t"] for r in PR)} (seq sure-wrong there: {sum(r["seq"]!=r["t"] and r["seqp"]>=0.5 for r in PR)})')
    E = collections.defaultdict(list)
    for r in R: E[r['id']].append(r)
    for e in ['pix', 'seq']: print(f'  exercise level ({e}): all aligned symbols right in {sum(all(r[e]==r["t"] for r in v) for v in E.values())}/{len(E)}')
    print(f'  exercise level (at least one engine right on every symbol): {sum(all(r["pix"]==r["t"] or r["seq"]==r["t"] for r in v) for v in E.values())}/{len(E)}')
    bins = [0, .5, .7, .9, .99, 1.01]
    for e in ['seq', 'pix']:
        print(f'  {e} reliability:', ' '.join(f'[{lo:.2f},{hi:.2f}) n={len(B)} acc={sum(r[e]==r["t"] for r in B)/len(B):.2f}' for lo, hi in zip(bins, bins[1:]) for B in [[r for r in R if lo <= r[e+"p"] < hi]] if B))
    print(f'  ms/symbol in Chromium: pixel {M["msPix"]:.3f}, seq {M["msSeq"]:.3f} (warm {M["msSeqWarm"]:.3f}); null checks {M["nullChecks"]}')
