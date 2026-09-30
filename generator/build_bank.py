"""Symbol bank + style statistics from the owner's TRAIN exercises only (split.json 'train'); aligned by align.js."""
import json, gzip, glob, collections, numpy as np
from ml import roles, tokens, target
SP = '__WORKDIR__'
split = json.load(open(SP + '/hwdata/round2/split.json'))
TRAIN, TEST = set(split['train']), set(split['testA']) | set(split['testB'])
items = {}
for f in glob.glob(SP + '/trainer_data2/samples/__WRITER__/ex/*.json'):
    for x in json.load(open(f))['items']:
        if x['id'] in TRAIN and not x['discarded'] and not x['warmup']: items[x['id']] = x
AL = [r for r in json.load(open('align_train.json')) if r['ok']]
assert all(r['id'] in TRAIN and r['id'] not in TEST for r in AL)
BASE = set('0123456789acemnsxzbdkt')   # sit on the baseline
def bbox(P): P = np.asarray(P, float); return P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()
def kind(c):
    c = c.lstrip('^')
    return 'd' if c.isdigit() else 'l' if c.isalpha() else 'o' if c in '(' else 'c' if c == ')' else 'p' if c == '.' else 'op'
def run(FSCALE, OPCY, SLOPES):
    bank = collections.defaultdict(list); ex_stats = []
    pair_gaps = collections.defaultdict(list); gaps = collections.defaultdict(list); pauses = []; fracs = []; pows = []; arrows = []; eqs = []; minus = []
    for r in AL:
        it = items[r['id']]; ST = it['strokes']; rl, arr = roles(it['promptText'])
        assert [t for t, _ in rl] == r['T'] and len(rl) == len(r['per'])
        syms = []
        for (tok, role), q in zip(rl, r['per']):
            idx = sorted(q['strokes']); pts = [p for k in idx for p in ST[k]]
            x0, y0, x1, y1 = bbox([p[:2] for p in pts])
            syms.append(dict(t=tok, c='bar' if role == 'bar' else tok.lstrip('^'), role=role, idx=idx, x0=x0, y0=y0, x1=x1, y1=y1, w=x1 - x0, h=y1 - y0, cx=(x0 + x1) / 2,
                             ts=ST[idx[0]][0][2], te=max(p[2] for k in idx for p in ST[k])))
        main = [s for s in syms if s['role'] == 'main']
        dig = [s['h'] for s in main if s['c'].isdigit()]
        body = [s['h'] for s in main if s['c'] in BASE]
        D = float(np.median(dig)) if len(dig) >= 2 else float(np.median(body)) if body else None
        bs = [s for s in main if s['c'] in BASE and D and 0.5 * D < s['h'] < 1.6 * D]
        if D and len(bs) >= 3:
            X = np.array([s['cx'] for s in bs]); Y = np.array([s['y1'] for s in bs]); keep = np.ones(len(X), bool)
            for _ in range(2):
                A = np.polyfit(X[keep], Y[keep], 1); res = Y - np.polyval(A, X); keep = np.abs(res) < max(0.25 * D, 2 * np.std(res[keep]) + 1)
                if keep.sum() < 3: keep[:] = True
            slope, icpt = A; resid = float(np.std((Y - np.polyval(A, X))[keep]) / D)
        else:
            # fraction-only exercises (e.g. [7/8]-[1/4]=[5/8]): D from all digits / typical fraction scale, baseline through the
            # main-row '=' and '-' (their usual height above the baseline, from exercises that have a real baseline)
            alld = [s['h'] for s in syms if s['c'].isdigit()]; ops = [s for s in main if s['c'] in '=-']
            if not alld or not ops: continue
            D = float(np.median(alld)) / FSCALE
            P = np.array([[s['cx'], (s['y0'] + s['y1']) / 2 - OPCY[s['c']] * D] for s in ops])
            slope = np.polyfit(P[:, 0], P[:, 1], 1)[0] if len(P) >= 2 and np.ptp(P[:, 0]) > 2 * D else float(np.median(SLOPES)); icpt = float(np.mean(P[:, 1] - slope * P[:, 0])); resid = None
        base = lambda x: slope * x + icpt
        # slant: shear of tall single symbols (dx per dy of principal axis)
        sh = []
        for s in main:
            if s['c'] in '14()7dbkt' and s['h'] > 0.6 * D:
                P = np.array([p[:2] for k in s['idx'] for p in ST[k]], float); vy = P[:, 1].var()
                if vy > 0: sh.append(float(np.cov(P[:, 0], P[:, 1])[0, 1] / vy))
        pr = [p[3] for s in syms for k in s['idx'] for p in ST[k]]
        ex_stats.append(dict(id=r['id'], D=D, fallback=resid is None, slope=float(slope), resid=resid, slant=float(np.median(sh)) if sh else None, x_start=syms[0]['x0'], y_start=float(base(syms[0]['cx'])),
                             pmean=float(np.mean(pr)), pstd=float(np.std(pr)), canvas=[it['device']['w'], it['device']['h']], dur=it['durMs'], n=len(syms), width=(max(s['x1'] for s in syms) - syms[0]['x0']) / D))
        for s in syms:
            idx = s['idx']; cxb = base(s['cx']); t0 = s['ts']
            strokes = [[[round((p[0] - s['x0']) / D, 4), round((p[1] - cxb) / D, 4), p[2] - t0, p[3]] for p in ST[k]] for k in idx]
            rec = dict(src=r['id'], role=s['role'], s=strokes, w=s['w'] / D, h=s['h'] / D, bot=(s['y1'] - cxb) / D, top=(s['y0'] - cxb) / D, D=D, slope=float(slope), dur=s['te'] - t0, n=len(idx))
            bank[s['c']].append(rec)
        for k in r.get('arrow') or []:
            if k:
                P = [p for j in k for p in ST[j]]; x0, y0, x1, y1 = bbox([p[:2] for p in P]); t0 = ST[k[0]][0][2]
                arrows.append(dict(src=r['id'], s=[[[round((p[0] - x0) / D, 4), round((p[1] - base((x0 + x1) / 2)) / D, 4), p[2] - t0, p[3]] for p in ST[j]] for j in k]))
        # gaps / pauses between consecutive main-row items (fractions and powers handled separately)
        seq = []; i = 0
        while i < len(syms):
            s = syms[i]
            if s['role'] in ('num', 'bar', 'den', 'numpow', 'denpow'):
                j = i
                while j < len(syms) and syms[j]['role'] in ('num', 'bar', 'den', 'numpow', 'denpow'): j += 1
                F = syms[i:j]; bar = [q for q in F if q['role'] == 'bar'][0]; nu = [q for q in F if q['role'] in ('num', 'numpow')]; de = [q for q in F if q['role'] in ('den', 'denpow')]
                nw = max(q['x1'] for q in nu) - min(q['x0'] for q in nu); dw = max(q['x1'] for q in de) - min(q['x0'] for q in de)
                bcy = (bar['y0'] + bar['y1']) / 2
                fracs.append(dict(bar_y=(bcy - base(bar['cx'])) / D, num_gap=(bcy - max(q['y1'] for q in nu)) / D, den_gap=(min(q['y0'] for q in de) - bcy) / D,
                                  scale=float(np.median([q['h'] for q in nu + de if q['c'] not in '+-('] or [D])) / D, over=(bar['w'] - max(nw, dw)) / D,
                                  num_dx=((min(q['x0'] for q in nu) + max(q['x1'] for q in nu)) / 2 - bar['cx']) / D, den_dx=((min(q['x0'] for q in de) + max(q['x1'] for q in de)) / 2 - bar['cx']) / D,
                                  num_gaps=[(b['x0'] - a['x1']) / D for a, b in zip(nu, nu[1:])], den_gaps=[(b['x0'] - a['x1']) / D for a, b in zip(de, de[1:])]))
                seq.append(dict(c='frac', x0=min(q['x0'] for q in F), x1=max(q['x1'] for q in F), ts=F[0]['ts'], te=F[-1]['te'], at=i)); i = j; continue
            if s['role'] == 'pow':
                b = seq[-1] if seq else None
                if b and b['c'] != 'frac':
                    pows.append(dict(base=b['c'], rel_top=(s['y1'] - b['y0']) / D if 'y0' in b else None, scale=s['h'] / D, bot=(s['y1'] - base(s['cx'])) / D, top=(s['y0'] - base(s['cx'])) / D, dx=(s['x0'] - b['x1']) / D))
                seq[-1] = dict(seq[-1], x1=max(seq[-1]['x1'], s['x1']), te=s['te'], c=seq[-1]['c'] + '^'); i += 1; continue
            seq.append(dict(c=s['c'], x0=s['x0'], x1=s['x1'], y0=s['y0'], y1=s['y1'], ts=s['ts'], te=s['te'], at=i)); i += 1
        arrow_after = set(arr)
        for a, b in zip(seq, seq[1:]):
            g = (b['x0'] - a['x1']) / D
            if b['at'] in arrow_after: gaps['arrow'].append(g); continue
            ka = 'f' if a['c'] == 'frac' else kind(a['c'].rstrip('^')); kb = 'f' if b['c'] == 'frac' else kind(b['c'])
            gaps[ka + '>' + kb].append(g); pauses.append(b['ts'] - a['te']); pair_gaps[a['c'].rstrip('^') + ('^' if a['c'].endswith('^') else '') + '|' + b['c']].append(g)
        for s in syms:
            if s['c'] == '=' and s['role'] == 'main':
                P = sorted([bbox([p[:2] for p in ST[k]]) for k in s['idx']], key=lambda b: b[1])
                eqs.append(dict(w=s['w'] / D, sep=((P[1][1] + P[1][3]) - (P[0][1] + P[0][3])) / 2 / D, cy=((s['y0'] + s['y1']) / 2 - base(s['cx'])) / D, dx=((P[1][0] + P[1][2]) - (P[0][0] + P[0][2])) / 2 / D, lw=[(b[2] - b[0]) / D for b in P]))
            if s['c'] == '-' and s['role'] == 'main':
                minus.append(dict(w=s['w'] / D, h=s['h'] / D, cy=((s['y0'] + s['y1']) / 2 - base(s['cx'])) / D))
    return bank, ex_stats, pair_gaps, gaps, pauses, fracs, pows, arrows, eqs, minus
b1 = run(0.85, {'=': -0.4, '-': -0.4}, [-0.16])
E1 = [e for e in b1[1] if not e['fallback']]
FS = float(np.median([f['scale'] for f in b1[5]])) if b1[5] else 0.85
OPC = {'=': float(np.median([e['cy'] for e in b1[8]])), '-': float(np.median([m['cy'] for m in b1[9]]))}
print('pass1 frac scale', FS, 'op cy', OPC)
bank, ex_stats, pair_gaps, gaps, pauses, fracs, pows, arrows, eqs, minus = run(FS, OPC, [e['slope'] for e in E1])
q = lambda v: {k: round(float(x), 4) for k, x in zip(['p5', 'p25', 'p50', 'p75', 'p95', 'mean', 'std'], list(np.percentile(v, [5, 25, 50, 75, 95])) + [np.mean(v), np.std(v)])} if len(v) else None
cls_stats = {c: dict(n=len(v), h=q([s['h'] for s in v]), w=q([s['w'] for s in v]), bot=q([s['bot'] for s in v]), top=q([s['top'] for s in v]), strokes=dict(collections.Counter(s['n'] for s in v)),
                     roles=dict(collections.Counter(s['role'] for s in v))) for c, v in sorted(bank.items())}
E = ex_stats
style = dict(
    note='TRAIN-only statistics of the owner\'s handwriting. Lengths are in units of D (median digit height of the exercise); y is relative to the fitted baseline (down = +).',
    n_exercises=len(E), train_ids_used=sorted({e['id'] for e in E}),
    D_px=q([e['D'] for e in E]), D_list=[round(e['D'], 2) for e in E],
    baseline=dict(slope=q([e['slope'] for e in E]), slope_list=[round(e['slope'], 4) for e in E], resid=q([e['resid'] for e in E if e['resid'] is not None]), resid_list=[round(e['resid'], 4) for e in E if e['resid'] is not None], n_fallback_baseline=sum(e['fallback'] for e in E)),
    slant=q([e['slant'] for e in E if e['slant'] is not None]), slant_list=[round(e['slant'], 4) for e in E if e['slant'] is not None],
    pressure=dict(mean=q([e['pmean'] for e in E]), std=q([e['pstd'] for e in E])),
    timing=dict(pause_ms=q(pauses), pause_list=[int(p) for p in pauses], dur_ms=q([e['dur'] for e in E])),
    start=dict(x=q([e['x_start'] for e in E]), y=q([e['y_start'] for e in E]), xy_list=[[round(e['x_start']), round(e['y_start'])] for e in E]),
    canvas=collections.Counter(tuple(e['canvas']) for e in E).most_common(), width_D=q([e['width'] for e in E]),
    classes=cls_stats,
    gaps={k: dict(q=q(v), list=[round(x, 3) for x in v]) for k, v in sorted(gaps.items())},
    pair_gaps={k: [round(x, 3) for x in v] for k, v in sorted(pair_gaps.items()) if len(v) >= 6},
    gaps_all=q([x for k, v in gaps.items() if k != 'arrow' for x in v]),
    touch_rate=round(float(np.mean([x < 0.03 for k, v in gaps.items() if k != 'arrow' for x in v])), 4),
    power=dict(n=len(pows), rel_top=q([p['rel_top'] for p in pows if p['rel_top'] is not None]), scale=q([p['scale'] for p in pows]), bot=q([p['bot'] for p in pows]), top=q([p['top'] for p in pows]), dx=q([p['dx'] for p in pows]), list=pows),
    fraction=dict(n=len(fracs), **{k: q([f[k] for f in fracs]) for k in ['bar_y', 'num_gap', 'den_gap', 'scale', 'over', 'num_dx', 'den_dx']},
                  inner_gap=q([g for f in fracs for g in f['num_gaps'] + f['den_gaps']]), list=fracs),
    equals=dict(n=len(eqs), w=q([e['w'] for e in eqs]), sep=q([e['sep'] for e in eqs]), cy=q([e['cy'] for e in eqs]), dx=q([e['dx'] for e in eqs]), top_vs_bottom_len=q([e['lw'][0] / max(1e-3, e['lw'][1]) for e in eqs])),
    minus=dict(n=len(minus), w=q([m['w'] for m in minus]), h=q([m['h'] for m in minus]), cy=q([m['cy'] for m in minus])),
    arrow=dict(n_drawn=len(arrows), n_prompts_with_arrow=sum('>' in items[r['id']]['promptText'] for r in AL)),
)
a, b = [], []
for c in '=':
    for smp in bank[c]:
        if smp['role'] != 'main': continue
        for st in smp['s']:
            P = np.array([p[:2] for p in st])
            if len(P) > 2 and np.ptp(P[:, 0]) > 0: a.append(np.arctan(smp['slope'])); b.append(np.arctan(np.polyfit(P[:, 0], P[:, 1], 1)[0]))
k = float(np.polyfit(a, b, 1)[0]); style['glyph_rotation_follows_baseline'] = dict(k=round(k, 3), how="'=' stroke angle vs baseline angle regression; glyphs turn with the line by this fraction")
style['equals']['angle_deg'] = q([float(np.degrees(v)) for v in b])
json.dump(style, open('style.json', 'w'), indent=1)
B = dict(note='Owner symbol bank, TRAIN items only. Per sample: strokes [[x,y,tMs,p]] with x from the symbol left edge and y from the local baseline, both in units of D; t from the symbol start.',
         symbols={c: v for c, v in sorted(bank.items())}, arrow=arrows)
json.dump(B, gzip.open('bank.json.gz', 'wt'))
srcs = {s['src'] for v in bank.values() for s in v} | {a['src'] for a in arrows}
assert srcs <= TRAIN and not (srcs & TEST)
print('exercises', len(E), 'sources', len(srcs), 'samples', sum(map(len, bank.values())))
print(' '.join(f"{c}:{len(v)}{'(THIN)' if len(v) < 10 else ''}" for c, v in sorted(bank.items(), key=lambda kv: -len(kv[1]))))
print('pow', len(pows), 'frac', len(fracs), 'eq', len(eqs), 'minus', len(minus), 'arrows', len(arrows), style['arrow'])
print('D', style['D_px']); print('slope', style['baseline']['slope']); print('touch', style['touch_rate'], 'gaps_all', style['gaps_all'])
