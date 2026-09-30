"""Label owner symbols: reader groups (reading order) aligned 1:1 with target tokens; geometry must agree
(target '-', '=', fraction bar <-> flat group). Only TRAIN-side exercises."""
import json, collections
G = json.load(open('groups_train.json')); split = json.load(open('split.json'))
fit, val = set(split['train_fit']), set(split['train_val'])
sy, ok, why = [], 0, collections.Counter()
for e in G:
    T, bar, gs = e['T'], e['bar'], e['G']
    if len(gs) != len(T):
        why['count %+d' % (len(gs) - len(T))] += 1; continue
    good = True
    for t, b, g in zip(T, bar, gs):
        geo = t in '-=' or b
        if geo != g['dash'] and not (t == '=' and len(g['s']) == 2):
            good = False; break
        if t == '.' and not g['tiny']:
            good = False; break
    if not good:
        why['geometry'] += 1; continue
    ok += 1
    for t, b, g in zip(T, bar, gs):
        c = t.replace('^', '')
        if b or c in '-=.':
            continue
        sy.append({'c': c, 's': g['s'], 'id': e['id'], 'part': 'fit' if e['id'] in fit else 'val', 'pow': t.startswith('^')})
json.dump(sy, open('owner_symbols_train.json', 'w'))
print('aligned', ok, '/', len(G), '=%.1f%%' % (100 * ok / len(G)), dict(why))
print('symbols', len(sy), 'fit', sum(s['part'] == 'fit' for s in sy), 'val', sum(s['part'] == 'val' for s in sy))
print(sorted(collections.Counter(s['c'] for s in sy).items(), key=lambda x: -x[1]))
