"""segmentation diagnosis on owner TRAIN items that forced-aligned: which tokens get merged/split by the reader."""
import json,sys,collections
FA={r['id']:r for r in json.load(open('../round2/falign_train_relax.json')) if r['ok']}
G={g['id']:g for g in json.load(open(sys.argv[1]))}
key=lambda pts:(round(pts[0][0],1),round(pts[0][1],1),len(pts))
cat=collections.Counter(); pairs=collections.Counter()
for i,r in FA.items():
    g=G[i]; owner={}
    for ti,q in enumerate(r['per']):
        for s in r['S'][q['j']:q['k']]: owner[key(s)]=ti
    gi={}
    for k,grp in enumerate(g['G']):
        for s in grp['s']: gi[key(s)]=k
    for ti,q in enumerate(r['per']):
        t=q['t'].replace('^','') if not r['bar'][ti] else 'bar'
        gs={gi.get(key(s)) for s in r['S'][q['j']:q['k']]}
        if None in gs: cat[(t,'lost')]+=1; continue
        if len(gs)>1: cat[(t,'split')]+=1; continue
        k=gs.pop(); others={owner.get(key(s)) for s in g['G'][k]['s']}-{ti}
        if others:
            cat[(t,'merged')]+=1
            for o in others: pairs[t+'+'+(r['per'][o]['t'].replace('^','') if not r['bar'][o] else 'bar')]+=1
        else: cat[(t,'ok')]+=1
tot=collections.Counter(); 
for (t,c),n in cat.items(): tot[c]+=n
print(tot)
bad=collections.Counter({k:v for k,v in cat.items() if k[1]!='ok'}); print(bad.most_common(20)); print(pairs.most_common(20))
