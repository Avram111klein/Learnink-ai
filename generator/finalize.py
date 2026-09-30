"""Verify the shards (sources TRAIN-only, no TEST ids, unique ids), write the 5,000 dev index, then READY.json LAST."""
import json, gzip, hashlib, random, datetime, collections, sys
SP = '__WORKDIR__'
sp = json.load(open(SP + '/hwdata/round2/split.json')); TR = set(sp['train']); TEST = set(sp['testA']) | set(sp['testB'])
shards = ['data/shard_00.jsonl.gz', 'data/shard_01.jsonl.gz']; rows = []; ids = set(); hard = 0
for s in shards:
    for n, l in enumerate(gzip.open(s, 'rt')):
        e = json.loads(l); src = set(e['sources'])
        assert src and src <= TR and not (src & TEST), e['id']
        assert e['id'] not in ids; ids.add(e['id']); rows.append((e['id'], s.split('/')[-1], n, e['template'])); hard += e['template'].startswith('hard')
assert len(rows) == 10000
B = json.load(gzip.open('bank.json.gz', 'rt'))
bsrc = {x['src'] for v in B['symbols'].values() for x in v} | {a['src'] for a in B['arrow']}
assert bsrc <= TR and not (bsrc & TEST)
r = random.Random(77); by = collections.defaultdict(list)
for row in rows: by[row[3]].append(row)
dev = []
for t, L in sorted(by.items()): r.shuffle(L); dev += L[:round(len(L) / 2)]
r.shuffle(dev); dev = dev[:5000]; have = {d[0] for d in dev}
for x in rows:
    if len(dev) >= 5000: break
    if x[0] not in have: dev.append(x); have.add(x[0])
assert len(dev) == 5000 == len(have)
json.dump(dict(note='5,000-exercise dev subset of the 10k synthetic set, stratified by template', n=len(dev), items=[dict(id=i, shard=s, line=n) for i, s, n, _ in dev]), open('data/dev5000_index.json', 'w'))
sha = {s.split('/')[-1]: hashlib.sha256(open(s, 'rb').read()).hexdigest() for s in shards}
READY = dict(count=10000, shards=[s.split('/')[-1] for s in shards], sha256=sha, devIndex='dev5000_index.json', devCount=len(dev),
             sourcesAreTrainOnly=True, testIdsChecked=True, bankSize=sum(len(v) for v in B['symbols'].values()), bankArrows=len(B['arrow']),
             hardMixShare=hard / 10000, generator=sys.argv[1] if len(sys.argv) > 1 else 'gen.py', generatedAt=datetime.datetime.now(datetime.timezone.utc).isoformat())
json.dump(READY, open('READY.json', 'w'), indent=1)
print(json.dumps(READY))
