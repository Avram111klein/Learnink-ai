"""Driver (gen.py untouched, default settings as sample500): shard k = 5000 exercises, ids SYN10K-<global index>, seed 10000+k."""
import sys, json, gzip, random, time, gen
k = int(sys.argv[1]); N = 5000; r = random.Random(10000 + k); t = time.time()
with gzip.open(f'data/shard_{k:02d}.jsonl.gz', 'wt') as f:
    for i in range(k * N, (k + 1) * N):
        e = gen.make(i, r); e['id'] = f'SYN10K-{i:05d}'; f.write(json.dumps(e, separators=(',', ':')) + '\n')
print(f'shard {k}: {N} in {time.time()-t:.0f}s', flush=True)
