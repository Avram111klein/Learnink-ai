"""Blind look-test (usage: looktest.py <samples.jsonl.gz> <outdir> <seed> [prev_KEY.json[,prev2_KEY.json...] to avoid their real items]): 24 synthetic + 24 real TRAIN exercises, same renderer, shuffled -> looktest/lt_XX.png; key outside the folder; numbered contact sheet."""
import json, gzip, glob, random, os, sys
SRC, OUT, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]); PREV = sys.argv[4] if len(sys.argv) > 4 else None
from PIL import Image, ImageDraw, ImageFont
import render
SP = '__WORKDIR__'
sp = json.load(open(SP + '/hwdata/round2/split.json')); TR = set(sp['train']); TEST = set(sp['testA']) | set(sp['testB'])
real = [x for f in sorted(glob.glob(SP + '/trainer_data2/samples/__WRITER__/ex/*.json')) for x in json.load(open(f))['items'] if x['id'] in TR and not x['discarded'] and not x['warmup']]
assert not any(x['id'] in TEST for x in real)
syn = [json.loads(l) for l in gzip.open(SRC, 'rt')]
r = random.Random(SEED)
if PREV:
    used = {v['id'] for p in PREV.split(',') for v in json.load(open(p)).values() if v['kind'] == 'real'}; real = [x for x in real if x['id'] not in used]
pick = [('real', x) for x in r.sample(real, 24)] + [('synthetic', x) for x in r.sample(syn, 24)]
r.shuffle(pick)
os.makedirs(OUT, exist_ok=True)
for f in glob.glob(OUT + '/*.png'): os.remove(f)
key, ims = {}, []
for i, (kind, x) in enumerate(pick, 1):
    im = render.render([s for s in x['strokes'] if s], pen=2.4); name = f'lt_{i:02d}.png'; im.save(OUT + '/' + name); ims.append(im)
    key[name] = dict(kind=kind, id=x['id'], prompt=x['promptText'])
json.dump(key, open(OUT + '_KEY.json', 'w'), indent=1)
cols, tw, th = 4, 450, 150
sheet = Image.new('L', (cols * tw, (len(ims) + cols - 1) // cols * th), 255); d = ImageDraw.Draw(sheet)
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
except Exception: font = ImageFont.load_default()
for i, im in enumerate(ims):
    x, y = (i % cols) * tw, (i // cols) * th
    sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y)); d.rectangle([x, y, x + tw - 1, y + th - 1], outline=180); d.text((x + 6, y + 4), f'{i+1:02d}', fill=0, font=font)
sheet.save(OUT + '_sheet.png')
print('real', sum(v['kind'] == 'real' for v in key.values()), 'synthetic', sum(v['kind'] == 'synthetic' for v in key.values()))
