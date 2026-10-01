# usage: attempt_eval.sh <reader.js> <tag> : owner TEST-A+B once + frozen/pen-digit checks (model r5)
S=__WORKDIR__; cd $S/hwdata/round3; export NODE_PATH=/home/claude/.npm-global/lib/node_modules
M=$S/trainer/model/current.json; R=$(readlink -f $1)
python3 -c "import json;a=json.load(open('../round2/ids_testA.json'));b=json.load(open('../round2/ids_testB.json'));json.dump(a+b,open('ids_testAB.json','w'))"
node harness.js read $R $M ../round2/items.json ids_testAB.json test_$2.json >/dev/null &
(cd $S/hw/training/handwriting && HWJS=$R node eval_alg.js $M ../../testset_frozen $S/hw_prev_0927/digits_strokes.json --json $S/hwdata/round3/frozen_$2.json >/dev/null 2>&1); wait
python3 - $2 <<'PY'
import json,sys;from score import score
t=sys.argv[1]; s=score(json.load(open(f'test_{t}.json'))); f=json.load(open(f'frozen_{t}.json'))
print(f"{t}: TEST-A+B n={s['n']} exact {s['nEx']} sym {s['sym']:.4f} falseX(shape) {s['nFx']} strict {s['nSx']} | digits {f['digits_ref']['single']}/{f['digits_ref']['triple']} | drawnSym {f['sym_synth']['overall']} realSym {f['sym_real']['overall']} | expr pen {f['expr_pen']['exact']} composed {f['expr_synth']['exact']} crohme {f['expr_real']['exact']} | errs {f['errors'][:1]}")
print('  top', s['top'][:6])
PY
