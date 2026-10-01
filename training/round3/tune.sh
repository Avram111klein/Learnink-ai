# usage: PARAMS='[...]' tune.sh tag  -> runs dev1200 + owner train, prints score per param set
S=__WORKDIR__; cd $S/hwdata/round3; export NODE_PATH=/home/claude/.npm-global/lib/node_modules
M=$S/trainer/model/current.json
node harness.js read hw_exp4.js $M dev5000.json ${DEV:-ids_dev1200.json} t_dev_$1.json >/dev/null &
node harness.js read hw_exp4.js $M ../round2/items.json ../round2/ids_train.json t_tr_$1.json >/dev/null; wait
python3 - $1 <<'PY'
import json,sys;from score import score
a=json.load(open(f't_dev_{sys.argv[1]}.json')); b=json.load(open(f't_tr_{sys.argv[1]}.json'))
P=json.loads(__import__('os').environ.get('PARAMS','[null]'))
if not isinstance(a[0],list): a,b=[a],[b]
for k,(x,y) in enumerate(zip(a,b)):
    s,t=score(x),score(y); print(json.dumps(P[k]) if P[k] else 'default','| dev sym %.4f ex %d fx %d sx %d | tr sym %.4f ex %d fx %d sx %d'%(s['sym'],s['nEx'],s['nFx'],s['nSx'],t['sym'],t['nEx'],t['nFx'],t['nSx']), t['top'][:3] if len(a)==1 else '')
PY
