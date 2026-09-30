# Learnink Handwriting Trainer

A standalone page (not part of the app). The writer copies printed exercises with a pen. The page reads them with the
app's reader (`training/handwriting/exp/hw_exp.js`, branch `wip/hw-algebra`, embedded verbatim; the only change is that the model
comes in through `makeHW(M)`), shows feedback, and stores every exercise centrally for later retraining.

## Publish (CTO)

Publish `index.html` together with `model/current.json` and `model/meta.json` (with the same relative paths). Do not publish `dev/` or `shots/`.

```js
capabilities: {
  db: { rules: [
    { path: "samples",        read: "admin", write: "admin" },
    { path: "samples/{self}", write: "interact" }
  ] },
  user: {}            // needed for user.id(), which addresses the viewer's own {self} subtree; no profile/email scopes
}
```

Each viewer writes only below `samples/<their id>`, and only the owner and editors can read all of it.
If `claude.use("db")` resolves `null`, or `user.id()` is `null`, or `window.claude` is missing (a local file), the page still works.
It keeps the samples in memory and shows "לא מחובר – הדגימות לא נשמרות".

## Stored documents

| path | what |
|---|---|
| `samples/<self>` | writer doc `{kind:"writer", devices:{<writerId>:{heldout,lastTs}}}`. The store cannot list sub-collections, so the retrainer queries `samples` and then each `samples/<id>/ex` |
| `samples/<self>/ex/<sessionId>-b<k>` | batch doc: `{kind:"batch", writerId, sessionId, batch:k, heldout, modelVersion, n, items:[…up to 10 exercises], ts}` |

**Batching.**
- A batch holds up to 10 exercises. It is rewritten whole (`set`, which is idempotent) at these times:
  - when it fills;
  - when the progress screen opens (a pause);
  - on `visibilitychange` to hidden and on `pagehide` (best effort);
  - every ~30 s while it has unsaved exercises.
- A batch is closed early if the next exercise would take it past ~150 KB. One exercise's strokes are thinned to at most ~60 KB, so every doc stays well under the 256 KiB limit.
- A closed batch is never reopened. Each session therefore leaves at most one batch that is not full.
- If the tab is killed, at most ~30 s of work can be lost. This happens only if `pagehide` could not complete its write.
- When a write fails, the exercises stay pending. The banner turns amber and reads "השמירה נכשלה – X תרגילים ממתינים, ננסה שוב"; on `quota_exceeded` it says the storage is full. The 30 s timer retries, and the banner turns green again after the next successful save.
- A body rejected with `invalid_argument` is never resent unchanged. Its unsaved exercises move to smaller closed batches. A single rejected exercise is dropped with a visible note.

**Each item in `items`** (one exercise):
- `id` (`<sessionId>-<seq>`), `writerId`, `sessionId`, `seq`, `heldout` (`fnv1a32(writerId) % 5 === 0`).
- `promptText`: the mini-language source. `[n/d]` is a stacked fraction, `^2` a power, and `>` the grey arrow, which is not copied.
- `targetText`, `targetTokens`.
- `strokes`: `[[x,y,tMs,pressure*100],…]` per stroke, integers in CSS px from the canvas top-left.
  - At most ~3,000 points and ~60 KB per exercise. Long strokes are thinned first, then taps and short strokes.
  - `thinned:true` marks an exercise that was thinned. Its stored strokes are then fewer than the reader saw; otherwise they are exactly what the reader saw.
- `readText`, `readTokens`, `readConf`, `readAlts`.
- `reader {code, atype:"alg", vars}`.
- `exact`, `discarded` (`true` after "טעיתי בהעתקה – דלג"), `warmup`, `template`.
- `modelVersion`.
- `input`: pen, touch, mouse or mixed, taken from the strokes actually kept. Palm rejection works like this:
  - Once a pen has been seen, touch is ignored. This is remembered on the device in `lnkt.penSeen`.
  - When a pen goes down, touch strokes of the current exercise are discarded.
- `device {w,h,sw,sh,dpr,ua}`.
- `durMs`, `ts`.

The docs hold no names and no claude user id. Only the path is scoped by `{self}`.

**Capacity.** An artifact's db holds at most 5,000 docs.
- Take one writer doc per writer, plus about 9 batch docs per 90-exercise session (10 per doc, one partial batch per session).
- That leaves room for about 49,000 exercises, or about 550 sessions.
- Very dense writing makes batches close early, which lowers this somewhat.
- `quota_exceeded` shows a banner.

## Model

`model/current.json` is `model_alg_r3.json`: 28 classes, int8, calibrated, trained without CROHME.
`model/meta.json` holds version 1 and its metrics from `qa_out/r3_exp.json`:
- `answerAcc`: real handwritten answers read exactly.
- `symbolAcc`: real single symbols.
- `falseX`: guarded sure-wrong at p≥0.5.

It also holds `symAcc` and `topConf`. The exercise generator uses them as a prior for which symbols are weak. The running counts of the session then take over.
To ship a new round, replace both files. The page shows `meta.version` and stamps it on every doc.

## Dev

- `node dev/build.js` rebuilds `index.html` from `dev/index.src.html` and `dev/hw_exp.js`.
- `node dev/serve.js 8765 &` then `NODE_PATH=$(npm root -g) node dev/test.js` runs the end-to-end test. It uses a fake db that enforces the rules, replays synthetic pen strokes, checks batching and the flush triggers, and checks parity against hw_exp.js. Screenshots go to `shots/`.
