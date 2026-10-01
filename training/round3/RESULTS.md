# Reader v4 (hw_exp4.js) – results, model r5 unchanged

Tuning used only: synthetic dev (500 of dev5000) + owner TRAIN (229). TEST-A+B (57) was read once per attempt.

| gate item | baseline r5 + hw_exp | attempt 1 (V4a–e) | attempt 2 (V4a–f) |
|---|---|---|---|
| 1. exact whole answers, TEST-A+B | 7/57 | 17/57 | **17/57** |
| 2. symbol %, TEST-A+B | 73.9 | 87.8 | **87.8** |
| 3a. false ✗, shape rule | 7/57 | 8/57 ✗ | **2/57** |
| 3b. false ✗, strict rule | 1/57 | 4/57 ✗ | **1/57** |
| 4. pen digits single / 3-digit | 96.25 / 91.73 | 96.25 / 91.73 | **96.25 / 91.73** |
| 5. drawn symbols (frozen) | 97.1 | 97.1 | **97.1** |
| 5. frozen pen answers exact | 68.7 | 74.0 | **74.0** |
| 5. frozen composed answers exact | 24.8 | 26.2 | **26.2** |
| 5. CROHME answers exact (eval only) | 9.1 | 9.8 | **9.8** |
| (info) sure-exact = a ✓ would be given | 7/57 | 16/57 | 12/57 |
| (info) unsure answers | 13/57 | 9/57 | 36/57 |

Train-side (for reference): owner TRAIN exact 28→84/229, symbols 70.6→86.0%, shape false ✗ 31→7, strict 7→2;
synthetic dev500 exact 48→182, symbols 79.9→89.0%, shape false ✗ 92→27, strict 18→6.

Rules: V4a ÷ assembly (dash+dots, or this writer's stacked ":"), V4b first-pass guard keeping "(" ")" and
superscripts apart unless they overlap ≥0.8 or touch, V4c classifier-scored merge (two parts that each read
confidently and do not touch stay apart), V4e drop a copied "⟶" arrow, V4f answer-confidence remap (raw 0.65 → 0.5).
V4d (baseline-relative powers) was tried and is OFF (it hurt on train-side); powers use the EXP2 rule.
