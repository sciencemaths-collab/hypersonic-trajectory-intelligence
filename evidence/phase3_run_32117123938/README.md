# HTI 0.8 frozen Phase 3 evidence

This directory contains the complete per-seed evaluator reports and aggregate
summary derived from the immutable prediction bundles uploaded by GitHub Actions
run [32117123938](https://github.com/sciencemaths-collab/hypersonic-trajectory-intelligence/actions/runs/32117123938).

## Frozen boundary

- Bundle-producing commit: `eac6b47c81255285012b0183c46d32c4d93f905d`
- Evaluator commit: `e873767f8c674fed07a4152ba153c6fce083d827`
- Frozen seeds: `101, 211, 307, 401, 503`
- Protocol SHA-256: `2155ddd14aba790d31988c728f7c6d3e8bb0a8c4389d671a5e4ca204c7b63bbb`
- Execution-config SHA-256: `f34489298c160ffdcedba8bfe18cad39f270bdd61788e0d6ffbc47bca75118e3`

No seed was regenerated and no model was retrained during report publication.
The evaluator commit corrects only the float32 serialization contract and
separates bundle-producing provenance from evaluator provenance.

## Claim-gate result

| Horizon | Core NLL | Full HTI NLL | Full − Core | Core ECE | Full ECE | Bootstrap support |
|---:|---:|---:|---:|---:|---:|---:|
| 0.4 s | 1.7879 | 1.9082 | +0.1203 | 0.1076 | 0.2925 | 0/5 |
| 0.5 s | 1.7052 | 1.8302 | +0.1250 | 0.1120 | 0.2932 | 0/5 |
| 0.6 s | 1.6823 | 1.7962 | +0.1139 | 0.1030 | 0.2885 | 0/5 |

The frozen experiment does not support a predictive-improvement claim for Full
HTI 0.8. Full HTI NLL was worse for every seed and horizon, ECE non-regression
failed throughout, all-seed class coverage failed at 0.5 and 0.6 seconds, and
the all-seed credible-region gate failed at every horizon.

The negative result is retained as falsification evidence. Final-test outcomes
must not be used to tune the frozen parameters or replace these seeds.

See `provenance.json` for immutable bundle and derived-report checksums.
