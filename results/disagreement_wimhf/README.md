# Experiment 4: Native Disagreement WIMHF

Command:

```bash
scripts/run_disagreement_wimhf.py --output-dir wimhf/outputs/disagreement_wimhf
```

`response_A` is the human-preferred response and `response_B` is the
human-rejected response. `delta_r_rm = rm_reward_human_preferred -
rm_reward_human_rejected`; negative values are RM disagreement.

Rows: 37004
Agree: 26176
Disagree: 10651
Zero-margin: 177

Matched before row cap: 37004
Matched by source_index: 37004
Matched by content hash fallback: 0
Unmatched RM rows: 0

Stage: `full_wimhf`
