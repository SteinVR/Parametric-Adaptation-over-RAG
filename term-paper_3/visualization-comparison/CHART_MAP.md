# Chart map

| Section | Analytical question | Family / type | Data | Supported takeaway |
|---|---|---|---|---|
| Current figures | What does the draft show now? | Embedded original PNGs | `assets/figures/fig01`–`fig05` | Preserves the current visual baseline without editing it. |
| Bootstrap effect | How uncertain are gains over Base-RAG? | Distribution / box plot | Per-question scores, 500 paired stratified bootstrap replicates | Small aggregate gains should be read with sampling uncertainty. |
| Seed stability | How much does training stochasticity move scores? | Comparison / grouped bar | Three seed-level Q_main values | Merge-RAG varies more across adapter pairings. |
| Retrieval contribution | Does adaptation replace retrieval? | Comparison / grouped bar | Matched retrieval on/off systems | Retrieval contributes far more than generator adaptation alone. |
| Evidence coverage | Where do gains appear relative to retrieved gold pages? | Comparison / grouped bar | Full, partial, and zero gold-page coverage | Fixed retrieval still creates materially different evidence regimes. |
| Answer-type deltas | Which answer types improve or regress? | Comparison / grouped horizontal bar | Per-type score minus Base-RAG | RAFT and CLM specialize differently; multi-name remains weak. |
| Pairwise outcomes | Are gains broad or localized? | Composition / 100% stacked bar | Wins, ties, losses vs Base-RAG | Most questions tie, so aggregate gains come from a minority of items. |

Palette policy: Base-RAG blue, RAFT orange, CLM green, Merge red; neutral grey for references and ties. New native charts also use non-color structure through grouping, labels, ordering, or reference lines.
