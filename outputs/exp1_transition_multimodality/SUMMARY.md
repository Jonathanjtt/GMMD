# Experiment 1 — transition length and conditional multimodality

Ran 3 fixed noisy states × 4 endpoints × 64 paired rollouts.
DINOv2 ViT-B/14 was fixed before analysis; negative BIC(2)−BIC(1) favors two components.

| image | s | Δt | preferred K (BIC) | BIC(2)−BIC(1) | bootstrap ARI |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.5798 | 0.0200 | 1 | 43.33 | 0.412 |
| 0 | 0.4998 | 0.1000 | 1 | 28.04 | 0.467 |
| 0 | 0.4502 | 0.1496 | 3 | -39.85 | 0.631 |
| 0 | 0.3502 | 0.2496 | 2 | -49.85 | 0.631 |
| 137 | 0.5798 | 0.0200 | 4 | 23.58 | 0.885 |
| 137 | 0.4998 | 0.1000 | 1 | 35.67 | 0.299 |
| 137 | 0.4502 | 0.1496 | 1 | 62.29 | 0.589 |
| 137 | 0.3502 | 0.2496 | 1 | 18.78 | 0.416 |
| 1024 | 0.5798 | 0.0200 | 1 | 36.55 | 0.669 |
| 1024 | 0.4998 | 0.1000 | 1 | 29.10 | 0.469 |
| 1024 | 0.4502 | 0.1496 | 1 | 13.75 | 0.675 |
| 1024 | 0.3502 | 0.2496 | 2 | -3.08 | 0.519 |

Figures: `figures/figure_1a_*`, `figure_1b_*`, and `figure_1c_*`.
