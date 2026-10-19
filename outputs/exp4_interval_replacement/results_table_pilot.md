| Method | Replaced interval | Segment steps / GMMD configuration | Number of images | FID ↓ | Precision ↑ | Recall ↑ | Segment cost ↓ | Full generation cost ↓ | Peak memory |
|---|---|---|---|---|---|---|---|---|---|
| A. Original teacher | none (original) | 40 fine SMLD steps (original) | 192 | 56.57 ± 2.16 | 0.776 | 0.542 | 1.46 s/img, 40 NFE/img | 73.2 s/img, 2000 NFE/img | 48.4 GB |
| B. Coarse teacher | grid 800→840 (t 0.5998→0.5798) | 4 coarse SMLD steps | 192 | 57.99 ± 2.15 | 0.719 | 0.458 | 0.15 s/img, 4 NFE/img | 72.3 s/img, 1964 NFE/img | 48.4 GB |
| C. GMMD replacement | grid 800→840 (t 0.5998→0.5798) | IBW K=1, B=4, 300 it × 0.1, prior-free init, 1 draw | 192 | 58.05 ± 2.30 | 0.786 | 0.500 | 43.75 s/img, 1200 NFE/img | 116.0 s/img, 3160 NFE/img | 49.9 GB |
