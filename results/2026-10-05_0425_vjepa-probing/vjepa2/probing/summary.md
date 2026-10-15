# V-JEPA 2 encoder probing

254 right-segment clips, 5-fold CV (`cv_folds`, seed 0). Every probe is fitted on real context-half features (encoded alone) with same-frame ball cells (`eval_decoder.examples_from`), then scored on held-out clips: context frames, real target frames (far-side cells never had a ball in training) and, for the last layer, the pretrained predictor's imagined target.

## Reference: tracker + rules

| variant | outcome AUROC | P(correct) | balanced acc | tgt cell AUROC | far cell AUROC | hit |
|---|---|---|---|---|---|---|
| continue | 0.539 | 0.333 | 0.333 | 0.597 | 0.814 | 0.227 |
| blockade (blockade fitted on outcomes) | 0.838 | 0.531 | 0.540 | 0.554 | 0.814 | 0.222 |

TAPNext + coded blockade on the earlier 88-clip set: outcome AUROC 0.98 (`docs/results/ball_comparison.md`).

## Factor effects (paired: configs identical except for one factor)

| factor | change | far cell AUROC | outcome AUROC (real) | tgt hit | imag outcome AUROC |
|---|---|---|---|---|---|
| model | mlp - linear | +0.000 (24/32) | -0.020 (23/32) | +0.026 (19/32) | -0.083 (6/12) |
| input | nbhd - token | +0.008 (34/38) | +0.001 (22/38) | -0.008 (9/38) | -0.057 (6/8) |
| input | global - token | -0.490 (0/28) | -0.398 (0/28) | -0.875 (0/28) | -0.099 (2/8) |
| norm | ln - raw | +0.001 (20/42) | +0.003 (23/42) | -0.001 (12/42) | -0.002 (4/12) |
| head | softmax - sigmoid | -0.023 (5/52) | +0.025 (32/52) | -0.035 (5/52) | +0.017 (6/12) |

Each cell: mean change, and how many pairs improved out of all pairs.

## All configs

| config | ctx cell AUROC | tgt cell AUROC | far cell AUROC | tgt hit | far hit | err px | outcome AUROC | P(correct) | imag far cell | imag outcome | imag P(correct, 1 ball) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L12-mlp-nbhd-ln-sigmoid | 1.000 | 0.997 | 0.999 | 0.997 | 1.000 | 16.300 | 1.000 | 0.630 |  |  |  |
| L8-mlp-nbhd-ln-sigmoid | 1.000 | 0.995 | 0.999 | 0.997 | 1.000 | 17.400 | 1.000 | 0.689 |  |  |  |
| L12-linear-nbhd-ln-sigmoid | 1.000 | 0.998 | 0.999 | 0.996 | 1.000 | 19.100 | 0.998 | 0.665 |  |  |  |
| L12-linear-nbhd-raw-sigmoid | 1.000 | 0.998 | 0.999 | 0.996 | 1.000 | 20.000 | 0.996 | 0.668 |  |  |  |
| L16-mlp-nbhd-ln-sigmoid | 1.000 | 0.997 | 0.999 | 0.997 | 1.000 | 14.000 | 1.000 | 0.615 |  |  |  |
| L20-linear-nbhd-raw-sigmoid | 1.000 | 0.997 | 0.999 | 0.995 | 1.000 | 17.400 | 0.994 | 0.671 |  |  |  |
| L8-linear-nbhd-ln-sigmoid | 1.000 | 0.997 | 0.999 | 0.999 | 1.000 | 18.400 | 0.996 | 0.704 |  |  |  |
| L8-linear-nbhd-raw-sigmoid | 1.000 | 0.997 | 0.999 | 0.998 | 1.000 | 18.900 | 0.997 | 0.707 |  |  |  |
| L12-mlp-token-ln-sigmoid | 1.000 | 0.978 | 0.999 | 1.000 | 1.000 | 17.500 | 1.000 | 0.655 |  |  |  |
| L16-linear-nbhd-ln-sigmoid | 1.000 | 0.998 | 0.999 | 0.996 | 1.000 | 16.200 | 0.994 | 0.630 |  |  |  |
| L16-linear-nbhd-raw-sigmoid | 1.000 | 0.998 | 0.999 | 0.997 | 1.000 | 16.400 | 0.994 | 0.630 |  |  |  |
| L20-linear-nbhd-ln-sigmoid | 1.000 | 0.997 | 0.999 | 0.994 | 1.000 | 16.600 | 0.994 | 0.674 |  |  |  |
| last-linear-nbhd-ln-sigmoid | 1.000 | 0.994 | 0.999 | 0.993 | 1.000 | 16.400 | 1.000 | 0.676 | 0.448 | 0.336 | 0.328 |
| last-linear-nbhd-raw-sigmoid | 1.000 | 0.994 | 0.999 | 0.993 | 1.000 | 16.400 | 1.000 | 0.675 | 0.445 | 0.335 | 0.329 |
| L16-mlp-token-ln-sigmoid | 0.999 | 0.975 | 0.999 | 1.000 | 1.000 | 15.600 | 1.000 | 0.663 |  |  |  |
| L20-mlp-nbhd-ln-sigmoid | 1.000 | 0.996 | 0.999 | 0.998 | 0.998 | 14.600 | 0.997 | 0.664 |  |  |  |
| L4-linear-nbhd-ln-sigmoid | 1.000 | 0.998 | 0.999 | 0.997 | 0.996 | 19.700 | 0.995 | 0.663 |  |  |  |
| L4-linear-nbhd-raw-sigmoid | 1.000 | 0.998 | 0.999 | 0.997 | 0.994 | 19.700 | 0.994 | 0.665 |  |  |  |
| L4-mlp-nbhd-ln-sigmoid | 1.000 | 0.995 | 0.999 | 0.986 | 0.994 | 18.400 | 0.992 | 0.707 |  |  |  |
| last-mlp-nbhd-ln-sigmoid | 1.000 | 0.995 | 0.999 | 0.998 | 1.000 | 15.000 | 1.000 | 0.664 | 0.680 | 0.477 | 0.337 |
| last-mlp-nbhd-raw-sigmoid | 1.000 | 0.995 | 0.999 | 0.997 | 1.000 | 15.100 | 1.000 | 0.668 | 0.681 | 0.451 | 0.337 |
| last-mlp-token-ln-sigmoid | 0.999 | 0.959 | 0.999 | 0.999 | 0.998 | 17.400 | 0.997 | 0.611 | 0.653 | 0.405 | 0.330 |
| last-mlp-token-raw-sigmoid | 0.999 | 0.959 | 0.999 | 0.999 | 0.998 | 17.600 | 0.997 | 0.608 | 0.644 | 0.430 | 0.330 |
| L8-mlp-nbhd-ln-softmax | 1.000 | 0.989 | 0.998 | 0.992 | 0.994 | 18.800 | 0.997 | 0.630 |  |  |  |
| L12-linear-token-ln-sigmoid | 0.999 | 0.969 | 0.998 | 1.000 | 1.000 | 16.200 | 0.997 | 0.641 |  |  |  |
| L20-mlp-token-ln-sigmoid | 0.999 | 0.960 | 0.998 | 1.000 | 1.000 | 17.000 | 0.997 | 0.656 |  |  |  |
| L12-linear-token-raw-sigmoid | 0.999 | 0.969 | 0.998 | 1.000 | 1.000 | 14.500 | 0.997 | 0.641 |  |  |  |
| L12-mlp-nbhd-ln-softmax | 1.000 | 0.991 | 0.998 | 0.992 | 0.990 | 22.300 | 0.998 | 0.623 |  |  |  |
| L16-mlp-nbhd-ln-softmax | 0.999 | 0.993 | 0.998 | 0.997 | 0.994 | 17.200 | 0.999 | 0.690 |  |  |  |
| L16-linear-token-ln-sigmoid | 0.998 | 0.961 | 0.997 | 1.000 | 1.000 | 14.700 | 0.990 | 0.646 |  |  |  |
| L16-linear-token-raw-sigmoid | 0.998 | 0.961 | 0.997 | 1.000 | 1.000 | 14.300 | 0.989 | 0.648 |  |  |  |
| L16-mlp-token-ln-softmax | 0.998 | 0.963 | 0.997 | 0.991 | 0.984 | 26.000 | 1.000 | 0.709 |  |  |  |
| last-linear-token-ln-sigmoid | 0.997 | 0.942 | 0.997 | 1.000 | 1.000 | 18.600 | 0.991 | 0.650 | 0.497 | 0.321 | 0.332 |
| last-linear-token-raw-sigmoid | 0.997 | 0.942 | 0.997 | 1.000 | 1.000 | 18.500 | 0.991 | 0.647 | 0.497 | 0.334 | 0.332 |
| L12-mlp-token-ln-softmax | 0.999 | 0.967 | 0.997 | 0.989 | 0.980 | 25.800 | 0.999 | 0.696 |  |  |  |
| L20-mlp-nbhd-ln-softmax | 0.999 | 0.991 | 0.997 | 0.990 | 0.974 | 17.800 | 1.000 | 0.697 |  |  |  |
| L20-mlp-token-ln-softmax | 0.997 | 0.951 | 0.997 | 0.996 | 0.988 | 25.000 | 1.000 | 0.719 |  |  |  |
| last-mlp-token-ln-softmax | 0.997 | 0.945 | 0.997 | 0.991 | 0.974 | 25.900 | 0.999 | 0.700 | 0.700 | 0.524 | 0.334 |
| last-mlp-token-raw-softmax | 0.997 | 0.945 | 0.997 | 0.990 | 0.972 | 25.700 | 0.999 | 0.700 | 0.694 | 0.529 | 0.336 |
| L8-mlp-token-ln-sigmoid | 1.000 | 0.958 | 0.997 | 1.000 | 1.000 | 19.500 | 0.991 | 0.651 |  |  |  |
| last-mlp-nbhd-ln-softmax | 0.999 | 0.988 | 0.997 | 0.990 | 0.990 | 18.500 | 1.000 | 0.690 | 0.659 | 0.228 | 0.305 |
| last-mlp-nbhd-raw-softmax | 0.999 | 0.988 | 0.997 | 0.990 | 0.990 | 18.400 | 1.000 | 0.691 | 0.661 | 0.231 | 0.310 |
| L20-linear-token-ln-sigmoid | 0.998 | 0.929 | 0.996 | 1.000 | 1.000 | 16.900 | 0.997 | 0.661 |  |  |  |
| L8-linear-token-ln-sigmoid | 0.999 | 0.960 | 0.996 | 1.000 | 1.000 | 17.800 | 0.992 | 0.636 |  |  |  |
| L8-linear-token-raw-sigmoid | 0.999 | 0.961 | 0.996 | 1.000 | 1.000 | 16.500 | 0.992 | 0.636 |  |  |  |
| L4-mlp-nbhd-ln-softmax | 1.000 | 0.971 | 0.996 | 0.966 | 0.974 | 24.300 | 0.998 | 0.705 |  |  |  |
| L20-linear-token-raw-sigmoid | 0.998 | 0.930 | 0.996 | 1.000 | 1.000 | 15.600 | 0.995 | 0.660 |  |  |  |
| L12-linear-nbhd-ln-softmax | 0.997 | 0.985 | 0.993 | 0.981 | 0.994 | 20.200 | 0.994 | 0.586 |  |  |  |
| L12-linear-nbhd-raw-softmax | 0.996 | 0.986 | 0.993 | 0.981 | 0.990 | 20.600 | 0.993 | 0.597 |  |  |  |
| L8-linear-nbhd-ln-softmax | 0.997 | 0.983 | 0.992 | 0.969 | 0.994 | 20.600 | 0.996 | 0.589 |  |  |  |
| L8-mlp-token-ln-softmax | 0.999 | 0.934 | 0.991 | 0.986 | 0.986 | 26.400 | 0.998 | 0.683 |  |  |  |
| L20-linear-token-raw-softmax | 0.983 | 0.934 | 0.991 | 0.957 | 0.918 | 23.500 | 0.998 | 0.630 |  |  |  |
| L8-linear-nbhd-raw-softmax | 0.997 | 0.982 | 0.990 | 0.963 | 0.990 | 21.100 | 0.997 | 0.606 |  |  |  |
| L20-linear-token-ln-softmax | 0.983 | 0.929 | 0.988 | 0.935 | 0.902 | 24.500 | 0.996 | 0.606 |  |  |  |
| L20-linear-nbhd-raw-softmax | 0.987 | 0.977 | 0.987 | 0.895 | 0.886 | 25.200 | 1.000 | 0.596 |  |  |  |
| L12-linear-token-ln-softmax | 0.987 | 0.937 | 0.987 | 0.979 | 0.984 | 28.600 | 0.996 | 0.635 |  |  |  |
| L20-linear-nbhd-ln-softmax | 0.987 | 0.974 | 0.987 | 0.887 | 0.878 | 24.700 | 0.999 | 0.583 |  |  |  |
| L16-linear-nbhd-raw-softmax | 0.994 | 0.981 | 0.986 | 0.901 | 0.880 | 24.600 | 0.996 | 0.625 |  |  |  |
| L12-linear-token-raw-softmax | 0.986 | 0.938 | 0.986 | 0.979 | 0.984 | 29.300 | 0.997 | 0.628 |  |  |  |
| L16-linear-nbhd-ln-softmax | 0.993 | 0.978 | 0.985 | 0.885 | 0.862 | 24.200 | 0.994 | 0.599 |  |  |  |
| L16-linear-token-raw-softmax | 0.984 | 0.942 | 0.985 | 0.961 | 0.978 | 27.200 | 0.997 | 0.636 |  |  |  |
| last-linear-nbhd-ln-softmax | 0.985 | 0.973 | 0.984 | 0.871 | 0.821 | 24.300 | 0.997 | 0.584 | 0.700 | 0.520 | 0.336 |
| last-linear-nbhd-raw-softmax | 0.985 | 0.972 | 0.984 | 0.855 | 0.825 | 24.900 | 0.997 | 0.574 | 0.696 | 0.523 | 0.336 |
| L16-linear-token-ln-softmax | 0.984 | 0.936 | 0.983 | 0.935 | 0.925 | 26.500 | 0.995 | 0.615 |  |  |  |
| L4-linear-nbhd-ln-softmax | 0.988 | 0.973 | 0.982 | 0.949 | 0.941 | 25.500 | 0.999 | 0.564 |  |  |  |
| L4-linear-nbhd-raw-softmax | 0.987 | 0.970 | 0.980 | 0.931 | 0.927 | 26.700 | 0.999 | 0.574 |  |  |  |
| L4-linear-token-raw-sigmoid | 0.989 | 0.898 | 0.978 | 1.000 | 1.000 | 14.500 | 0.992 | 0.658 |  |  |  |
| L4-mlp-token-ln-sigmoid | 0.997 | 0.882 | 0.978 | 0.998 | 0.998 | 19.500 | 0.992 | 0.656 |  |  |  |
| L4-linear-token-ln-sigmoid | 0.990 | 0.900 | 0.978 | 1.000 | 1.000 | 16.300 | 0.991 | 0.655 |  |  |  |
| last-linear-token-ln-softmax | 0.984 | 0.915 | 0.976 | 0.837 | 0.731 | 25.300 | 0.993 | 0.562 | 0.671 | 0.505 | 0.334 |
| last-linear-token-raw-softmax | 0.984 | 0.912 | 0.974 | 0.818 | 0.717 | 25.600 | 0.991 | 0.555 | 0.668 | 0.511 | 0.334 |
| L8-linear-token-ln-softmax | 0.982 | 0.900 | 0.967 | 0.962 | 0.986 | 27.300 | 0.998 | 0.616 |  |  |  |
| L4-mlp-token-ln-softmax | 0.991 | 0.857 | 0.966 | 0.998 | 0.998 | 19.900 | 0.998 | 0.693 |  |  |  |
| L8-linear-token-raw-softmax | 0.980 | 0.893 | 0.962 | 0.961 | 0.982 | 27.300 | 0.998 | 0.626 |  |  |  |
| L4-linear-token-raw-softmax | 0.940 | 0.840 | 0.934 | 0.979 | 0.996 | 24.000 | 0.999 | 0.623 |  |  |  |
| L4-linear-token-ln-softmax | 0.941 | 0.847 | 0.933 | 0.984 | 0.988 | 22.300 | 0.999 | 0.643 |  |  |  |
| last-linear-global-ln-softmax | 0.994 | 0.785 | 0.579 | 0.243 | 0.000 | 70.500 | 0.796 | 0.298 | 0.457 | 0.397 | 0.326 |
| last-linear-global-raw-softmax | 0.994 | 0.786 | 0.578 | 0.253 | 0.000 | 69.900 | 0.790 | 0.297 | 0.462 | 0.388 | 0.325 |
| last-linear-global-ln-sigmoid | 0.993 | 0.795 | 0.566 | 0.235 | 0.000 | 70.400 | 0.651 | 0.329 | 0.490 | 0.518 | 0.329 |
| L20-linear-global-raw-sigmoid | 0.992 | 0.757 | 0.564 | 0.087 | 0.000 | 105.200 | 0.697 | 0.300 |  |  |  |
| last-linear-global-raw-sigmoid | 0.993 | 0.797 | 0.564 | 0.239 | 0.000 | 70.100 | 0.642 | 0.326 | 0.489 | 0.519 | 0.329 |
| L20-linear-global-ln-sigmoid | 0.992 | 0.757 | 0.557 | 0.085 | 0.000 | 105.200 | 0.698 | 0.303 |  |  |  |
| last-mlp-global-raw-softmax | 0.995 | 0.748 | 0.531 | 0.170 | 0.000 | 79.300 | 0.524 | 0.266 | 0.390 | 0.231 | 0.330 |
| last-mlp-global-ln-softmax | 0.995 | 0.748 | 0.530 | 0.163 | 0.000 | 80.000 | 0.521 | 0.268 | 0.392 | 0.226 | 0.330 |
| L16-linear-global-ln-sigmoid | 0.989 | 0.659 | 0.524 | 0.023 | 0.000 | 161.600 | 0.609 | 0.301 |  |  |  |
| L16-linear-global-raw-sigmoid | 0.989 | 0.660 | 0.521 | 0.022 | 0.000 | 155.200 | 0.585 | 0.300 |  |  |  |
| L8-linear-global-raw-sigmoid | 0.984 | 0.582 | 0.516 | 0.046 | 0.000 | 155.900 | 0.399 | 0.337 |  |  |  |
| L8-linear-global-ln-softmax | 0.985 | 0.559 | 0.514 | 0.028 | 0.000 | 173.900 | 0.739 | 0.329 |  |  |  |
| L8-linear-global-raw-softmax | 0.985 | 0.563 | 0.514 | 0.029 | 0.000 | 173.100 | 0.708 | 0.327 |  |  |  |
| last-mlp-global-ln-sigmoid | 0.996 | 0.758 | 0.513 | 0.223 | 0.000 | 68.800 | 0.550 | 0.308 | 0.379 | 0.244 | 0.335 |
| last-mlp-global-raw-sigmoid | 0.996 | 0.757 | 0.513 | 0.231 | 0.000 | 68.400 | 0.565 | 0.305 | 0.379 | 0.243 | 0.336 |
| L8-linear-global-ln-sigmoid | 0.984 | 0.574 | 0.512 | 0.045 | 0.000 | 157.200 | 0.392 | 0.337 |  |  |  |
| L4-linear-global-ln-sigmoid | 0.973 | 0.579 | 0.491 | 0.072 | 0.000 | 162.100 | 0.501 | 0.316 |  |  |  |
| L4-linear-global-raw-sigmoid | 0.973 | 0.581 | 0.489 | 0.073 | 0.000 | 160.800 | 0.501 | 0.316 |  |  |  |
| L12-linear-global-raw-sigmoid | 0.988 | 0.544 | 0.488 | 0.031 | 0.000 | 146.900 | 0.493 | 0.310 |  |  |  |
| L12-linear-global-ln-sigmoid | 0.988 | 0.542 | 0.487 | 0.030 | 0.000 | 146.900 | 0.473 | 0.310 |  |  |  |
| L12-linear-global-raw-softmax | 0.988 | 0.513 | 0.478 | 0.030 | 0.000 | 154.600 | 0.734 | 0.318 |  |  |  |
| L12-linear-global-ln-softmax | 0.989 | 0.509 | 0.477 | 0.029 | 0.000 | 155.000 | 0.748 | 0.318 |  |  |  |
| L20-linear-global-ln-softmax | 0.994 | 0.729 | 0.449 | 0.102 | 0.000 | 105.000 | 0.680 | 0.291 |  |  |  |
| L20-linear-global-raw-softmax | 0.993 | 0.727 | 0.437 | 0.104 | 0.000 | 105.100 | 0.657 | 0.286 |  |  |  |
| L4-linear-global-ln-softmax | 0.970 | 0.522 | 0.397 | 0.061 | 0.000 | 204.800 | 0.392 | 0.330 |  |  |  |
| L4-linear-global-raw-softmax | 0.969 | 0.525 | 0.393 | 0.062 | 0.000 | 204.100 | 0.393 | 0.330 |  |  |  |
| L16-linear-global-ln-softmax | 0.991 | 0.618 | 0.337 | 0.023 | 0.000 | 159.800 | 0.666 | 0.298 |  |  |  |
| L16-linear-global-raw-softmax | 0.991 | 0.613 | 0.334 | 0.024 | 0.000 | 156.900 | 0.621 | 0.292 |  |  |  |

## Three-way outcome with the away-from-plank bounce readout

Most "hidden" passes stop at the plank edge still partly visible (only 26 of 77 vanish fully), so `returned()` (gone, then back on the near side) can't tell bounce from hidden even on ground truth. `away_from_plank` scores whether the visible ball moves away from the plank over the first 8 target steps; it fits nothing.

Ceiling (same readout on tracker ball cells): bounce-vs-hidden AUROC 0.9589, P(correct) 0.742, balanced accuracy 0.81.

| config | real: bounce-vs-hidden AUROC | real: P(correct) | real: balanced acc | imagined: bounce-vs-hidden AUROC | imagined: P(correct) | imagined: balanced acc |
|---|---|---|---|---|---|---|
| L16-mlp-token-ln-sigmoid | 0.837 | 0.722 | 0.808 |  |  |  |
| L16-mlp-token-ln-softmax | 0.870 | 0.722 | 0.809 |  |  |  |
| L12-mlp-token-ln-sigmoid | 0.833 | 0.720 | 0.809 |  |  |  |
| L8-mlp-token-ln-sigmoid | 0.828 | 0.720 | 0.812 |  |  |  |
| L20-mlp-token-ln-sigmoid | 0.835 | 0.719 | 0.799 |  |  |  |
| last-mlp-token-raw-sigmoid | 0.839 | 0.719 | 0.812 | 0.284 | 0.319 | 0.333 |
| last-mlp-token-ln-sigmoid | 0.838 | 0.718 | 0.812 | 0.296 | 0.320 | 0.333 |
| L20-mlp-token-ln-softmax | 0.871 | 0.715 | 0.799 |  |  |  |
| L12-mlp-token-ln-softmax | 0.850 | 0.714 | 0.814 |  |  |  |
| last-mlp-nbhd-ln-sigmoid | 0.813 | 0.713 | 0.755 | 0.460 | 0.332 | 0.338 |
| last-mlp-token-ln-softmax | 0.874 | 0.713 | 0.792 | 0.367 | 0.329 | 0.333 |
| last-mlp-token-raw-softmax | 0.871 | 0.713 | 0.788 | 0.359 | 0.329 | 0.333 |
| last-mlp-nbhd-raw-sigmoid | 0.810 | 0.712 | 0.755 | 0.452 | 0.332 | 0.338 |
| last-mlp-nbhd-raw-softmax | 0.904 | 0.709 | 0.749 | 0.435 | 0.300 | 0.328 |
| L16-mlp-nbhd-ln-sigmoid | 0.797 | 0.708 | 0.812 |  |  |  |
| last-mlp-nbhd-ln-softmax | 0.910 | 0.708 | 0.749 | 0.430 | 0.298 | 0.314 |
| L12-mlp-nbhd-ln-sigmoid | 0.805 | 0.707 | 0.752 |  |  |  |
| L8-mlp-token-ln-softmax | 0.843 | 0.707 | 0.783 |  |  |  |
| L4-mlp-nbhd-ln-sigmoid | 0.785 | 0.706 | 0.724 |  |  |  |
| L4-mlp-token-ln-sigmoid | 0.822 | 0.706 | 0.736 |  |  |  |
| L16-mlp-nbhd-ln-softmax | 0.863 | 0.705 | 0.805 |  |  |  |
| L20-mlp-nbhd-ln-softmax | 0.915 | 0.705 | 0.845 |  |  |  |
| L20-mlp-nbhd-ln-sigmoid | 0.797 | 0.704 | 0.810 |  |  |  |
| L8-mlp-nbhd-ln-sigmoid | 0.719 | 0.697 | 0.750 |  |  |  |
| L12-linear-token-ln-softmax | 0.813 | 0.695 | 0.804 |  |  |  |
| L12-linear-token-raw-softmax | 0.811 | 0.694 | 0.818 |  |  |  |
| L16-linear-token-raw-sigmoid | 0.795 | 0.693 | 0.739 |  |  |  |
| L16-linear-token-ln-sigmoid | 0.794 | 0.692 | 0.734 |  |  |  |
| L12-mlp-nbhd-ln-softmax | 0.867 | 0.691 | 0.759 |  |  |  |
| L4-linear-token-raw-sigmoid | 0.817 | 0.689 | 0.837 |  |  |  |
| L4-linear-token-ln-sigmoid | 0.813 | 0.687 | 0.834 |  |  |  |
| L12-linear-token-raw-sigmoid | 0.788 | 0.686 | 0.704 |  |  |  |
| L8-linear-token-raw-sigmoid | 0.791 | 0.686 | 0.710 |  |  |  |
| L12-linear-token-ln-sigmoid | 0.782 | 0.685 | 0.704 |  |  |  |
| L8-linear-token-ln-sigmoid | 0.784 | 0.684 | 0.710 |  |  |  |
| L16-linear-token-raw-softmax | 0.862 | 0.683 | 0.764 |  |  |  |
| L20-linear-token-raw-sigmoid | 0.778 | 0.680 | 0.674 |  |  |  |
| L20-linear-token-ln-sigmoid | 0.771 | 0.678 | 0.669 |  |  |  |
| L4-linear-nbhd-raw-sigmoid | 0.682 | 0.677 | 0.748 |  |  |  |
| L4-mlp-token-ln-softmax | 0.790 | 0.677 | 0.705 |  |  |  |
| L4-linear-nbhd-ln-sigmoid | 0.686 | 0.675 | 0.748 |  |  |  |
| L20-linear-token-raw-softmax | 0.841 | 0.673 | 0.704 |  |  |  |
| L8-mlp-nbhd-ln-softmax | 0.707 | 0.673 | 0.687 |  |  |  |
| L8-linear-nbhd-ln-sigmoid | 0.624 | 0.671 | 0.711 |  |  |  |
| L8-linear-nbhd-raw-sigmoid | 0.616 | 0.671 | 0.697 |  |  |  |
| last-linear-token-raw-sigmoid | 0.798 | 0.671 | 0.669 | 0.428 | 0.332 | 0.355 |
| L12-linear-nbhd-raw-sigmoid | 0.627 | 0.670 | 0.754 |  |  |  |
| last-linear-token-ln-sigmoid | 0.797 | 0.670 | 0.669 | 0.433 | 0.332 | 0.354 |
| L12-linear-nbhd-ln-sigmoid | 0.629 | 0.669 | 0.738 |  |  |  |
| L16-linear-nbhd-raw-sigmoid | 0.636 | 0.667 | 0.737 |  |  |  |
| L4-mlp-nbhd-ln-softmax | 0.638 | 0.667 | 0.672 |  |  |  |
| L16-linear-nbhd-ln-sigmoid | 0.637 | 0.666 | 0.728 |  |  |  |
| L16-linear-token-ln-softmax | 0.858 | 0.657 | 0.700 |  |  |  |
| L8-linear-token-raw-softmax | 0.784 | 0.655 | 0.795 |  |  |  |
| L20-linear-nbhd-raw-sigmoid | 0.597 | 0.653 | 0.709 |  |  |  |
| last-linear-nbhd-ln-sigmoid | 0.618 | 0.652 | 0.698 | 0.457 | 0.332 | 0.310 |
| last-linear-nbhd-raw-sigmoid | 0.615 | 0.652 | 0.694 | 0.458 | 0.332 | 0.295 |
| L12-linear-nbhd-ln-softmax | 0.792 | 0.650 | 0.752 |  |  |  |
| L12-linear-nbhd-raw-softmax | 0.791 | 0.650 | 0.744 |  |  |  |
| L8-linear-token-ln-softmax | 0.788 | 0.646 | 0.794 |  |  |  |
| L20-linear-nbhd-raw-softmax | 0.847 | 0.645 | 0.779 |  |  |  |
| L20-linear-token-ln-softmax | 0.835 | 0.645 | 0.687 |  |  |  |
| L20-linear-nbhd-ln-sigmoid | 0.593 | 0.644 | 0.695 |  |  |  |
| L16-linear-nbhd-raw-softmax | 0.821 | 0.639 | 0.745 |  |  |  |
| L16-linear-nbhd-ln-softmax | 0.800 | 0.623 | 0.716 |  |  |  |
| L20-linear-nbhd-ln-softmax | 0.838 | 0.620 | 0.725 |  |  |  |
| L4-linear-token-ln-softmax | 0.720 | 0.620 | 0.687 |  |  |  |
| L8-linear-nbhd-raw-softmax | 0.674 | 0.620 | 0.733 |  |  |  |
| L8-linear-nbhd-ln-softmax | 0.687 | 0.615 | 0.741 |  |  |  |
| last-linear-nbhd-ln-softmax | 0.714 | 0.608 | 0.693 | 0.450 | 0.332 | 0.335 |
| L4-linear-token-raw-softmax | 0.684 | 0.604 | 0.692 |  |  |  |
| L4-linear-nbhd-ln-softmax | 0.639 | 0.600 | 0.696 |  |  |  |
| L4-linear-nbhd-raw-softmax | 0.634 | 0.597 | 0.676 |  |  |  |
| last-linear-nbhd-raw-softmax | 0.700 | 0.595 | 0.684 | 0.441 | 0.332 | 0.338 |
| last-linear-token-ln-softmax | 0.708 | 0.577 | 0.646 | 0.464 | 0.331 | 0.328 |
| last-linear-token-raw-softmax | 0.692 | 0.565 | 0.636 | 0.448 | 0.331 | 0.323 |
| last-linear-global-ln-softmax | 0.596 | 0.375 | 0.357 | 0.465 | 0.323 | 0.323 |
| last-linear-global-raw-softmax | 0.598 | 0.375 | 0.357 | 0.455 | 0.321 | 0.323 |
| L8-linear-global-ln-softmax | 0.599 | 0.365 | 0.387 |  |  |  |
| L20-linear-global-ln-softmax | 0.618 | 0.364 | 0.348 |  |  |  |
| L20-linear-global-ln-sigmoid | 0.702 | 0.363 | 0.369 |  |  |  |
| L8-linear-global-raw-softmax | 0.608 | 0.363 | 0.388 |  |  |  |
| L20-linear-global-raw-sigmoid | 0.686 | 0.361 | 0.370 |  |  |  |
| L20-linear-global-raw-softmax | 0.613 | 0.360 | 0.334 |  |  |  |
| L4-linear-global-ln-sigmoid | 0.803 | 0.360 | 0.333 |  |  |  |
| L4-linear-global-raw-sigmoid | 0.808 | 0.360 | 0.333 |  |  |  |
| L4-linear-global-raw-softmax | 0.818 | 0.355 | 0.259 |  |  |  |
| last-linear-global-raw-sigmoid | 0.649 | 0.355 | 0.350 | 0.545 | 0.336 | 0.323 |
| L12-linear-global-ln-softmax | 0.682 | 0.354 | 0.359 |  |  |  |
| L16-linear-global-ln-sigmoid | 0.651 | 0.354 | 0.374 |  |  |  |
| L4-linear-global-ln-softmax | 0.814 | 0.354 | 0.249 |  |  |  |
| last-linear-global-ln-sigmoid | 0.648 | 0.354 | 0.355 | 0.541 | 0.335 | 0.315 |
| L12-linear-global-ln-sigmoid | 0.752 | 0.353 | 0.338 |  |  |  |
| L12-linear-global-raw-sigmoid | 0.756 | 0.353 | 0.338 |  |  |  |
| L12-linear-global-raw-softmax | 0.679 | 0.353 | 0.359 |  |  |  |
| L16-linear-global-raw-sigmoid | 0.632 | 0.351 | 0.374 |  |  |  |
| L16-linear-global-ln-softmax | 0.561 | 0.350 | 0.346 |  |  |  |
| last-mlp-global-ln-sigmoid | 0.589 | 0.350 | 0.376 | 0.570 | 0.344 | 0.333 |
| last-mlp-global-raw-sigmoid | 0.597 | 0.350 | 0.376 | 0.574 | 0.344 | 0.333 |
| L8-linear-global-ln-sigmoid | 0.648 | 0.349 | 0.342 |  |  |  |
| L8-linear-global-raw-sigmoid | 0.651 | 0.349 | 0.342 |  |  |  |
| L16-linear-global-raw-softmax | 0.547 | 0.344 | 0.340 |  |  |  |
| last-mlp-global-ln-softmax | 0.504 | 0.339 | 0.352 | 0.571 | 0.328 | 0.333 |
| last-mlp-global-raw-softmax | 0.503 | 0.339 | 0.352 | 0.570 | 0.328 | 0.333 |

## Other side: right-trained probes on left-segment clips

Fitted on all right clips' real context; the left clips (ball from the other side) are never seen.

| config | ctx cell AUROC | tgt cell AUROC | far cell AUROC | tgt hit | outcome AUROC | bounce-vs-hidden (away) | P(correct, away) | imag outcome AUROC |
|---|---|---|---|---|---|---|---|---|
| L8-linear-token-ln-sigmoid | 0.999 | 0.983 | 0.995 | 0.999 | 0.975 | 0.667 | 0.651 |  |
| L8-linear-token-ln-softmax | 0.964 | 0.943 | 0.965 | 0.927 | 1.000 | 0.771 | 0.538 |  |
| L8-linear-nbhd-ln-sigmoid | 1.000 | 0.998 | 1.000 | 0.997 | 1.000 | 0.528 | 0.627 |  |
| L8-linear-nbhd-ln-softmax | 0.988 | 0.978 | 0.994 | 0.949 | 1.000 | 0.681 | 0.515 |  |
| L8-mlp-token-ln-sigmoid | 0.999 | 0.984 | 0.998 | 0.994 | 0.993 | 0.847 | 0.702 |  |
| L8-mlp-token-ln-softmax | 0.998 | 0.974 | 0.994 | 0.976 | 0.995 | 0.861 | 0.603 |  |
| L8-mlp-nbhd-ln-sigmoid | 1.000 | 0.997 | 1.000 | 0.999 | 0.999 | 0.799 | 0.695 |  |
| L8-mlp-nbhd-ln-softmax | 0.999 | 0.996 | 0.999 | 0.994 | 0.999 | 0.757 | 0.605 |  |
| L12-linear-token-ln-sigmoid | 0.998 | 0.992 | 0.998 | 1.000 | 0.993 | 0.694 | 0.658 |  |
| L12-linear-token-ln-softmax | 0.895 | 0.967 | 0.977 | 0.950 | 0.990 | 0.764 | 0.504 |  |
| L12-linear-nbhd-ln-sigmoid | 1.000 | 0.998 | 1.000 | 0.994 | 1.000 | 0.597 | 0.639 |  |
| L12-linear-nbhd-ln-softmax | 0.962 | 0.988 | 0.995 | 0.959 | 0.986 | 0.785 | 0.527 |  |
| L12-mlp-token-ln-sigmoid | 1.000 | 0.994 | 0.999 | 0.999 | 0.995 | 0.757 | 0.704 |  |
| L12-mlp-token-ln-softmax | 0.999 | 0.988 | 0.996 | 0.996 | 0.988 | 0.826 | 0.656 |  |
| L12-mlp-nbhd-ln-sigmoid | 1.000 | 0.998 | 1.000 | 0.996 | 1.000 | 0.764 | 0.681 |  |
| L12-mlp-nbhd-ln-softmax | 0.999 | 0.994 | 0.999 | 0.989 | 1.000 | 0.819 | 0.613 |  |
| L16-linear-token-ln-sigmoid | 0.998 | 0.988 | 0.998 | 1.000 | 0.987 | 0.715 | 0.661 |  |
| L16-linear-token-ln-softmax | 0.849 | 0.963 | 0.974 | 0.893 | 0.974 | 0.757 | 0.430 |  |
| L16-linear-nbhd-ln-sigmoid | 1.000 | 0.998 | 0.999 | 0.993 | 1.000 | 0.569 | 0.643 |  |
| L16-linear-nbhd-ln-softmax | 0.817 | 0.975 | 0.984 | 0.828 | 0.988 | 0.729 | 0.468 |  |
| L16-mlp-token-ln-sigmoid | 1.000 | 0.994 | 0.999 | 1.000 | 0.995 | 0.535 | 0.671 |  |
| L16-mlp-token-ln-softmax | 0.999 | 0.988 | 0.997 | 0.984 | 1.000 | 0.792 | 0.677 |  |
| L16-mlp-nbhd-ln-sigmoid | 1.000 | 0.999 | 1.000 | 0.996 | 1.000 | 0.826 | 0.691 |  |
| L16-mlp-nbhd-ln-softmax | 0.999 | 0.996 | 0.999 | 0.996 | 0.990 | 0.861 | 0.612 |  |
| last-linear-token-ln-sigmoid | 0.999 | 0.981 | 0.995 | 0.997 | 0.993 | 0.743 | 0.667 | 0.490 |
| last-linear-token-ln-softmax | 0.939 | 0.966 | 0.977 | 0.865 | 0.961 | 0.660 | 0.415 | 0.546 |
| last-linear-nbhd-ln-sigmoid | 1.000 | 0.998 | 0.999 | 0.992 | 1.000 | 0.611 | 0.643 | 0.435 |
| last-linear-nbhd-ln-softmax | 0.927 | 0.977 | 0.984 | 0.852 | 0.999 | 0.757 | 0.437 | 0.625 |
| last-mlp-token-ln-sigmoid | 0.999 | 0.988 | 0.997 | 0.999 | 0.993 | 0.486 | 0.661 | 0.484 |
| last-mlp-token-ln-softmax | 0.998 | 0.983 | 0.996 | 0.983 | 0.998 | 0.819 | 0.670 | 0.561 |
| last-mlp-nbhd-ln-sigmoid | 1.000 | 0.998 | 0.999 | 0.987 | 1.000 | 0.792 | 0.702 | 0.438 |
| last-mlp-nbhd-ln-softmax | 0.998 | 0.993 | 0.999 | 0.989 | 1.000 | 0.819 | 0.641 | 0.423 |

## Blockade (label-free)

```json
{
 "blockade_range_y": [
  178.8,
  324.9
 ],
 "p_blocked_tgt": {
  "config": "L16-mlp-nbhd-ln-sigmoid",
  "auroc_blocked": 1.0,
  "mean_inside_range": 0.935,
  "mean_outside_range": 0.227
 },
 "p_blocked_img": {
  "config": "last-mlp-token-raw-softmax",
  "auroc_blocked": 0.5293,
  "mean_inside_range": 0.853,
  "mean_outside_range": 0.872
 },
 "distinctiveness": {
  "L4": {
   "plank_in_range": 0.0193,
   "plank_out_of_range": 0.0191,
   "plank_row_profile": [
    0.0247,
    0.026,
    0.0206,
    0.0177,
    0.019,
    0.0183,
    0.0227,
    0.0207,
    0.0206,
    0.0186,
    0.0157,
    0.0173,
    0.0161,
    0.0191,
    null,
    null
   ]
  },
  "L8": {
   "plank_in_range": 0.012,
   "plank_out_of_range": 0.0108,
   "plank_row_profile": [
    0.0108,
    0.0096,
    0.0118,
    0.0113,
    0.0121,
    0.0122,
    0.0132,
    0.0129,
    0.0112,
    0.0119,
    0.0111,
    0.0101,
    0.0111,
    0.01,
    null,
    null
   ]
  },
  "L12": {
   "plank_in_range": 0.0115,
   "plank_out_of_range": 0.0094,
   "plank_row_profile": [
    0.0096,
    0.0089,
    0.0079,
    0.0091,
    0.0096,
    0.0102,
    0.0128,
    0.0133,
    0.0118,
    0.0112,
    0.0104,
    0.0089,
    0.0099,
    0.0102,
    null,
    null
   ]
  },
  "L16": {
   "plank_in_range": 0.0084,
   "plank_out_of_range": 0.0065,
   "plank_row_profile": [
    0.01,
    0.0048,
    0.005,
    0.0058,
    0.0066,
    0.0071,
    0.0094,
    0.0121,
    0.0089,
    0.0075,
    0.0061,
    0.0065,
    0.0065,
    0.0075,
    null,
    null
   ]
  },
  "L20": {
   "plank_in_range": 0.0176,
   "plank_out_of_range": 0.0145,
   "plank_row_profile": [
    0.0217,
    0.01,
    0.0101,
    0.0128,
    0.0151,
    0.0149,
    0.0211,
    0.0235,
    0.018,
    0.0164,
    0.0133,
    0.0149,
    0.014,
    0.0171,
    null,
    null
   ]
  },
  "last": {
   "plank_in_range": 0.0824,
   "plank_out_of_range": 0.072,
   "plank_row_profile": [
    0.0873,
    0.0468,
    0.047,
    0.0681,
    0.0715,
    0.074,
    0.0871,
    0.1002,
    0.0877,
    0.0818,
    0.0671,
    0.0766,
    0.0763,
    0.0837,
    null,
    null
   ]
  }
 },
 "surprise_plank_row_hidden_minus_through": [
  -0.0069,
  0.0069,
  0.0068,
  0.0039,
  0.0012,
  0.0073,
  0.0121,
  0.0158,
  0.0126,
  0.0053,
  0.0077,
  0.0068,
  0.0091,
  0.0023,
  null,
  null
 ]
}
```

## Figures

![blocked_vs_crossing.png](blocked_vs_crossing.png)
![layers.png](layers.png)
![static_distinctiveness.png](static_distinctiveness.png)
![surprise_by_outcome.png](surprise_by_outcome.png)
![where_ball_last_seen.png](where_ball_last_seen.png)
