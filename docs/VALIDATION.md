# Validation

The test suite covers score-test calculations, variance-component fitting,
kernel construction, permutation calibration, AnnData conversion, spot-level
preprocessing, PVE calculation, plotting helpers, and end-to-end pipelines.

Run:

```bash
python -m pytest -q
python scripts/benchmark_reference.py
```

The numerical regression tests compare fixed-input outputs against independent
reference calculations. The benchmark fails if the maximum absolute numerical
difference exceeds `1e-12` or if the runtime ratio exceeds its guardrail.

The optional `mgcv` tests require the R packages installed by
`environment.yml`. SpotClean tests run only when that optional backend is
available.

The article settings are covered explicitly:

- reciprocal median coordinate-range gamma;
- half-median squared-distance gamma with deterministic row subsampling;
- interpolation-based permutation ECDF;
- deterministic random Fourier features;
- deterministic permutation streams;
- CLR composition transformation.

Public data analyses may differ across platforms because of preprocessing,
coordinate scaling, feature availability, and cell-type composition estimates.
For reproducibility, save the complete result metadata together with every
result table.
