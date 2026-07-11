# Method capability evidence

`src/quant_agent/data/method_capabilities.yaml` stores facts used by the
deterministic compatibility engine. It is intentionally separate from the method
catalog: catalog fields describe what a method claims to do, while capability
fields describe where the implementation is documented or verified to work.

## Evidence policy

Use primary sources only:

1. the method paper;
2. the official implementation repository and documentation;
3. a versioned, reproducible run produced by this project.

Unknown values must remain empty or `null`. Do not convert absence of evidence into
an incompatibility. `supported_families` and `supported_gpu_arches` are positive
evidence by default. They become exhaustive only when the corresponding policy is
explicitly `allowlist` and the cited source clearly makes that claim.

Each derived fact uses:

- `field`: capability field supported by the fact;
- `value`: normalized value;
- `confidence`: `documented` or `inferred`;
- `source`: primary URL or repository-relative project report;
- `note`: scope and caveats.

## Support tiers

- `verified`: this project contains an end-to-end run for at least one model/GPU;
- `experimental`: execution or measurement plumbing exists but lacks a current
  public validation matrix;
- `catalog_only`: research can consider the method, but Adapt relies on repository
  discovery and there is no verified adapter claim.

The initial dataset deliberately has many unknowns. Paper/repository research should
enrich entries incrementally, with tests and source review, instead of filling all 35
methods from model memory in one pass.
