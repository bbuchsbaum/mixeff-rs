# MMTrust-PSD LMM Prototype

This note defines the first direct positive-semidefinite covariance prototype
for Gaussian LMMs. It is intentionally smaller than the full MMTrust-PSD
research program: it proves that `mixeff-rs` can optimize over covariance
blocks `G >= 0` while continuing to evaluate the profiled objective through the
existing PLS / blocked Cholesky machinery.

## Scope

The first prototype is LMM-only and supports:

- scalar random-effect variance blocks `(1 | group)`;
- full 2x2 random intercept/slope blocks `(1 + x | group)`;
- ML and REML objectives through the existing profiled evaluator;
- objective parity against the current theta optimizer;
- rank-one active-face reduction for certified singular 2x2 blocks.

It explicitly does not support GLMMs, dense `V` or `P` construction,
matrix-free exact scores, or multi-block trust-region subproblems yet.

## Parameterization

The prototype's optimization variables are covariance entries, not Cholesky
entries.

Scalar block:

```text
g = [v], v >= 0
```

2x2 block:

```text
g = [g00, g01, g11]
G = [[g00, g01],
     [g01, g11]]
G >= 0
```

For objective evaluation only, the prototype converts `G` to the existing
lower Cholesky theta representation:

```text
G = L L'
theta = vech_lower(L)
```

This preserves the current model representation and avoids changing the PLS
evaluator. The optimizer-facing state is still covariance-space state.

## Feasibility

Scalar feasibility is `v >= 0`.

For 2x2 blocks, feasibility is checked by the cone condition:

```text
g00 >= 0
g11 >= 0
g00 * g11 - g01^2 >= -tol
```

Invalid covariance proposals are rejected by returning a large objective.
The prototype does not project invalid proposals back to the cone; that belongs
in the later trust-region implementation.

## Objective Evaluation

The prototype must never form dense marginal covariance matrices. Every trial
covariance block is converted to theta and evaluated by the existing LMM
objective path:

```text
G trial -> theta trial -> profiled PLS objective
```

This gives a clean parity check:

```text
F_psd(G_hat) ~= F_theta(theta_hat)
```

where `G_hat = Lambda(theta_hat) Lambda(theta_hat)'`.

## Acceptance Gates

The POC is acceptable when:

- scalar covariance-space optimization matches the current scalar theta
  optimizer objective within `1e-6 * (1 + |F_theta|)`;
- 2x2 covariance-space optimization matches the current vector theta optimizer
  objective within the same tolerance;
- a KKT-certified singular 2x2 block can be optimized on a rank-one active
  face, reducing the local covariance dimension from 3 to 1 while preserving
  the same objective tolerance;
- all POC evaluations use the existing PLS / blocked Cholesky machinery;
- no dense `V`, dense `P`, or response-space covariance matrix is introduced.

## Active-Face Reduction

For a certified rank-one 2x2 covariance block:

```text
G = U diag(lambda, 0) U'
```

the POC holds the active eigenvector `u` fixed and searches only over the
non-negative active variance coordinate:

```text
G(a) = a u u', a >= 0
```

This reduces the local covariance search from:

```text
[g00, g01, g11]  # 3 variables
```

to:

```text
[a]              # 1 variable on the certified active face
```

The prototype still evaluates every trial by converting `G(a)` back to theta
and calling the existing profiled objective. The face is used only for the
outer search geometry.

## Next Steps

After this POC, the real MMTrust-PSD work should replace the simple
pattern-search POC with a cone-aware trust-region subproblem, reuse the
covariance KKT certificates as termination diagnostics, and generalize
active-face reduction beyond the rank-one 2x2 case.
