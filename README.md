# denario-neutrino-hierarchy-desi-dr2

**Scientist:** denario-6
**Date:** 2026-05-01


# Neutrino Mass Hierarchy: Bayesian Evidence with DESI DR2 Constraints

## Project Goal

Apply the Bayesian evidence methodology of Jimenez, Pena-Garay, Short, Simpson & Verde (2022) [arXiv:2203.14247, JCAP] to the new, tightest-ever cosmological neutrino mass constraints from DESI DR2 (Elbers et al. 2025, PRD 112, 083513). The goal is to compute Bayes factors for Normal Hierarchy (NH) vs Inverted Hierarchy (IH), using both the SJPV (hierarchical logarithmic) and HS (objective Bayesian / Jeffreys) prior frameworks. The paper must: (1) highlight the foreseeing power of the 2022 methodology in predicting NH; (2) include a rigorous mathematical/philosophical section on prior choice in Bayesian model comparison; (3) include many detailed plots.

---

## Data Sources

### 1. DESI DR2 MCMC Chains (PRIMARY DATA — download at runtime)

The DESI DR2 BAO cosmology chains are publicly available at:
  https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/cobaya/

The primary chain for Σmν (free neutrino mass, ΛCDM) is at:
  https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/cobaya/base_mnu/desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck-NPIPE-highl-CamSpec-TTTEEE_planck-act-dr6-lensing/

Files: chain.1.txt, chain.2.txt, chain.3.txt, chain.4.txt
Format: cobaya MCMC chains, ASCII. Column headers in chain.1.txt.
Read with GetDist: `from getdist.mcsamples import loadMCSamples; s = loadMCSamples(root)` where root is the path without the chain number suffix.

Additional chains to download for comparison (same base URL structure):
- base_mnu/desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck2018-highl-plik-TTTEEE_planck-act-dr6-lensing/ (plik CMB likelihood)
- base_mnu/desi-bao-all_planck2018-lowl-TT-clik_planck2020-lollipop-lowlE_planck2020-hillipop-TTTEEE_planck-act-dr6-lensing/ (L-H CMB likelihood)
- base_mnu_w_wa/desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck-NPIPE-highl-CamSpec-TTTEEE_planck-act-dr6-lensing/ (w0wa dark energy model)

The chain column of interest is `mnu` (= Σmν in eV). Use GetDist to extract the 1D marginalized posterior P(Σmν | data).

### 2. Published Numerical Results (to be used directly in code)

From Elbers et al. 2025 (DESI DR2):

| Dataset | Σmν upper limit (95%) | σ(Σmν) eV | Notes |
|---|---|---|---|
| DESI DR2 BAO + CMB (CamSpec, baseline) | 0.0642 eV | 0.020 | PRIMARY |
| DESI DR2 BAO + CMB (plik) | 0.0691 eV | — | Alternative CMB |
| DESI DR2 BAO + CMB (L-H) | 0.0774 eV | — | Alternative CMB |
| DESI DR2 BAO + CMB + Pantheon+ | 0.0704 eV | — | +SNIa |
| DESI DR2 BAO + CMB + Union3 | 0.0674 eV | — | +SNIa |
| DESI DR2 BAO + CMB + DESY5 | 0.0744 eV | — | +SNIa |
| DESI DR2 BAO + CMB (w0waCDM) | 0.163 eV | — | Dark energy extension |
| DESI DR2 BAO + CMB + DESY5 (w0waCDM) | 0.129 eV | — | Dark energy extension |
| Feldman-Cousins (ΛCDM) | 0.053 eV | — | Physical boundary corrected |

Profile likelihood (Gaussian model for cosmological constraint):
- ΛCDM baseline: μ₀ = −0.036 eV, σ = 0.043 eV (from Feldman-Cousins analysis in DESI paper)
- When used as likelihood for Bayesian evidence: truncate at Σmν = 0 (non-negativity)

### 3. Historical Cosmological Constraints (for comparison/evolution plots)

All at 95% CL:
| Year | Limit (eV) | Source |
|---|---|---|
| 2002 | 1.80 | 2dFGRS |
| 2012 | 0.44 | WMAP-9 |
| 2013 | 0.25 | Planck 2013 |
| 2015 | 0.18 | Planck 2015 |
| 2016 | 0.13 | Planck+BAO galaxy power spectrum |
| 2021a | 0.102 | Planck+BAO+RSD (eBOSS) |
| 2021b | 0.099 | Planck+BAO+RSD+SNe |
| 2021c | 0.089 | Planck+BAO+Ly-α |
| 2022a | 0.071 | DESI DR1 BAO + CMB (plik) |
| 2025 | 0.0642 | DESI DR2 BAO + CMB (baseline) |

### 4. Oscillation Parameters — NuFIT 6.0 (Esteban et al. 2024, arXiv:2410.05380)

Mass-squared splittings (68% CL, including SK atmospheric data):
- Δm²₂₁ = 7.49 ± 0.19 × 10⁻⁵ eV² (solar splitting)
- Δm²₃₁ = +2.513 ± 0.020 × 10⁻³ eV² (Normal Ordering)
- Δm²₃₂ = −2.484 ± 0.020 × 10⁻³ eV² (Inverted Ordering)

Δχ² between orderings (NuFIT 6.0):
- With SK atmospheric: Δχ²(IO - NO) = +6.1 (normal ordering preferred)
- Without SK atmospheric: Δχ²(IO - NO) ≈ +2.5 (normal ordering weakly preferred)

Minimum sum of masses (from oscillations alone):
- NH minimum: Σmν^(NH,min) = 0.05878 ± 0.00023 eV (degenerate case m₁→0)
- IH minimum: Σmν^(IH,min) = 0.09892 ± 0.00041 eV (degenerate case m₃→0)

For comparison, NuFIT 5.1 (used in 2022 paper):
- Δm²₂₁ = 7.42 ± 0.21 × 10⁻⁵ eV²
- Δm²₃₁ = +2.510 ± 0.027 × 10⁻³ eV² (NH)
- Δm²₃₂ = −2.490 ± 0.027 × 10⁻³ eV² (IH)
- With SK-atm: Δχ²(IO - NO) = 7.0
- Without SK-atm: Δχ²(IO - NO) = 2.6

---

## Methodology (from Jimenez et al. 2022, arXiv:2203.14247)

### Physical Model

Three neutrino mass eigenstates (m₁, m₂, m₃). Two orderings:
- NH: m₁ ≤ m₂ ≤ m₃ → lightest is m₁, intermediate m₂, heaviest m₃
- IH: m₃ ≤ m₁ ≤ m₂ → lightest is m₃, intermediate m₁, heaviest m₂

Parameterize as (mL, mM, mH) = (lightest, middle, heaviest). Then:

For NH: m₂ = sqrt(m₁² + Δm²₂₁), m₃ = sqrt(m₁² + Δm²₃₁)
For IH: m₁ = sqrt(m₃² + |Δm²₃₂| - Δm²₂₁), m₂ = sqrt(m₃² + |Δm²₃₂|)
Sum: Σ = mL + mM + mH

The cosmological likelihood depends only on Σ.
The oscillation likelihood depends on Δm²₂₁ and Δm²₃ℓ.

### Bayesian Evidence

For each ordering M ∈ {NH, IH}:

P(D | M) = ∫ P(D_cosmo | Σ) P(D_osc | m₁, m₂, m₃) P(m₁, m₂, m₃ | M) dm₁ dm₂ dm₃

Bayes factor: K = P(D | NH) / P(D | IH)

The evidence ratio K is the primary output. On the Jeffreys scale:
- K > 3.2: "substantial" evidence
- K > 10: "strong" evidence
- K > 100: "decisive" evidence

### Prior 1: SJPV (Hierarchical Logarithmic — Jimenez et al.)

The prior on each individual mass before oscillation data is log-normal:
  P(log mᵢ | μ, σ) ~ N(log μ, σ)

with hyperpriors (uniform in log space):
  π(log μ) ~ U(log(5×10⁻⁴ eV), log(0.3 eV))
  π(log σ) ~ U(log(5×10⁻³), log(20))

The three masses are exchangeable (drawn from the same common distribution), reflecting the belief that all three masses share a common generation mechanism. The total prior marginalizes over μ and σ.

The evidence integral becomes 5-dimensional: (m₁ or mL, μ, σ) plus constraints from oscillations.

Implementation: numerical integration or MCMC over (mL, μ, σ) with the oscillation data folded in.

### Prior 2: HS (Objective Bayesian / Jeffreys-Fisher — Simpson et al.)

Reference prior derived from Fisher information of the oscillation likelihood:
  P(mL, mM, mH) ∝ J(mL, mM, mH) = 4(mL·mM + mL·mH + mM·mH)

This is the Jacobian of the transformation from natural oscillation parameters (φ=Δm²₂₁, ψ=Δm²₃ℓ, Σ) to individual masses.

Upper truncation at Σmax = 1.5 eV (improper prior regularization).

### Cosmological Likelihood Model

From the DESI DR2 chain, extract the marginalized 1D posterior P(Σmν | DESI+CMB). For the Bayesian evidence calculation, treat this as the likelihood: P(D_cosmo | Σ).

Two representations:
1. Direct: use kernel density estimate (KDE) of chain samples for P(Σ | data)
2. Gaussian approximation: P(Σ | data) ∝ exp(−(Σ − μ₀)²/(2σ²)) × θ(Σ) where θ is Heaviside

For the Gaussian approximation, fit to the chain posterior: find μ₀ and σ such that 95th percentile matches published limit. Also use the profile likelihood values (μ₀ = −0.036 eV, σ = 0.043 eV) for Feldman-Cousins comparison.

### Double-Beta Decay Implications (Majorana Neutrinos)

Effective Majorana mass:
  mββ = |Σᵢ Uₑᵢ² mᵢ| where U is PMNS matrix

For the theoretical upper limit of the half-life of neutrinoless double-beta decay:
  T^(0νββ)_{1/2} ∝ 1/mββ²

Using the posterior on (m₁, m₂, m₃) from the NH analysis, compute the posterior on mββ and compare to IH case. Show the allowed ranges of mββ for NH vs IH given DESI DR2 + oscillation constraints. PMNS parameters from NuFIT 6.0.

---

## Required Analysis Steps

The engineer must implement the following:

### Step 1: Download and Process DESI DR2 Chains
- Download chain.1.txt through chain.4.txt for the primary dataset (ΛCDM base_mnu, DESI BAO + CMB CamSpec)
- Use GetDist (pip: getdist) to read chains and extract Σmν posterior
- Fit Gaussian (possibly truncated at 0) to the posterior: find μ₀, σ
- Compare KDE vs Gaussian: plot both
- Save processed posterior to disk

### Step 2: Implement Bayesian Evidence Computation
- For both NH and IH:
  - SJPV prior: Monte Carlo integration over (mL, μ, σ) with oscillation + cosmological likelihood
  - HS prior: Monte Carlo integration over mL with Jacobian prior + oscillation + cosmological likelihood
- Compute evidence for each ordering and each prior
- Compute Bayes factors K = E(NH)/E(IH)
- Include and exclude the Δχ² contribution from oscillation global fit separately

### Step 3: Historical Evolution of Bayes Factors
- Compute K(Σ_limit) as a function of the historical cosmological upper limits (2002–2025)
- Show the "prediction" power: as constraints tightened, K grew, and the 2022 paper's framework was already conclusive
- Make a timeline plot showing how K evolved from "no evidence" to "decisive"

### Step 4: Prior Sensitivity Analysis
- Vary the SJPV hyperprior boundaries and show K is stable
- Compare SJPV, HS, and (as a check) a simple flat prior
- Show prior volume plots to illustrate geometric argument

### Step 5: Individual Mass Posteriors
- For NH and IH: compute posteriors on m₁, m₂, m₃ (or mL, mM, mH)
- Triangle plot (corner plot) of (mL, mM, mH, Σ) for NH and IH
- Show mass spectra (m₁, m₂, m₃ as function of mL) for NH and IH

### Step 6: Double-Beta Decay
- Compute posterior on mββ for NH and IH given DESI DR2 + oscillations
- Plot 1D posterior of mββ for both hierarchies
- Compare to sensitivity of planned experiments (nEXO: ~5 meV, KamLAND-Zen: ~20 meV)
- Show upper limit on T^(0νββ)_{1/2} implied by NH + DESI DR2

### Step 7: Dark Energy Dependence
- Repeat evidence computation using the w0waCDM limit (Σmν < 0.163 eV): how does K change?
- Show K vs assumed dark energy model

### Step 8: Summary Statistics and Tables
- Table of K values for all dataset combinations and both priors
- Compare 2022 results (NuFIT 5.1 + older cosmology) vs 2025 results (NuFIT 6.0 + DESI DR2)

---

## Software Requirements

- Python packages: numpy, scipy, matplotlib, getdist, corner, astropy
- All packages available in /opt/denario-venv
- Download chains at runtime using: requests or urllib
- GPU not needed; pure CPU computation
- Multiprocessing allowed: use up to 8 workers
- Memory: chains are ~50-100 MB each; 4 chains × ~50 MB = manageable

---

## Notes for the Planner

1. This is a COMPUTATION project, not a machine-learning project. The core task is Bayesian integration.
2. The DESI chains must be downloaded at the start of the first step.
3. The evidence integrals can be done with numpy/scipy (Gaussian quadrature or Monte Carlo). MCMC is not needed.
4. GetDist is key for reading cobaya chains — it handles the weight column automatically.
5. The Δχ² contribution enters multiplicatively: K_total = K_prior × exp(Δχ²_IO-NO / 2). Report separately and combined.
6. Be precise about NuFIT version: use NuFIT 6.0 values (arXiv:2410.05380) as primary, NuFIT 5.1 (used in 2022 paper) for comparison.
7. For corner plots use the `corner` package; for 1D posteriors use matplotlib.
8. Always label which prior (SJPV or HS) and which dataset is being used.
9. This paper targets JCAP — use professional scientific style.
