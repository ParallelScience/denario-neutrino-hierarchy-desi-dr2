

Iteration 0:
### Project Summary: Bayesian Evidence for Neutrino Mass Hierarchy (DESI DR2)

**1. Methodology & Data**
*   **Objective:** Compute Bayes factors $K = P(D|NH)/P(D|IH)$ using DESI DR2 BAO+CMB chains and NuFIT 6.0 oscillation parameters.
*   **Likelihood:** Cosmological likelihood $P(D_{cosmo}|\Sigma)$ derived from DESI DR2 chains (base $\Lambda$CDM: $\Sigma m_\nu < 0.0642$ eV, 95% CL). Modeled as a truncated Gaussian ($\mu_0 = -0.009$ eV, $\sigma = 0.036$ eV).
*   **Priors:** 
    *   **SJPV:** Hierarchical log-normal (exchangeable masses, hyper-marginalized).
    *   **HS:** Objective Jeffreys-Fisher (Jacobian $J \propto \sum m_i m_j$).
*   **Oscillation Input:** NuFIT 6.0 ($\Delta\chi^2(IO-NO) = +6.1$).

**2. Key Findings**
*   **Decisive Evidence:** Under $\Lambda$CDM, $K_{full}$ (including $\Delta\chi^2$) is $\sim 10,231$ (SJPV) and $\sim 465$ (HS). Both exceed the "decisive" threshold ($K > 100$).
*   **Robustness:** The preference for NH is driven by the cosmological limit falling below the IH minimum mass ($\Sigma_{min}^{IH} \approx 0.099$ eV).
*   **Model Dependency:** In $w_0w_a$CDM, $K_{full}$ (HS) drops to $\sim 43$, shifting from "decisive" to "strong" evidence, indicating sensitivity to dark energy parameterization.
*   **$0\nu\beta\beta$ Implications:** NH posterior for $m_{\beta\beta}$ is suppressed (median $3.28$ meV, 95% UL $9.32$ meV), suggesting next-gen experiments (nEXO) may not detect a signal if NH is correct.

**3. Limitations & Uncertainties**
*   **Prior Sensitivity:** While both priors yield decisive results in $\Lambda$CDM, the magnitude of $K$ varies by an order of magnitude due to the geometric volume penalty imposed by the SJPV prior on the IH degenerate spectrum.
*   **Model Dependence:** The hierarchy preference is significantly relaxed when dark energy is allowed to vary ($w_0w_a$CDM), though it remains statistically significant.

**4. Future Directions**
*   **Extension:** Incorporate upcoming DESI Y5 data to further tighten $\Sigma m_\nu$ and test the stability of the $w_0w_a$CDM evidence.
*   **Refinement:** Investigate the impact of non-standard cosmological models (e.g., varying $N_{eff}$) on the hierarchy preference.
*   **Integration:** Use the derived $m_{\beta\beta}$ posteriors to provide formal Bayesian exclusion limits for specific $0\nu\beta\beta$ experimental designs.
        