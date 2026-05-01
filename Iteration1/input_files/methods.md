1. **Data Acquisition and Likelihood Construction**
   - Download DESI DR2 MCMC chains (base_mnu and base_mnu_w_wa). Use `getdist` to extract the 1D marginalized posterior for $\Sigma m_\nu$.
   - Construct the cosmological likelihood $P(D_{\text{cosmo}} | \Sigma)$ using a Kernel Density Estimate (KDE) and a truncated Gaussian fit ($\mu_0, \sigma$) constrained to $\Sigma m_\nu \geq 0$.
   - Perform a "prior-data conflict" assessment: calculate the Box p-value to quantify tension between the cosmological data and the physical boundary. Explicitly distinguish between the *prior-driven* evidence (volume excluded by $\Sigma < 0$) and the *likelihood-driven* evidence to isolate the data's contribution to the hierarchy preference.

2. **Refined Oscillation Likelihood and Marginalization**
   - Implement the oscillation likelihood $P(D_{\text{osc}} | m_1, m_2, m_3)$ using the full NuFIT 6.0 covariance matrix.
   - Perform full marginalization over the oscillation parameters ($\Delta m^2_{21}, \Delta m^2_{3\ell}$) to ensure the oscillation constraint is integrated into the evidence calculation without double-counting.

3. **Bayesian Evidence Computation**
   - Compute the evidence $P(D|M)$ for NH and IH using the SJPV hierarchical logarithmic prior and the HS objective prior (Jacobian $J(m_L, m_M, m_H)$).
   - Apply the $\Delta\chi^2$ contribution from the NuFIT 6.0 global fit as a multiplicative factor to the evidence ratio.
   - Calculate the Bayes factor $K = P(D|NH)/P(D|IH)$ and categorize the strength of evidence on the Jeffreys scale.

4. **Model-Hierarchy Degeneracy Analysis**
   - Quantify the correlation between $\Sigma m_\nu$ and $w_a$ in the DESI DR2 $w_0w_a$CDM chains.
   - Compare Bayes factors derived from $\Lambda$CDM versus $w_0w_a$CDM to assess the robustness of the hierarchy preference against dark energy parameterization.
   - Frame the results as a "Model-Hierarchy Degeneracy" to demonstrate the sensitivity of the hierarchy preference to the assumed cosmological model.

5. **Stress Testing the Hierarchy Preference**
   - Perform a sensitivity analysis by calculating $K$ as a function of the central value ($\mu_0$) of the cosmological constraint, keeping the width ($\sigma$) fixed to the DESI DR2 value.
   - Identify the "breaking point" $\mu_0$ at which $K$ drops below 10 for both SJPV and HS priors.
   - Generate a 2D contour plot of $K(\mu_0, \sigma)$ to provide a comprehensive "stability map" of the hierarchy preference.

6. **Philosophical and Geometric Justification**
   - Draft a formal section contrasting the Bayesian evidence approach with the frequentist Feldman-Cousins profile likelihood.
   - Explain how the Bayesian framework utilizes "Occam's Razor" by penalizing the IH for the fine-tuning required by its higher minimum mass threshold.
   - Plot the "Prior Volume" for both hierarchies to visualize the geometric effects of the SJPV and HS priors.

7. **Double-Beta Decay ($m_{\beta\beta}$) Conditional Predictions**
   - Compute the posterior for $m_{\beta\beta} = |\sum U_{ei}^2 m_i|$ by Monte Carlo integration over the PMNS matrix elements, sampling mixing angles from NuFIT 6.0 covariance matrices and Majorana phases uniformly in $[0, 2\pi]$.
   - Plot the 1D posterior of $m_{\beta\beta}$ for both hierarchies, overlaying sensitivity thresholds for nEXO and KamLAND-Zen.
   - Frame these results as "conditional predictions" based on the validity of the $\Lambda$CDM model.

8. **Historical Evolution and Final Reporting**
   - Re-run the evidence calculation for historical cosmological upper limits (2002–2025) to generate a timeline plot of $K$.
   - Compile a summary table of $K$ values across all datasets, priors, and dark energy models.
   - Document the impact of updating from NuFIT 5.1 to 6.0 and synthesize the final conclusion regarding the statistical preference for the Normal Hierarchy.