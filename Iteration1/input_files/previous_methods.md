1. **Data Acquisition and Preprocessing**:
    - Download DESI DR2 MCMC chains (base_mnu and base_mnu_w_wa) with a check for file existence to prevent redundant downloads.
    - Use `getdist` to load chains and extract the 1D marginalized posterior for $\Sigma m_\nu$.
    - Construct the cosmological likelihood $P(D_{\text{cosmo}} | \Sigma)$ using two methods: (a) a Kernel Density Estimate (KDE) of the chains, and (b) a truncated Gaussian fit ($\mu_0, \sigma$) constrained to $\Sigma m_\nu \geq 0$.
    - Benchmark these against the Feldman-Cousins profile likelihood ($\mu_0 = -0.036$ eV, $\sigma = 0.043$ eV) to ensure consistency with the DESI DR2 statistical treatment.

2. **Bayesian Evidence Framework Implementation**:
    - Define the hierarchy-specific likelihoods by combining $P(D_{\text{cosmo}} | \Sigma)$ with the oscillation likelihood $P(D_{\text{osc}} | m_1, m_2, m_3)$.
    - Construct $P(D_{\text{osc}} | \theta)$ using NuFIT 6.0 $\chi^2$ values, modeled as a product of Gaussians for mass-squared splittings and mixing angles, ensuring proper normalization such that $\int P(D_{\text{osc}} | \theta) d\theta = 1$.
    - Implement the SJPV hierarchical logarithmic prior (marginalizing over hyper-parameters $\mu, \sigma$) and the HS objective prior (using the Jacobian $J(m_L, m_M, m_H)$).
    - Use numerical integration to compute the evidence $P(D|M)$ for each hierarchy.

3. **Bayes Factor Calculation and Sensitivity**:
    - Compute the Bayes factor $K = P(D|NH)/P(D|IH)$ for both priors.
    - Separate the contribution from the oscillation data ($\Delta\chi^2$ between orderings) from the cosmological evidence.
    - Include a "Flat Prior" (uniform in $\Sigma m_\nu$) as a control case to ensure robustness.
    - Perform sensitivity checks by varying SJPV hyperprior boundaries and comparing results against the FC-corrected profile likelihood.

4. **Prior Volume and Philosophical Analysis**:
    - Calculate and plot the "Prior Volume" for both hierarchies to visualize the geometric effects of the SJPV and HS frameworks.
    - Include a formal section contrasting the geometric volume effects of the SJPV and HS priors to explain differences in Bayes factors, addressing the project's requirement for a rigorous mathematical/philosophical discussion.

5. **Historical Evolution Analysis**:
    - Compile historical cosmological upper limits (2002–2025) and re-run the evidence calculation for each.
    - Generate a timeline plot showing the evolution of $K$ to demonstrate the predictive power of the methodology and the transition from "no evidence" to "decisive" evidence on the Jeffreys scale.

6. **Mass Posterior and Spectral Analysis**:
    - Generate 1D and 2D posterior distributions for individual mass eigenstates ($m_1, m_2, m_3$) and $\Sigma$ using the `corner` package.
    - Construct mass spectrum plots showing the relationship between individual masses as a function of $m_L$ for both hierarchies.

7. **Double-Beta Decay ($m_{\beta\beta}$) Projections**:
    - Compute the posterior for $m_{\beta\beta} = |\sum U_{ei}^2 m_i|$ by performing Monte Carlo integration over the PMNS matrix elements.
    - Sample mixing angles using NuFIT 6.0 covariance matrices and Majorana phases ($\alpha_1, \alpha_2$) uniformly in $[0, 2\pi]$.
    - Plot the 1D posterior of $m_{\beta\beta}$ for both hierarchies, overlaying sensitivity thresholds for nEXO and KamLAND-Zen.

8. **Dark Energy Dependency and Final Reporting**:
    - Repeat evidence computation using $w_0w_a$CDM chains to quantify the degradation of $K$ relative to the $\Lambda$CDM baseline.
    - Compile a summary table of $K$ values across all datasets and priors.
    - Explicitly state the preference in terms of "decisive evidence for NH" vs "decisive evidence against IH" and document the impact of updating from NuFIT 5.1 to 6.0.