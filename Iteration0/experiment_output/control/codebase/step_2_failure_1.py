# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import hashlib
import numpy as np
import concurrent.futures
from scipy.special import ndtr

def compute_sjpv_prior(mL, mM, mH):
    S1 = np.log(mL) + np.log(mM) + np.log(mH)
    S2 = np.log(mL)**2 + np.log(mM)**2 + np.log(mH)**2
    V = S2 - (S1**2) / 3.0
    x_min = np.log(5e-4)
    x_max = np.log(0.3)
    y_min = np.log(5e-3)
    y_max = np.log(20.0)
    y = np.linspace(y_min, y_max, 500)
    dy = y[1] - y[0]
    S1 = S1[:, np.newaxis]
    V = V[:, np.newaxis]
    y = y[np.newaxis, :]
    ey = np.exp(y)
    e2y = np.exp(2*y)
    u_max = np.sqrt(3) * (x_max - S1/3.0) / ey
    u_min = np.sqrt(3) * (x_min - S1/3.0) / ey
    phi_diff = ndtr(u_max) - ndtr(u_min)
    log_term1 = -np.log(mL[:, np.newaxis]) - np.log(mM[:, np.newaxis]) - np.log(mH[:, np.newaxis]) - np.log(2 * np.pi * np.sqrt(3)) - 2*y
    log_term2 = -V / (2 * e2y)
    log_phi_diff = np.log(np.maximum(phi_diff, 1e-300))
    log_integrand = log_term1 + log_term2 + log_phi_diff
    integrand = np.exp(log_integrand)
    integral = np.trapz(integrand, dx=dy, axis=1)
    prior_vol = (x_max - x_min) * (y_max - y_min)
    return integral / prior_vol

def compute_hs_prior(mL, mM, mH, sigma_val):
    Z_HS = (1.5**6) / 60.0
    prior = (4.0 / Z_HS) * (mL*mM + mL*mH + mM*mH)
    prior[sigma_val > 1.5] = 0.0
    return prior

def compute_evidence(hierarchy, prior_type, mu_cosmo, sigma_cosmo, N=10000000, chunk_size=1000000):
    mL_max = 0.8
    total_integral = 0.0
    for i in range(0, N, chunk_size):
        current_chunk = min(chunk_size, N - i)
        if hierarchy == 'NH':
            dm2_21 = np.random.normal(7.49e-5, 0.19e-5, current_chunk)
            dm2_3l = np.random.normal(2.513e-3, 0.020e-3, current_chunk)
        else:
            dm2_21 = np.random.normal(7.49e-5, 0.19e-5, current_chunk)
            dm2_3l = np.random.normal(-2.484e-3, 0.020e-3, current_chunk)
        mL = np.random.uniform(1e-6, mL_max, current_chunk)
        if hierarchy == 'NH':
            m1 = mL
            m2 = np.sqrt(m1**2 + dm2_21)
            m3 = np.sqrt(m1**2 + dm2_3l)
            m_L, m_M, m_H = m1, m2, m3
        else:
            m3 = mL
            m2 = np.sqrt(m3**2 + np.abs(dm2_3l))
            m1 = np.sqrt(m3**2 + np.abs(dm2_3l) - dm2_21)
            m_L, m_M, m_H = m3, m1, m2
        valid = (m1 > 0) & (m2 > 0) & (m3 > 0)
        m_L = np.where(valid, m_L, 1e-6)
        m_M = np.where(valid, m_M, 1e-6)
        m_H = np.where(valid, m_H, 1e-6)
        sigma_val = m_L + m_M + m_H
        like_cosmo = np.exp(-0.5 * ((sigma_val - mu_cosmo) / sigma_cosmo)**2)
        like_cosmo[~valid] = 0.0
        jacobian = 1.0 / (4.0 * m_M * m_H)
        if prior_type == 'SJPV':
            prior = 6.0 * compute_sjpv_prior(m_L, m_M, m_H)
            integrand = like_cosmo * prior * jacobian
        elif prior_type == 'HS':
            prior = compute_hs_prior(m_L, m_M, m_H, sigma_val)
            integrand = like_cosmo * prior * jacobian
        elif prior_type == 'Flat':
            dSigma_dmL = 1.0 + m_L/m_M + m_L/m_H
            prior = np.ones_like(sigma_val) / 1.5
            prior[sigma_val > 1.5] = 0.0
            integrand = like_cosmo * prior * dSigma_dmL
        total_integral += np.sum(integrand)
    evidence = mL_max * (total_integral / N)
    return evidence

def task(args):
    ds_name, prior, hierarchy, mu, sigma = args
    seed_str = ds_name + "_" + prior + "_" + hierarchy
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    E = compute_evidence(hierarchy, prior, mu, sigma, N=10000000, chunk_size=1000000)
    return ds_name, prior, hierarchy, E

if __name__ == '__main__':
    datasets = {'baseline': {'mu': -0.009, 'sigma': 0.036}, 'plik': {'mu': -0.005, 'sigma': 0.037}, 'lh': {'mu': -0.012, 'sigma': 0.043}, 'w0wa': {'mu': 0.047, 'sigma': 0.065}, 'FC': {'mu': -0.036, 'sigma': 0.043}}
    priors = ['SJPV', 'HS', 'Flat']
    hierarchies = ['NH', 'IH']
    tasks = []
    for ds_name, ds_params in datasets.items():
        for prior in priors:
            for hierarchy in hierarchies:
                tasks.append((ds_name, prior, hierarchy, ds_params['mu'], ds_params['sigma']))
    results_dict = {}
    print("Starting Monte Carlo integration for Bayesian Evidence...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(task, tasks):
            ds_name, prior, hierarchy, E = res
            if ds_name not in results_dict:
                results_dict[ds_name] = {}
            if prior not in results_dict[ds_name]:
                results_dict[ds_name][prior] = {}
            results_dict[ds_name][prior][hierarchy] = E
    final_results = []
    print("\n" + "="*70)
    print("BAYESIAN EVIDENCE COMPUTATION RESULTS")
    print("="*70)
    for ds_name in datasets.keys():
        for prior in priors:
            E_NH_base = results_dict[ds_name][prior]['NH']
            E_IH_base = results_dict[ds_name][prior]['IH']
            K_base = E_NH_base / E_IH_base
            delta_chi2 = 6.1
            E_NH_full = E_NH_base
            E_IH_full = E_IH_base * np.exp(-delta_chi2 / 2.0)
            K_full = E_NH_full / E_IH_full
            print("Dataset: " + ds_name.ljust(10) + " | Prior: " + prior.ljust(4))
            print("  E_NH (base) = " + str(E_NH_base))
            print("  E_IH (base) = " + str(E_IH_base))
            print("  Bayes Factor K (cosmo+prior) = " + str(K_base))
            print("  Bayes Factor K (total w/ osc) = " + str(K_full))
            print("-" * 70)
            final_results.append({'dataset': ds_name, 'prior': prior, 'E_NH_base': E_NH_base, 'E_IH_base': E_IH_base, 'K_base': K_base, 'E_NH_full': E_NH_full, 'E_IH_full': E_IH_full, 'K_full': K_full})
    output_filepath = os.path.join("data", "evidence_results_step_2.json")
    with open(output_filepath, 'w') as f:
        json.dump(final_results, f, indent=4)
    print("\nResults successfully saved to " + output_filepath)