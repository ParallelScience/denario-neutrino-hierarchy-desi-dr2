# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from datetime import datetime
from scipy.special import ndtr
from scipy.integrate import trapezoid

def compute_sjpv_prior(mL, mM, mH):
    S1 = np.log(mL) + np.log(mM) + np.log(mH)
    S2 = np.log(mL)**2 + np.log(mM)**2 + np.log(mH)**2
    V = S2 - (S1**2) / 3.0
    x_min = np.log(5e-4)
    x_max = np.log(0.3)
    y_min = np.log(5e-3)
    y_max = np.log(20.0)
    y = np.linspace(y_min, y_max, 100)
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
    integral = trapezoid(integrand, dx=dy, axis=1)
    prior_vol = (x_max - x_min) * (y_max - y_min)
    return integral / prior_vol

def compute_hs_prior(mL, mM, mH, sigma_val):
    Z_HS = (1.5**6) / 60.0
    prior = (4.0 / Z_HS) * (mL*mM + mL*mH + mM*mH)
    prior[sigma_val > 1.5] = 0.0
    return prior

def compute_evidence_nufit(hierarchy, prior_type, mu_cosmo, sigma_cosmo, nufit_version='6.0', N=10000000, chunk_size=100000):
    if nufit_version == '6.0':
        dm2_21_mean, dm2_21_std = 7.49e-5, 0.19e-5
        dm2_3l_nh_mean, dm2_3l_nh_std = 2.513e-3, 0.020e-3
        dm2_3l_ih_mean, dm2_3l_ih_std = -2.484e-3, 0.020e-3
    else:
        dm2_21_mean, dm2_21_std = 7.42e-5, 0.21e-5
        dm2_3l_nh_mean, dm2_3l_nh_std = 2.510e-3, 0.027e-3
        dm2_3l_ih_mean, dm2_3l_ih_std = -2.490e-3, 0.027e-3
    mL_max = 0.8
    total_integral = 0.0
    for i in range(0, N, chunk_size):
        current_chunk = min(chunk_size, N - i)
        dm2_21 = np.random.normal(dm2_21_mean, dm2_21_std, current_chunk)
        if hierarchy == 'NH':
            dm2_3l = np.random.normal(dm2_3l_nh_mean, dm2_3l_nh_std, current_chunk)
        else:
            dm2_3l = np.random.normal(dm2_3l_ih_mean, dm2_3l_ih_std, current_chunk)
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
        total_integral += np.sum(integrand)
    evidence = mL_max * (total_integral / N)
    return evidence

def task(args):
    ds_name, prior, hierarchy, mu, sigma, nufit_version = args
    seed_str = ds_name + "_" + prior + "_" + hierarchy + "_" + nufit_version
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    E = compute_evidence_nufit(hierarchy, prior, mu, sigma, nufit_version=nufit_version, N=10000000, chunk_size=100000)
    return ds_name, prior, hierarchy, nufit_version, E

if __name__ == '__main__':
    step2_results_path = os.path.join('data', 'evidence_results_step_2.json')
    if os.path.exists(step2_results_path):
        with open(step2_results_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
    datasets_new = {'Pantheon+': {'mu': 0.0, 'sigma': 0.0704 / 1.96}, 'Union3': {'mu': 0.0, 'sigma': 0.0674 / 1.96}, 'DESY5': {'mu': 0.0, 'sigma': 0.0744 / 1.96}, 'DESY5_w0wa': {'mu': 0.0, 'sigma': 0.129 / 1.96}}
    priors = ['SJPV', 'HS']
    hierarchies = ['NH', 'IH']
    new_ds_names = list(datasets_new.keys()) + ['baseline_nufit5.1']
    all_results = [res for res in all_results if res['dataset'] not in new_ds_names]
    tasks_list = []
    for ds_name, ds_params in datasets_new.items():
        for prior in priors:
            for hierarchy in hierarchies:
                tasks_list.append((ds_name, prior, hierarchy, ds_params['mu'], ds_params['sigma'], '6.0'))
    for prior in priors:
        for hierarchy in hierarchies:
            tasks_list.append(('baseline_nufit5.1', prior, hierarchy, -0.009, 0.036, '5.1'))
    new_results_dict = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(task, tasks_list):
            ds_name, prior, hierarchy, nufit_version, E = res
            if ds_name not in new_results_dict:
                new_results_dict[ds_name] = {}
            if prior not in new_results_dict[ds_name]:
                new_results_dict[ds_name][prior] = {}
            new_results_dict[ds_name][prior][hierarchy] = E
    for ds_name in new_results_dict.keys():
        for prior in priors:
            E_NH_base = new_results_dict[ds_name][prior]['NH']
            E_IH_base = new_results_dict[ds_name][prior]['IH']
            K_base = E_NH_base / E_IH_base if E_IH_base > 0 else np.inf
            delta_chi2 = 7.0 if 'nufit5.1' in ds_name else 6.1
            E_NH_full = E_NH_base
            E_IH_full = E_IH_base * np.exp(-delta_chi2 / 2.0)
            K_full = E_NH_full / E_IH_full if E_IH_full > 0 else np.inf
            all_results.append({'dataset': ds_name, 'prior': prior, 'E_NH_base': E_NH_base, 'E_IH_base': E_IH_base, 'K_base': K_base, 'E_NH_full': E_NH_full, 'E_IH_full': E_IH_full, 'K_full': K_full})
    output_filepath = os.path.join('data', 'all_evidence_results_step_7.json')
    with open(output_filepath, 'w') as f:
        json.dump(all_results, f, indent=4)
    def get_evidence_level(k):
        if k > 100: return "Decisive"
        if k > 10: return "Strong"
        if k > 3.2: return "Substantial"
        return "Weak/None"
    display_order = ['baseline', 'plik', 'lh', 'Pantheon+', 'Union3', 'DESY5', 'w0wa', 'DESY5_w0wa', 'FC']
    plt.rcParams['text.usetex'] = False
    dataset_labels = ['Baseline\n(CamSpec)', 'plik CMB', 'L-H CMB', '+Pantheon+', '+Union3', '+DESY5', 'w0waCDM', 'w0waCDM\n+DESY5', 'Feldman-\nCousins']
    sjpv_k = []
    hs_k = []
    for ds in display_order:
        sjpv_val = next((item['K_full'] for item in all_results if item['dataset'] == ds and item['prior'] == 'SJPV'), 1.0)
        hs_val = next((item['K_full'] for item in all_results if item['dataset'] == ds and item['prior'] == 'HS'), 1.0)
        sjpv_k.append(sjpv_val)
        hs_k.append(hs_val)
    x = np.arange(len(display_order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width/2, sjpv_k, width, label='SJPV Prior', color='#1f77b4', edgecolor='black')
    ax.bar(x + width/2, hs_k, width, label='HS Prior', color='#d62728', edgecolor='black')
    ax.set_ylabel('Bayes Factor K (Total)', fontsize=14)
    ax.set_title('Bayesian Evidence for Normal Hierarchy Across Datasets (NuFIT 6.0)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, rotation=45, ha='right', fontsize=12)
    ax.set_yscale('log')
    ax.axhline(y=3.2, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y=10, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.7)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3, which='both')
    fig.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filepath = os.path.join('data', 'bayes_factors_summary_7_' + timestamp + '.png')
    fig.savefig(plot_filepath, dpi=300)