# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtr
from scipy.integrate import trapezoid
import concurrent.futures
from datetime import datetime

def compute_sjpv_prior_custom(mL, mM, mH, x_min_val, x_max_val, y_min_val, y_max_val):
    S1 = np.log(mL) + np.log(mM) + np.log(mH)
    S2 = np.log(mL)**2 + np.log(mM)**2 + np.log(mH)**2
    V = S2 - (S1**2) / 3.0
    x_min = np.log(x_min_val)
    x_max = np.log(x_max_val)
    y_min = np.log(y_min_val)
    y_max = np.log(y_max_val)
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

def compute_evidence_custom(hierarchy, prior_type, mu_cosmo, sigma_cosmo, x_min_val=5e-4, x_max_val=0.3, y_min_val=5e-3, y_max_val=20.0, N=5000000, chunk_size=100000):
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
            prior = 6.0 * compute_sjpv_prior_custom(m_L, m_M, m_H, x_min_val, x_max_val, y_min_val, y_max_val)
            integrand = like_cosmo * prior * jacobian
        elif prior_type == 'HS':
            prior = compute_hs_prior(m_L, m_M, m_H, sigma_val)
            integrand = like_cosmo * prior * jacobian
        elif prior_type == 'Flat':
            prior = np.ones_like(sigma_val) / 1.5
            prior[sigma_val > 1.5] = 0.0
            integrand = like_cosmo * prior
        total_integral += np.sum(integrand)
    return mL_max * (total_integral / N)

def get_prior_samples(prior_type, N=2000000, chunk_size=100000):
    mL_max = 0.8
    sigma_nh_list, weights_nh_list, sigma_ih_list, weights_ih_list = [], [], [], []
    for i in range(0, N, chunk_size):
        current_chunk = min(chunk_size, N - i)
        dm2_21_nh = np.random.normal(7.49e-5, 0.19e-5, current_chunk)
        dm2_3l_nh = np.random.normal(2.513e-3, 0.020e-3, current_chunk)
        mL_nh = np.random.uniform(1e-6, mL_max, current_chunk)
        m1_nh, m2_nh, m3_nh = mL_nh, np.sqrt(mL_nh**2 + dm2_21_nh), np.sqrt(mL_nh**2 + dm2_3l_nh)
        valid_nh = (m1_nh > 0) & (m2_nh > 0) & (m3_nh > 0)
        sigma_nh = m1_nh + m2_nh + m3_nh
        jacobian_nh = 1.0 / (4.0 * m2_nh * m3_nh)
        prior_nh = 6.0 * compute_sjpv_prior_custom(m1_nh, m2_nh, m3_nh, 5e-4, 0.3, 5e-3, 20.0) if prior_type == 'SJPV' else compute_hs_prior(m1_nh, m2_nh, m3_nh, sigma_nh)
        weights_nh = prior_nh * jacobian_nh
        weights_nh[~valid_nh] = 0.0
        sigma_nh_list.append(sigma_nh); weights_nh_list.append(weights_nh)
        dm2_21_ih = np.random.normal(7.49e-5, 0.19e-5, current_chunk)
        dm2_3l_ih = np.random.normal(-2.484e-3, 0.020e-3, current_chunk)
        mL_ih = np.random.uniform(1e-6, mL_max, current_chunk)
        m3_ih, m2_ih, m1_ih = mL_ih, np.sqrt(mL_ih**2 + np.abs(dm2_3l_ih)), np.sqrt(mL_ih**2 + np.abs(dm2_3l_ih) - dm2_21_ih)
        valid_ih = (m1_ih > 0) & (m2_ih > 0) & (m3_ih > 0)
        sigma_ih = m1_ih + m2_ih + m3_ih
        jacobian_ih = 1.0 / (4.0 * m1_ih * m2_ih)
        prior_ih = 6.0 * compute_sjpv_prior_custom(m3_ih, m1_ih, m2_ih, 5e-4, 0.3, 5e-3, 20.0) if prior_type == 'SJPV' else compute_hs_prior(m3_ih, m1_ih, m2_ih, sigma_ih)
        weights_ih = prior_ih * jacobian_ih
        weights_ih[~valid_ih] = 0.0
        sigma_ih_list.append(sigma_ih); weights_ih_list.append(weights_ih)
    return np.concatenate(sigma_nh_list), np.concatenate(weights_nh_list), np.concatenate(sigma_ih_list), np.concatenate(weights_ih_list), N, mL_max

def plot_prior_volumes(sigma_nh, w_nh, sigma_ih, w_ih, title, ax, N, mL_max=0.8):
    bins = np.linspace(0, 0.3, 200)
    bin_width = bins[1] - bins[0]
    density_weights_nh = (w_nh * mL_max / N) / bin_width
    density_weights_ih = (w_ih * mL_max / N) / bin_width
    hist_nh, _ = np.histogram(sigma_nh, bins=bins, weights=density_weights_nh)
    hist_ih, _ = np.histogram(sigma_ih, bins=bins, weights=density_weights_ih)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.plot(bin_centers, hist_nh, label='NH', color='blue', lw=2)
    ax.plot(bin_centers, hist_ih, label='IH', color='red', lw=2, linestyle='--')
    ax.set_xlabel('Sum of Neutrino Masses Sigma [eV]')
    ax.set_ylabel('Prior Probability Density P(Sigma | M)')
    ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0, 0.3)

def sensitivity_task(args):
    name, prior_type, hierarchy, x_min, x_max, y_min, y_max = args
    np.random.seed(int(hashlib.md5((name + prior_type + hierarchy).encode()).hexdigest(), 16) % (2**32))
    E = compute_evidence_custom(hierarchy, prior_type, -0.009, 0.036, x_min, x_max, y_min, y_max, N=5000000)
    return name, prior_type, hierarchy, E

if __name__ == '__main__':
    sigma_nh_sjpv, w_nh_sjpv, sigma_ih_sjpv, w_ih_sjpv, N_sjpv, mL_max_sjpv = get_prior_samples('SJPV', N=2000000)
    sigma_nh_hs, w_nh_hs, sigma_ih_hs, w_ih_hs, N_hs, mL_max_hs = get_prior_samples('HS', N=2000000)
    variations = [('Baseline SJPV', 'SJPV', 5e-4, 0.3, 5e-3, 20.0), ('Expanded mu', 'SJPV', 1e-4, 1.0, 5e-3, 20.0), ('Expanded sigma', 'SJPV', 5e-4, 0.3, 1e-3, 50.0), ('Both Expanded', 'SJPV', 1e-4, 1.0, 1e-3, 50.0), ('Flat Prior', 'Flat', 5e-4, 0.3, 5e-3, 20.0)]
    tasks = [(name, ptype, 'NH', xmin, xmax, ymin, ymax) for name, ptype, xmin, xmax, ymin, ymax in variations] + [(name, ptype, 'IH', xmin, xmax, ymin, ymax) for name, ptype, xmin, xmax, ymin, ymax in variations]
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for name, ptype, hierarchy, E in executor.map(sensitivity_task, tasks):
            if name not in results: results[name] = {}
            results[name][hierarchy] = E
    osc_factor = np.exp(6.1 / 2.0)
    sensitivity_data = [{'name': name, 'K_base': results[name]['NH'] / results[name]['IH'], 'K_full': (results[name]['NH'] / results[name]['IH']) * osc_factor} for name, _, _, _, _, _ in variations]
    with open(os.path.join('data', 'sensitivity_results_step_4.json'), 'w') as f: json.dump(sensitivity_data, f, indent=4)
    fig = plt.figure(figsize=(18, 6))
    plot_prior_volumes(sigma_nh_sjpv, w_nh_sjpv, sigma_ih_sjpv, w_ih_sjpv, 'SJPV Prior Volume', fig.add_subplot(131), N_sjpv, mL_max_sjpv)
    plot_prior_volumes(sigma_nh_hs, w_nh_hs, sigma_ih_hs, w_ih_hs, 'HS Prior Volume', fig.add_subplot(132), N_hs, mL_max_hs)
    ax3 = fig.add_subplot(133)
    ax3.bar(np.arange(len(sensitivity_data)), [d['K_full'] for d in sensitivity_data], color='skyblue', edgecolor='black')
    ax3.set_xticks(np.arange(len(sensitivity_data))); ax3.set_xticklabels([d['name'] for d in sensitivity_data], rotation=45, ha='right')
    ax3.set_yscale('log'); ax3.axhline(y=100, color='gray', linestyle=':', label='Decisive (K>100)'); ax3.legend()
    plt.tight_layout(); plt.savefig(os.path.join('data', 'prior_volume_sensitivity_4_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'), dpi=300)