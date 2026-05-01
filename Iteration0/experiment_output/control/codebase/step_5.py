# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import corner
import matplotlib.pyplot as plt
from datetime import datetime
import concurrent.futures
from scipy.special import ndtr
from scipy.integrate import trapezoid
plt.rcParams['text.usetex'] = False
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
def generate_chunk(args):
    hierarchy, mu_cosmo, sigma_cosmo, N_chunk, seed = args
    np.random.seed(seed)
    mL_max = 0.3
    dm2_21 = np.random.normal(7.49e-5, 0.19e-5, N_chunk)
    if hierarchy == 'NH':
        dm2_3l = np.random.normal(2.513e-3, 0.020e-3, N_chunk)
        mL = np.random.uniform(1e-6, mL_max, N_chunk)
        m1 = mL
        m2 = np.sqrt(m1**2 + dm2_21)
        m3 = np.sqrt(m1**2 + dm2_3l)
        m_L, m_M, m_H = m1, m2, m3
        jacobian = 1.0 / (4.0 * m2 * m3)
    else:
        dm2_3l = np.random.normal(-2.484e-3, 0.020e-3, N_chunk)
        mL = np.random.uniform(1e-6, mL_max, N_chunk)
        m3 = mL
        m2 = np.sqrt(m3**2 + np.abs(dm2_3l))
        m1 = np.sqrt(m3**2 + np.abs(dm2_3l) - dm2_21)
        m_L, m_M, m_H = m3, m1, m2
        jacobian = 1.0 / (4.0 * m1 * m2)
    valid = (m1 > 0) & (m2 > 0) & (m3 > 0)
    m_L = np.where(valid, m_L, 1e-6)
    m_M = np.where(valid, m_M, 1e-6)
    m_H = np.where(valid, m_H, 1e-6)
    sigma_val = m_L + m_M + m_H
    like_cosmo = np.exp(-0.5 * ((sigma_val - mu_cosmo) / sigma_cosmo)**2)
    like_cosmo[~valid] = 0.0
    prior = np.zeros(N_chunk)
    sub_chunk_size = 100000
    for i in range(0, N_chunk, sub_chunk_size):
        end = min(i + sub_chunk_size, N_chunk)
        prior[i:end] = 6.0 * compute_sjpv_prior_custom(m_L[i:end], m_M[i:end], m_H[i:end], 5e-4, 0.3, 5e-3, 20.0)
    weights = like_cosmo * prior * jacobian
    weights[~valid] = 0.0
    weights[np.isnan(weights)] = 0.0
    weights[np.isinf(weights)] = 0.0
    return m1, m2, m3, sigma_val, weights
def generate_posterior_samples_mp(hierarchy, mu_cosmo, sigma_cosmo, N_total=10000000, num_workers=8):
    chunk_size = N_total // num_workers
    tasks = [(hierarchy, mu_cosmo, sigma_cosmo, chunk_size, np.random.randint(0, 2**32)) for _ in range(num_workers)]
    m1_list, m2_list, m3_list, sigma_list, weights_list = [], [], [], [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in executor.map(generate_chunk, tasks):
            m1, m2, m3, sigma_val, weights = res
            m1_list.append(m1)
            m2_list.append(m2)
            m3_list.append(m3)
            sigma_list.append(sigma_val)
            weights_list.append(weights)
    m1_all = np.concatenate(m1_list)
    m2_all = np.concatenate(m2_list)
    m3_all = np.concatenate(m3_list)
    sigma_all = np.concatenate(sigma_list)
    weights_all = np.concatenate(weights_list)
    weights_sum = np.sum(weights_all)
    if weights_sum == 0:
        weights_all = np.ones(len(weights_all)) / len(weights_all)
    else:
        weights_all /= weights_sum
    num_samples = 100000
    indices = np.random.choice(len(weights_all), size=num_samples, p=weights_all)
    samples = np.vstack([m1_all[indices], m2_all[indices], m3_all[indices], sigma_all[indices]]).T
    return samples
def plot_corner(samples, hierarchy):
    labels = ['m1 [eV]', 'm2 [eV]', 'm3 [eV]', 'Sigma m_nu [eV]']
    fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={'fontsize': 12}, label_kwargs={'fontsize': 14}, color='blue' if hierarchy == 'NH' else 'red', hist_kwargs={'density': True})
    fig.suptitle('Posterior Distributions - ' + hierarchy + ' (Baseline + SJPV)', fontsize=16, y=1.02)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join('data', 'corner_plot_' + hierarchy + '_5_' + timestamp + '.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
def plot_mass_spectrum():
    mL = np.logspace(-4, -0.5, 500)
    dm2_21 = 7.49e-5
    dm2_31 = 2.513e-3
    m1_nh = mL
    m2_nh = np.sqrt(m1_nh**2 + dm2_21)
    m3_nh = np.sqrt(m1_nh**2 + dm2_31)
    sigma_nh = m1_nh + m2_nh + m3_nh
    dm2_32_abs = 2.484e-3
    m3_ih = mL
    m2_ih = np.sqrt(m3_ih**2 + dm2_32_abs)
    m1_ih = np.sqrt(m3_ih**2 + dm2_32_abs - dm2_21)
    sigma_ih = m1_ih + m2_ih + m3_ih
    plt.figure(figsize=(12, 8))
    plt.plot(mL, m1_nh, 'b-', label='m1 (NH)', linewidth=2)
    plt.plot(mL, m2_nh, 'b--', label='m2 (NH)', linewidth=2)
    plt.plot(mL, m3_nh, 'b:', label='m3 (NH)', linewidth=2)
    plt.plot(mL, sigma_nh, 'k-', label='Sigma m_nu (NH)', linewidth=2.5)
    plt.plot(mL, m1_ih, 'r-', label='m1 (IH)', linewidth=2)
    plt.plot(mL, m2_ih, 'r--', label='m2 (IH)', linewidth=2)
    plt.plot(mL, m3_ih, 'r:', label='m3 (IH)', linewidth=2)
    plt.plot(mL, sigma_ih, 'k--', label='Sigma m_nu (IH)', linewidth=2.5)
    plt.axhline(np.sqrt(dm2_31), color='blue', alpha=0.4, linestyle='-', label='NH m3 min (0.050 eV)')
    plt.axhline(np.sqrt(dm2_32_abs), color='red', alpha=0.4, linestyle='-', label='IH m1, m2 min (0.050 eV)')
    plt.axhline(0.05878, color='black', alpha=0.4, linestyle='-', label='NH Sigma m_nu min (0.059 eV)')
    plt.axhline(0.09892, color='black', alpha=0.4, linestyle='--', label='IH Sigma m_nu min (0.099 eV)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Lightest neutrino mass mL [eV]', fontsize=14)
    plt.ylabel('Mass [eV]', fontsize=14)
    plt.title('Neutrino Mass Spectrum and Sum (NuFIT 6.0)', fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join('data', 'mass_spectrum_5_' + timestamp + '.png')
    plt.savefig(filepath, dpi=300)
    plt.close()
if __name__ == '__main__':
    mu_cosmo = -0.009
    sigma_cosmo = 0.036
    samples_nh = generate_posterior_samples_mp('NH', mu_cosmo, sigma_cosmo, N_total=10000000, num_workers=8)
    samples_ih = generate_posterior_samples_mp('IH', mu_cosmo, sigma_cosmo, N_total=10000000, num_workers=8)
    samples_filepath = os.path.join('data', 'posterior_samples_step_5.npz')
    np.savez(samples_filepath, samples_nh=samples_nh, samples_ih=samples_ih)
    plot_corner(samples_nh, 'NH')
    plot_corner(samples_ih, 'IH')
    plot_mass_spectrum()