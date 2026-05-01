# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def main():
    np.random.seed(42)
    data_dir = "data/"
    samples_filepath = os.path.join(data_dir, 'posterior_samples_step_5.npz')
    if not os.path.exists(samples_filepath):
        print("Error: " + samples_filepath + " not found. Please run Step 5 first.")
        return
    data = np.load(samples_filepath)
    samples_nh = data['samples_nh']
    samples_ih = data['samples_ih']
    m1_nh = samples_nh[:, 0]
    m2_nh = samples_nh[:, 1]
    m3_nh = samples_nh[:, 2]
    m1_ih = samples_ih[:, 0]
    m2_ih = samples_ih[:, 1]
    m3_ih = samples_ih[:, 2]
    N_nh = len(m1_nh)
    N_ih = len(m1_ih)
    s12_sq_nh = np.clip(np.random.normal(0.304, 0.012, N_nh), 0.0, 1.0)
    s12_sq_ih = np.clip(np.random.normal(0.304, 0.012, N_ih), 0.0, 1.0)
    s13_sq_nh = np.clip(np.random.normal(0.02220, 0.00062, N_nh), 0.0, 1.0)
    s13_sq_ih = np.clip(np.random.normal(0.02238, 0.00062, N_ih), 0.0, 1.0)
    c12_sq_nh = 1.0 - s12_sq_nh
    c13_sq_nh = 1.0 - s13_sq_nh
    c12_sq_ih = 1.0 - s12_sq_ih
    c13_sq_ih = 1.0 - s13_sq_ih
    alpha1_nh = np.random.uniform(0, 2 * np.pi, N_nh)
    alpha2_nh = np.random.uniform(0, 2 * np.pi, N_nh)
    alpha1_ih = np.random.uniform(0, 2 * np.pi, N_ih)
    alpha2_ih = np.random.uniform(0, 2 * np.pi, N_ih)
    term1_nh = c12_sq_nh * c13_sq_nh * m1_nh
    term2_nh = s12_sq_nh * c13_sq_nh * m2_nh * np.exp(1j * alpha1_nh)
    term3_nh = s13_sq_nh * m3_nh * np.exp(1j * alpha2_nh)
    mbb_nh = np.abs(term1_nh + term2_nh + term3_nh)
    term1_ih = c12_sq_ih * c13_sq_ih * m1_ih
    term2_ih = s12_sq_ih * c13_sq_ih * m2_ih * np.exp(1j * alpha1_ih)
    term3_ih = s13_sq_ih * m3_ih * np.exp(1j * alpha2_ih)
    mbb_ih = np.abs(term1_ih + term2_ih + term3_ih)
    mbb_nh_meV = mbb_nh * 1000.0
    mbb_ih_meV = mbb_ih * 1000.0
    mbb_nh_95 = np.percentile(mbb_nh_meV, 95)
    mbb_ih_95 = np.percentile(mbb_ih_meV, 95)
    mbb_nh_lower = np.percentile(mbb_nh_meV, 2.5)
    mbb_nh_upper = np.percentile(mbb_nh_meV, 97.5)
    mbb_ih_lower = np.percentile(mbb_ih_meV, 2.5)
    mbb_ih_upper = np.percentile(mbb_ih_meV, 97.5)
    print("=" * 60)
    print("EFFECTIVE MAJORANA MASS (m_bb) PROJECTIONS")
    print("=" * 60)
    print("Normal Hierarchy (NH):")
    print("  95th percentile upper limit: " + str(round(mbb_nh_95, 2)) + " meV")
    print("  95% Credible Interval:       [" + str(round(mbb_nh_lower, 2)) + ", " + str(round(mbb_nh_upper, 2)) + "] meV")
    print("  Median:                      " + str(round(np.median(mbb_nh_meV), 2)) + " meV")
    print("\nInverted Hierarchy (IH):")
    print("  95th percentile upper limit: " + str(round(mbb_ih_95, 2)) + " meV")
    print("  95% Credible Interval:       [" + str(round(mbb_ih_lower, 2)) + ", " + str(round(mbb_ih_upper, 2)) + "] meV")
    print("  Median:                      " + str(round(np.median(mbb_ih_meV), 2)) + " meV")
    print("=" * 60)
    output_data_path = os.path.join(data_dir, 'mbb_posterior_step_6.npz')
    np.savez(output_data_path, mbb_nh=mbb_nh, mbb_ih=mbb_ih)
    print("Posterior data saved to " + output_data_path)
    plt.rcParams['text.usetex'] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    bins = np.linspace(0, 60, 120)
    ax1.hist(mbb_nh_meV, bins=bins, density=True, alpha=0.6, color='blue', label='NH Posterior')
    ax1.hist(mbb_ih_meV, bins=bins, density=True, alpha=0.6, color='red', label='IH Posterior')
    ax1.axvline(5, color='green', linestyle='--', linewidth=2, label='nEXO (~5 meV)')
    ax1.axvline(20, color='orange', linestyle='--', linewidth=2, label='KamLAND-Zen (~20 meV)')
    ax1.set_xlabel('Effective Majorana Mass m_bb [meV]', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Posterior Distribution of m_bb', fontsize=14)
    ax1.set_xlim(0, 60)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    mbb_range = np.linspace(0.1, 60, 500)
    T_half = 1e27 * (20.0 / mbb_range)**2
    ax2.plot(mbb_range, T_half, 'k-', linewidth=2, label='T_1/2 proportional to 1/m_bb^2')
    T_nexo = 1e27 * (20.0 / 5.0)**2
    T_klz = 1e27 * (20.0 / 20.0)**2
    ax2.axhline(T_nexo, color='green', linestyle='--', linewidth=2, label='nEXO Sensitivity')
    ax2.axhline(T_klz, color='orange', linestyle='--', linewidth=2, label='KamLAND-Zen Sensitivity')
    ax2.axvspan(mbb_nh_lower, mbb_nh_upper, color='blue', alpha=0.15, label='NH 95% CI')
    ax2.axvspan(mbb_ih_lower, mbb_ih_upper, color='red', alpha=0.15, label='IH 95% CI')
    ax2.set_yscale('log')
    ax2.set_xlabel('Effective Majorana Mass m_bb [meV]', fontsize=12)
    ax2.set_ylabel('Half-life T_1/2 [years]', fontsize=12)
    ax2.set_title('Implied Lower Bound on T_1/2', fontsize=14)
    ax2.set_xlim(0, 60)
    ax2.set_ylim(1e25, 1e30)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filepath = os.path.join(data_dir, 'mbb_projections_6_' + timestamp + '.png')
    plt.savefig(plot_filepath, dpi=300)
    print("Plot saved to " + plot_filepath)

if __name__ == '__main__':
    main()