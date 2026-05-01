# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from getdist.mcsamples import loadMCSamples
from scipy.stats import norm
from scipy.optimize import minimize
import datetime

def trunc_gauss_pdf(x, mu, sigma):
    pdf = norm.pdf(x, loc=mu, scale=sigma)
    cdf_0 = norm.cdf(0, loc=mu, scale=sigma)
    if cdf_0 >= 0.999999:
        return np.zeros_like(x) if not np.isscalar(x) else 0.0
    res = pdf / (1.0 - cdf_0)
    if np.isscalar(x):
        if x < 0: return 0.0
    else:
        res[x < 0] = 0.0
    return res

def trunc_gauss_cdf(x, mu, sigma):
    cdf = norm.cdf(x, loc=mu, scale=sigma)
    cdf_0 = norm.cdf(0, loc=mu, scale=sigma)
    if cdf_0 >= 0.999999:
        return np.ones_like(x) if not np.isscalar(x) else 1.0
    res = (cdf - cdf_0) / (1.0 - cdf_0)
    if np.isscalar(x):
        if x < 0: return 0.0
    else:
        res[x < 0] = 0.0
    return res

if __name__ == '__main__':
    data_dir = 'data/'
    base_url = 'https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/cobaya/'
    datasets = {'baseline': 'base_mnu/desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck-NPIPE-highl-CamSpec-TTTEEE_planck-act-dr6-lensing/', 'plik': 'base_mnu/desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck2018-highl-plik-TTTEEE_planck-act-dr6-lensing/', 'lh': 'base_mnu/desi-bao-all_planck2018-lowl-TT-clik_planck2020-lollipop-lowlE_planck2020-hillipop-TTTEEE_planck-act-dr6-lensing/', 'w0wa': 'base_mnu_w_wa/desi-bao-all_planck2018-lowl-TT-clik_planck2018-lowl-EE-clik_planck-NPIPE-highl-CamSpec-TTTEEE_planck-act-dr6-lensing/'}
    for name, path in datasets.items():
        for i in range(1, 5):
            url = base_url + path + 'chain.' + str(i) + '.txt'
            dest = os.path.join(data_dir, name + '.' + str(i) + '.txt')
            if not os.path.exists(dest):
                urllib.request.urlretrieve(url, dest)
    samples_dict = {}
    for name in datasets.keys():
        root = os.path.join(data_dir, name)
        s = loadMCSamples(root, settings={'ignore_rows': 0.3})
        samples_dict[name] = s
    limits_95 = {'baseline': 0.0642, 'plik': 0.0691, 'lh': 0.0774, 'w0wa': 0.163}
    fit_params = {}
    kde_profiles = {}
    for name, s in samples_dict.items():
        p = s.get1DDensity('mnu')
        kde_profiles[name] = {'x': p.x, 'P': p.P}
        limit = limits_95[name]
        def loss(params):
            mu, sigma = params
            pdf_fit = trunc_gauss_pdf(p.x, mu, sigma)
            mse = np.mean((pdf_fit - p.P)**2)
            cdf_95 = trunc_gauss_cdf(limit, mu, sigma)
            penalty = 1e6 * (cdf_95 - 0.95)**2
            return mse + penalty
        samps = s.samples[:, s.index['mnu']]
        weights = s.weights
        mean_guess = np.average(samps, weights=weights)
        std_guess = np.sqrt(np.average((samps - mean_guess)**2, weights=weights))
        res = minimize(loss, [mean_guess, std_guess], method='L-BFGS-B', bounds=[(-0.5, 0.5), (0.001, 0.5)])
        mu_fit, sigma_fit = res.x
        fit_params[name] = {'mu': mu_fit, 'sigma': sigma_fit}
    plt.figure(figsize=(10, 6))
    x_vals = np.linspace(0, 0.2, 500)
    p_base = kde_profiles['baseline']
    plt.plot(p_base['x'], p_base['P'], label='KDE (DESI DR2 Baseline)', color='black', lw=2)
    mu_fit = fit_params['baseline']['mu']
    sigma_fit = fit_params['baseline']['sigma']
    pdf_fit = trunc_gauss_pdf(x_vals, mu_fit, sigma_fit)
    plt.plot(x_vals, pdf_fit, label='Truncated Gaussian Fit (mu=' + str(round(mu_fit, 3)) + ', sigma=' + str(round(sigma_fit, 3)) + ')', color='blue', linestyle='--')
    mu_fc = -0.036
    sigma_fc = 0.043
    pdf_fc = trunc_gauss_pdf(x_vals, mu_fc, sigma_fc)
    plt.plot(x_vals, pdf_fc, label='Feldman-Cousins Profile (mu=-0.036, sigma=0.043)', color='red', linestyle='-.')
    plt.xlabel('Sigma m_nu [eV]', fontsize=14)
    plt.ylabel('Posterior Density', fontsize=14)
    plt.title('Cosmological Likelihood Comparison (DESI DR2 Baseline)', fontsize=16)
    plt.xlim(0, 0.15)
    plt.ylim(0, max(p_base['P']) * 1.2)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = os.path.join(data_dir, 'likelihood_comparison_1_' + timestamp + '.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    save_data = {}
    for name, s in samples_dict.items():
        samps = s.samples[:, s.index['mnu']]
        weights = s.weights
        save_data[name + '_samples'] = samps
        save_data[name + '_weights'] = weights
        save_data[name + '_mu'] = fit_params[name]['mu']
        save_data[name + '_sigma'] = fit_params[name]['sigma']
        save_data[name + '_kde_x'] = kde_profiles[name]['x']
        save_data[name + '_kde_p'] = kde_profiles[name]['P']
    save_data['fc_mu'] = mu_fc
    save_data['fc_sigma'] = sigma_fc
    npz_filename = os.path.join(data_dir, 'desi_dr2_cosmo_likelihood.npz')
    np.savez(npz_filename, **save_data)