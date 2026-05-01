# filename: codebase/step_3.py
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
sys.path.insert(0, os.path.abspath('codebase'))
from step_2 import compute_evidence
def task(args):
    year, limit, source, prior, hierarchy = args
    seed_str = str(year) + '_' + str(limit) + '_' + prior + '_' + hierarchy
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    mu = 0.0
    sigma = limit / 1.96
    E = compute_evidence(hierarchy, prior, mu, sigma, N=1000000, chunk_size=100000)
    return year, limit, source, prior, hierarchy, E
if __name__ == '__main__':
    historical_data = [(2002.0, 1.80, '2dFGRS'), (2012.0, 0.44, 'WMAP-9'), (2013.0, 0.25, 'Planck 2013'), (2015.0, 0.18, 'Planck 2015'), (2016.0, 0.13, 'Planck+BAO'), (2021.0, 0.102, 'Planck+BAO+RSD'), (2021.3, 0.099, 'Planck+BAO+RSD+SNe'), (2021.6, 0.089, 'Planck+BAO+Ly-alpha'), (2022.0, 0.071, 'DESI DR1'), (2025.0, 0.0642, 'DESI DR2')]
    priors = ['SJPV', 'HS']
    hierarchies = ['NH', 'IH']
    tasks = []
    for year, limit, source in historical_data:
        for prior in priors:
            for hierarchy in hierarchies:
                tasks.append((year, limit, source, prior, hierarchy))
    results_dict = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(task, tasks):
            year, limit, source, prior, hierarchy, E = res
            key = (year, limit, source)
            if key not in results_dict:
                results_dict[key] = {}
            if prior not in results_dict[key]:
                results_dict[key][prior] = {}
            results_dict[key][prior][hierarchy] = E
    delta_chi2 = 6.1
    osc_factor = np.exp(delta_chi2 / 2.0)
    final_results = []
    for key in sorted(results_dict.keys(), key=lambda x: x[0]):
        year, limit, source = key
        for prior in priors:
            E_NH = results_dict[key][prior]['NH']
            E_IH = results_dict[key][prior]['IH']
            K_base = E_NH / E_IH if E_IH > 0 else np.inf
            K_full = K_base * osc_factor
            final_results.append({'year': year, 'limit': limit, 'source': source, 'prior': prior, 'E_NH_base': E_NH, 'E_IH_base': E_IH, 'K_base': K_base, 'K_full': K_full})
    output_json = os.path.join('data', 'historical_evidence_step_3.json')
    with open(output_json, 'w') as f:
        json.dump(final_results, f, indent=4)
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(12, 7))
    years_sjpv = [r['year'] for r in final_results if r['prior'] == 'SJPV']
    k_full_sjpv = [r['K_full'] for r in final_results if r['prior'] == 'SJPV']
    years_hs = [r['year'] for r in final_results if r['prior'] == 'HS']
    k_full_hs = [r['K_full'] for r in final_results if r['prior'] == 'HS']
    plt.plot(years_sjpv, k_full_sjpv, marker='o', markersize=8, linestyle='-', linewidth=2, color='#1f77b4', label='SJPV Prior (Total K)')
    plt.plot(years_hs, k_full_hs, marker='s', markersize=8, linestyle='--', linewidth=2, color='#d62728', label='HS Prior (Total K)')
    plt.axhline(y=3.2, color='gray', linestyle=':', alpha=0.7)
    plt.text(2001.2, 3.5, 'Substantial (K > 3.2)', color='gray', fontsize=10, fontweight='bold')
    plt.axhline(y=10, color='gray', linestyle=':', alpha=0.7)
    plt.text(2001.2, 11, 'Strong (K > 10)', color='gray', fontsize=10, fontweight='bold')
    plt.axhline(y=100, color='gray', linestyle=':', alpha=0.7)
    plt.text(2001.2, 110, 'Decisive (K > 100)', color='gray', fontsize=10, fontweight='bold')
    plt.axvline(x=2022, color='green', linestyle='-.', alpha=0.6, linewidth=2)
    trans = plt.gca().get_xaxis_transform()
    plt.text(2021.7, 0.5, 'Jimenez et al. 2022', color='green', rotation=90, verticalalignment='center', transform=trans, fontsize=11, fontweight='bold')
    plt.yscale('log')
    plt.xlim(2001, 2026)
    plt.xlabel('Year')
    plt.ylabel('Bayes Factor K (NH vs IH)')
    plt.title('Historical Evolution of Bayesian Evidence for Normal Hierarchy')
    plt.grid(True, which='major', ls='-', alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filepath = os.path.join('data', 'historical_bayes_factor_3_' + timestamp + '.png')
    plt.savefig(plot_filepath, dpi=300)