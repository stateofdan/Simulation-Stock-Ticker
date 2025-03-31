import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def fit_distribution(data):
    distributions = [stats.norm, stats.lognorm, stats.expon, stats.gamma, stats.beta]
    results = {}
    for distribution in distributions:
        params = distribution.fit(data)
        aic = -2 * np.sum(np.log(distribution.pdf(data, *params))) + 2 * len(params)
        bic = -2 * np.sum(np.log(distribution.pdf(data, *params))) + len(params) * np.log(len(data))
        results[distribution.name] = {'params': params, 'aic': aic, 'bic': bic, 'best_aic': False, 'best_bic': False}
    best_fit_aic = min(results, key=lambda x: results[x]['aic'])
    results[best_fit_aic]['best_aic'] = True
    best_fit_bic = min(results, key=lambda x: results[x]['bic'])
    results[best_fit_bic]['best_bic'] = True
    results = {'best_fit_aic': best_fit_aic, 'best_fit_bic': best_fit_bic, 'results': results}
    return results

def plot_distributions(data, results):
    plt.figure(figsize=(12, 8))
    sns.histplot(data, bins=30, kde=False, stat='density', color='gray', alpha=0.5)

    x = np.linspace(min(data), max(data), 1000)
    for name, params in results.items():
        print (f'{name}: {params['params']}')
        dist = getattr(stats, name)
        plt.plot(x, dist.pdf(x, *params['params']), label=name)

    plt.legend()
    plt.title('Fitted Distributions')
    #plt.show()