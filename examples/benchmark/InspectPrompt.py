'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    InspectPrompt.py
'''
import statistics
from typing import List, Dict
from matplotlib import pyplot as plt

def inspect_prompt_list(prompts: List[str], dataset_name: str, plot: bool = False) -> Dict:
    len_list = [len(p) for p in prompts]
    max_val = max(len_list)
    min_val = min(len_list)
    mean = statistics.mean(len_list)
    median = statistics.median(len_list)
    stdev = statistics.stdev(len_list)
    quantiles = statistics.quantiles(len_list, n=10)

    if plot:
        plt.cla()
        fname = f"histogram_{dataset_name}.png"
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Prompt length (token)')
        plt.ylabel('Counts')
        plt.title(f'{dataset_name}')
        bin_size = 500
        bins = [i * 500 for i in range(0, (max(len_list) + bin_size - 1) // bin_size + 1)]
        plt.hist(len_list, bins=bins, rwidth=0.85)
        plt.savefig(fname)

    return {
        'max': max_val,
        'min': min_val,
        'mean': mean,
        'median': median,
        'stdev': stdev,
        'quantiles': quantiles
    }
