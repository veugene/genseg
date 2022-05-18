<<<<<<< HEAD
import pprint
import os
import re

import numpy as np

directory = "/scratch/mgazda/genseg_mai/ablation_lits/lits_4fold/bds3/"

sim_names = ['bds3_001LZ_dice100_sum1 (f0.15)',
             'bds3_001LZ_dice100_sum1_rec200 (f0.15)',
             'bds3_001LZ_dice100_sum1_rec200_z5 (f0.15)',
             'bds3_003LZ_dice100_sum1 (f0.15)',
             'bds3_001LZ_dice10_sum1 (f0.15)',
             'bds3_001LZ_dice10_sum1_rec200 (f0.15)',
             'bds3_001LZ_dice10_sum1_rec200_z5 (f0.15)',
             'bds3_003LZ_dice10_sum1 (f0.15)',
             'bds3_001LZ_dice1_sum1 (f0.15)',
             'bds3_001LZ_dice1_sum1_rec200 (f0.15)',
             'bds3_001LZ_dice1_sum1_rec200_z5 (f0.15)',
             'bds3_003LZ_dice1_sum1 (f0.15)',
             ]


# directory = '/scratch/mgazda/genseg_mai/ablation_lits/ae_seg/lits_4fold/ae/'
# sim_names = ['ae_001LZ (f0.15)']

def get_best_score(f):
    lines = f.readlines()
    best_score = 0
    best_l_num = 0
    l_num = 0
    regex = re.compile('(?<=dice\=)-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?')
    for l in lines:
        result = regex.search(l)
        if result:
            l_num += 1
            score = float(l[result.start():result.end()])
            if score < best_score:
                best_score = score
                best_l_num = l_num
    return best_score, best_l_num, len(lines)

def get_best_test_score(f):
    lines = f.readlines()
    regex = re.compile('(?<=dice\=)-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?')
    for index, l in enumerate(lines):
        if index == 1:
            result = regex.search(l)
            if result:
                score = float(l[result.start():result.end()])
                return score




class Result:
    def __init__(self, name):
        self.name = name
        self.folds = {0: {'1234': 0, '4444': 0, '8888': 0},
                      1: {'1234': 0, '4444': 0, '8888': 0},
                      2: {'1234': 0, '4444': 0, '8888': 0},
                      3: {'1234': 0, '4444': 0, '8888': 0}
                      }

    def get_results(self):
        for fold in self.folds.keys():
            for seed in self.folds[fold].keys():
                val_log = directory + self.name + "/" + str(fold) + "/" + seed
                f = open(r"{}".format(os.path.join(val_log+ "/val_log.txt")), 'r')
                best_score, _, _ = get_best_score(f)
                self.folds[fold][seed] = best_score
        return self.folds

results = {}
for sim in sim_names:
    result = Result(sim)
    result.get_results()
    results[sim] = result.get_results()

results_avg_across_seeds = {}
# we need the best config across all the seeds. this is how it looks like
# {'bds3_001LZ_dice100_sum1 (f0.15)':
# {0: {'1234': -0.705, '4444': -0.705, '8888': -0.695},
# 1: {'1234': -0.793, '4444': -0.805, '8888': -0.802},
# 2: {'1234': -0.822, '4444': -0.83, '8888': -0.832},
# 3: {'1234': -0.858, '4444': -0.864, '8888': -0.861}},
# 'bds3_001LZ_dice100_sum1_rec200 (f0.15)':
# {0: {'1234': -0.717, '4444': -0.702, '8888': -0.721},
# 1: {'1234': -0.804, '4444': -0.808, '8888': -0.803},
# 2: {'1234': -0.829, '4444': -0.83, '8888': -0.836},
# 3: {'1234': -0.86, '4444': -0.867, '8888': -0.871}},
for simulation_name in results.keys():
    results_avg_across_seeds[simulation_name] = {}
    for seed in ['1234', '4444', '8888']:
        results_avg_across_seeds[simulation_name][seed] = np.average([results[simulation_name][0][seed],
                                                                      results[simulation_name][1][seed],
                                                                      results[simulation_name][2][seed],
                                                                      results[simulation_name][3][seed],
                                                                      ])

final_results = {}
seeds = ['1234', '4444', '8888']

print('RESULTS AVG ACROSS SEEDS')
print(results_avg_across_seeds)

print('SIM_NAMES: {}'.format(sim_names))

for seed in seeds:
    temp = [results_avg_across_seeds[sim_name][seed] for sim_name in sim_names]
    print('TEMP: {}'.format(temp))
    argmax_result = np.argmin(temp)
    print('BEST WAS: {}'.format(sim_names[argmax_result]))
    final_results[seed] = sim_names[argmax_result]

print('FINAL RESULTS')
print(final_results)
folds = [0, 1, 2, 3]

test_final_results_averaged_across_seeds = {}
for seed in seeds:
    res = []
    for fold in folds:
        val_log = directory + final_results[seed] + "/" + str(fold) + "/" + seed
        f = open(r"{}".format(os.path.join(val_log+ "/test_log.txt")), 'r')
        best_score = get_best_test_score(f)
        res.append(best_score)
        print('for seed: {} and fold: {} the best results is: {}'.format(seed, fold, best_score))
    print(res)
    test_final_results_averaged_across_seeds[seed] = np.average(res)

print(test_final_results_averaged_across_seeds)



print('---------------------------------')
pprint.pprint(results)
