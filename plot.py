import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.environ['ROOT_PATH'])


def read_fitness_scores(base_path):
    # Dictionary to hold generation-wise fitness scores
    fitness_scores = {}

    pattern = os.path.join(base_path, 'group_*', 'fitness_scores', '*_*.txt')
    files = glob.glob(pattern)

    # Read each file and extract the fitness scores
    for file in files:
        # Extract generationID from the filename
        filename = os.path.basename(file)
        generation_id = int(filename.split('_')[0])

        # Read the fitness score from the file
        with open(file, 'r') as f:
            score = float(f.read().strip())

        # Append the score to the list of scores for this generation
        if generation_id not in fitness_scores:
            fitness_scores[generation_id] = []
        fitness_scores[generation_id].append(score)

    return fitness_scores


def plot_fitness_scores(agg_type):
    plt.figure(figsize=(10, 8))
    map2label = {'revolve_database': 'REvolve', 'eureka_database': 'Eureka', 'revolve_auto_database': 'REvolve Auto',
                 'eureka_auto_database': 'Eureka Auto', 't2r_database': 'T2R'}
    # generations = np.arange(0, 7, 1)
    for baseline in baselines:
        base_path = os.path.join(os.environ["ROOT_PATH"], f'{baseline}/gpt-4')
        fitness_scores = read_fitness_scores(base_path)
        generations = np.arange(0, 7, 1)[:len(fitness_scores)]
        if agg_type == 'Average':
            agg_scores = [sum(scores) / len(scores) for scores in (fitness_scores[gen] for gen in generations)]
        else:  # Best
            agg_scores = [max(scores) for scores in (fitness_scores[gen] for gen in generations)]
        plt.plot(generations, agg_scores, marker='o', label=map2label[baseline])

    plt.plot(generations, [0.90] * len(generations), label='Human Driving')
    plt.plot(generations, [0.65] * len(generations), label='Human Designed Reward')
    plt.xticks(np.arange(0, 6.5, 1))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel(f'{agg_type} Fitness Score', fontsize=16)
    plt.legend(fontsize=20)
    # plt.title('Fitness Score', fontsize=18)
    # plt.savefig(f'{plot_path}/combined_fitness.png')
    plt.grid(True)
    plt.show()


# Base directory path
# Read and plot the fitness scores
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Fitness Score')
    parser.add_argument('--agg_type', required=True, type=str, choices=['Best', 'Average'],
                        help='Fitness Score aggregation type')
    args = parser.parse_args()
    baselines = ['revolve_database', 'eureka_database', 'revolve_auto_database', 'eureka_auto_database', 't2r_database']
    plot_fitness_scores(agg_type=args.agg_type)
