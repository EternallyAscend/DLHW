import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import string
import glob
from random_human import random_human
from scipy.stats import t


def plots(xs, ys, xlabel, ylabel, title, legends, loc="lower right", color=['b','y','g', 'r']):
    if not os.path.exists('figs'):
        os.makedirs('figs')
    for i,x in enumerate(xs):
        plt.plot(x,ys[i], linewidth=1.5,color=color[i],)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join('figs', title + ".pdf"))
    plt.close()

def plots_err(xs, ys, ystd, xlabel, ylabel, title, legends, loc="lower right", color=['b','y','g', 'r']):

    if not os.path.exists('figs'):
        os.makedirs('figs')
    for i,x in enumerate(xs):
        plt.plot(x,ys[i], color=color[i], linewidth=1.5,)
        if True: 
            plt.fill_between(x, np.array(ys[i])-2*np.array(ystd[i]), np.array(ys[i])+2*np.array(ystd[i]), color=color[i], alpha=0.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join('figs', title + ".pdf"))
    plt.close()

def find_checkpoint(base_path, postfix):
    model_dicts = []
    game_names = []
    for path_to_load in sorted(glob.glob(base_path + '/*' + postfix), reverse=False):
        for job_lib_file in sorted(glob.glob(path_to_load + '/*' + '_bestq.pkl'), reverse=False):
            model_dict = torch.load(job_lib_file, map_location=torch.device('cpu'))
            model_dicts.append(model_dict)
            game_name = str(os.path.basename(path_to_load))[:-2]
            game_names.append(game_name)
    return model_dicts, game_names

if __name__ == '__main__':
    base_path = '../swin_results/model_savedir/'
    model_dicts1, game_names1 = find_checkpoint(base_path, '00')
    model_dicts2, game_names2 = find_checkpoint(base_path, '01')

    legends = ['Swin DQN', 'Double DQN']
    perf_range = np.arange(0, 8, 0.1)
    perf_scores1 = np.zeros(len(perf_range))
    perf_scores2 = np.zeros(len(perf_range))

    for i, model_dict1 in enumerate(model_dicts1):
        model_dict2 = model_dicts2[i]
        game_name = game_names1[i]
        assert game_name == game_names2[i]

        info = model_dict1['info']
        perf1 = model_dict1['perf']
        perf2 = model_dict2['perf']
        titile_name = string.capwords(game_name.replace("_", " "))

        steps1 = perf1['steps']
        steps2 = perf2['steps']
        eval_steps1 = perf1['eval_steps']
        eval_steps2 = perf2['eval_steps']

        y1_mean_scores = perf1['eval_rewards']
        y1_std_scores = perf1['eval_stds']
        y1q = perf1['q_record']

        y2_mean_scores = perf2['eval_rewards']
        y2_std_scores = perf2['eval_stds']
        y2q = perf2['q_record']

    ## 标准化最高评分
        highest_score1 = (perf1['highest_eval_score'][-1]-random_human[game_name][0])/(random_human[game_name][1]-random_human[game_name][0])

        highest_score2 = (perf2['highest_eval_score'][-1]-random_human[game_name][0])/(random_human[game_name][1]-random_human[game_name][0])

        print(game_name, perf1['highest_eval_score'][-1], round(highest_score1,2), perf2['highest_eval_score'][-1], round(highest_score2, 2))

    ## 平均评分
        title = "Mean Evaluation Scores in "+ titile_name
        plots_err(
            [eval_steps1, eval_steps2],
            [y1_mean_scores, y2_mean_scores],
            [y1_std_scores, y2_std_scores],
            "Steps",
            "Scores",
            title,
            legends,
        )
    ## 最大Q值
        title = "Maximal Q-values in "+ titile_name
        plots(
            [steps1, steps2],
            [y1q, y2q],
            "Steps",
            "Q values",
            title,
            legends,
            loc="upper left"
        )
