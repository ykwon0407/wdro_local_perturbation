from glob import glob
import numpy as np
import json
import pandas as pd
from scipy.stats import ttest_ind

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')

model_list = ['erm','mixup','l2_0.001','l2_0.002','l2_0.004','l2_0.008']
name_list=['ERM','Mixup','l2_0.001','l2_0.002','l2_0.004','l2_0.008']
color_list=['gray', 'red', 'skyblue', 'dodgerblue','blue', 'navy'] # darkslateblue

'''
Summary clean samples
'''
sample_size_list = [2500,5000,25000,50000]
label_font_size = 22
title_font_size = 25
legend_font_size = 18
SIGMA = 1. # 1.96
n_last = 1 # 20
t_sig_level = 0.05

def summarize_final_results(dataset='cifar10', n_last=n_last, result_path='../results'):
    '''
    Calculate the mean and standard deviation of results of the last 'n_last' epochs.
    '''        
    split_list = ['tr','val','test']
    df = pd.DataFrame(columns=['data','model','sample','mean','std'])

    for model in model_list:
        for sample in sample_size_list:
            path_list = sorted(
                        glob(
                            result_path+'/{}/{}.*@{}-1/accuracies.txt'.format(model, dataset, sample)
                        ))
            tmp = []
            for path in path_list:
                f = json.load(open(path, "r"))
                tmp.append(pd.DataFrame(f, index=split_list))
            
            assert len(tmp) == 5, 'Check the number of experiments (seeds), we run 5 times'
            tmp = pd.concat(tmp, sort=False)
            tmp_test = tmp.loc['test'].iloc[:,-n_last:]
            tmp_test = np.array(tmp_test)

            tmp_dict = {'data': dataset,
                        'model': model,
                        'sample': sample,
                        'mean': np.mean(tmp_test), 
                        'std': np.std(np.mean(tmp_test, axis=1)),
                        'raw_data': np.mean(tmp_test, axis=1)}
            df = df.append(tmp_dict, ignore_index=True)

    return df    


def print_t_test_results(dataset='cifar10'):
    df_data = summarize_final_results(dataset=dataset, n_last=n_last)

    for sample_size in sample_size_list:
        print('-'*10)
        print('dataset:',dataset)
        print('sample:', sample_size)
        print('-'*10)
        
        df = df_data[df_data['sample']==sample_size].reset_index()
        best_acc_index = df['mean'].values.argmax()
        for j in range(df.shape[0]):
            t_result = ttest_ind(df['raw_data'][best_acc_index], df['raw_data'][j], equal_var=False)
            print(name_list[j], '/ p-value: ', t_result.pvalue, '/ is equivalnet? ', t_result.pvalue > t_sig_level)

def show_summary_plot(dataset='cifar10'):
    df_data = summarize_final_results(dataset=dataset, n_last=n_last)

    plt.figure(figsize=(8,6))
    # plt.xlabel("$\log$(# of samples)", fontsize=label_font_size)
    plt.xlabel("Sample size $n$", fontsize=label_font_size)
    plt.ylabel('Test accuracy (in %)', fontsize=label_font_size)
    plt.title('Dataset: {}'.format(dataset), fontsize=title_font_size)
    
    for i in range(len(color_list)):
        ymean = df_data['mean'][df_data['model']==model_list[i]]
        yerr = df_data['std'][df_data['model']==model_list[i]]
        plt.plot(sample_size_list, ymean, marker='o', color=color_list[i], label=name_list[i], alpha=0.75) 
        plt.fill_between(sample_size_list, ymean-SIGMA*yerr, ymean+SIGMA*yerr,
                         alpha=0.2, edgecolor=color_list[i],
                         facecolor=color_list[i])
    plt.legend(loc='lower right', borderaxespad=0., fontsize=legend_font_size)

    
def print_summary_table(dataset='cifar10'):
    df_data = summarize_final_results(dataset, n_last=n_last)
    ncol = 2 + len(df_data['model'].drop_duplicates()) + 2
    nrow = 1 + len(df_data['sample'].drop_duplicates())
    model_names = [data.replace('_', '-') for data in df_data['model'].drop_duplicates()]
    line = str(r'\begin{tabular}{' + 'c'*ncol + '}\n'
               + r'\hline' + '\n'
              )
    line = line + str(r'\hline' + '\n'
                      + r'\hline' + '\n'
                      + r'\multirow{5}{*}{%s}' % df_data['data'][0] + '\n'
                     ) 
    for sample in sorted(df_data['sample'].drop_duplicates()):
        line = line + '& %d ' % sample
        for model in df_data['model'].drop_duplicates():
            ind = (df_data['sample'] == sample) & (df_data['model'] == model)
            line = line + r'&$%.1f\pm%.1f$ ' % (df_data['mean'][ind], df_data['std'][ind])
        line = line + r'\tabularnewline' + '\n'
        # line = line + r'\cline{2-%d}' % ncol + '\n')
    line = line[:-12]
    line = line + r'\hline' + '\n' + r'\end{tabular}'
    
    print(line)



'''
Summary noisy samples
'''

def summarize_noise_results(dataset='cifar10', result_path='../results', **kwargs):
    '''
    Calculate the mean and standard deviation of accuracies with noisy images.
    '''
    mean = kwargs.get('mean', 0)
    var = kwargs.get('var', None)
    mode = kwargs.get('mode', None)
    p = kwargs.get('p', None)

    if mode is 'gaussian':
        noise_settings = {'Mode':'Gaussian', 'Mean':mean, 'Var':var, 'seed':None}
    elif mode is not None:
        noise_settings = {'Mode':mode, 'p':p, 'seed':None}
    else:
        assert False, 'mode is None'

    split_list = ['noise_tr','noise_val','noise_test','gap_tr','gap_val','gap_test']
    df = pd.DataFrame(columns=['data','model','sample','noise_settings',
                               'noise_tr_mean','noise_tr_std','noise_test_mean','noise_test_std',
                               'gap_tr_mean','gap_tr_std','gap_test_mean','gap_test_std',])

    for model in model_list:
        for sample in sample_size_list:
            path_list = sorted(
                        glob(
                            result_path+'/{}/{}.*@{}-1/noise.txt'.format(model, dataset, sample)
                        ))
            tmp = []
            for path in path_list:
                f = json.load(open(path, "r"))
                f_key = str(noise_settings)[:-1] + ", 'steps': '06553600'}"
                try:
                    acc = f[str(f_key)]
                except:
                    print(path)
                    print(str(f_key))
                    acc = f[str(f_key)]
                # acc: [noise_tr, noise_val, noise_test, tr, val, test]
                acc[3] = acc[0] - acc[3]
                acc[4] = acc[1] - acc[4]
                acc[5] = acc[2] - acc[5]
                tmp.append(pd.DataFrame(acc, index=split_list))
            tmp = pd.concat(tmp, sort=False)
            tmp_mean = tmp.groupby(level=0).mean()
            tmp_std = tmp.groupby(level=0).std()
            tmp_dict = {'data': dataset,
                        'model': model,
                        'sample': sample,
                        'noise_settings':str(noise_settings),
                        'noise_tr_mean': tmp_mean.loc['noise_tr'].iloc[0],
                        'noise_tr_std': tmp_std.loc['noise_tr'].iloc[0],
                        'noise_test_mean': tmp_mean.loc['noise_test'].iloc[0],
                        'noise_test_std': tmp_std.loc['noise_test'].iloc[0],
                        'gap_tr_mean': tmp_mean.loc['gap_tr'].iloc[0],
                        'gap_tr_std': tmp_std.loc['gap_tr'].iloc[0],
                        'gap_test_mean': tmp_mean.loc['gap_test'].iloc[0],
                        'gap_test_std': tmp_std.loc['gap_test'].iloc[0],
                        'noise_test_raw_data': np.array(tmp.loc['noise_test']).reshape(-1),
                       }
            df = df.append(tmp_dict, ignore_index=True)
    return df

def print_t_test_noise_results(dataset='cifar10', **kwargs):
    mean = kwargs.get('mean', 0)
    var = kwargs.get('var', None)
    mode = kwargs.get('mode', None)
    p = kwargs.get('p', None)

    if mode is 'gaussian':
        noise_settings = {'Mode':'Gaussian', 'Mean':mean, 'Var':var, 'seed':None}
    elif mode is not None:
        noise_settings = {'Mode':mode, 'p':p, 'seed':None}
    else:
        assert False, 'mode is None'
        
    df_data = summarize_noise_results(dataset=dataset, result_path='../results', **kwargs)
    
    for sample_size in sample_size_list:
        print('-'*10)
        print('dataset:',dataset)
        print('sample:', sample_size)
        print('noise:', noise_settings)
        print('-'*10)
        
        df = df_data[df_data['sample']==sample_size].reset_index()
        best_acc_index = df['noise_test_mean'].values.argmax()
        for j in range(df.shape[0]):
            t_result = ttest_ind(df['noise_test_raw_data'][best_acc_index], df['noise_test_raw_data'][j], equal_var=False)
            print(name_list[j], '/ p-value: ', t_result.pvalue, '/ is equivalnet? ', t_result.pvalue > t_sig_level)

def show_noise_plot(dataset='cifar10', split='train', **kwargs):
    df_data = summarize_noise_results(dataset=dataset, **kwargs)
    
    plt.figure(figsize=(8,6))
    # plt.xlabel('$\log$(# of samples)', fontsize=label_font_size)
    plt.xlabel("Sample size $n$", fontsize=label_font_size)
    plt.ylabel('Accuracy (in %)', fontsize=label_font_size)
    plt.title('Dataset: {}, Noise:{}'.format(dataset, df_data['noise_settings'][0]), fontsize=title_font_size)
    
    if split is 'train':
        for i in range(len(color_list)):
            ymean = df_data['noise_tr_mean'][df_data['model']==model_list[i]]
            yerr = df_data['noise_tr_std'][df_data['model']==model_list[i]]
            plt.plot(sample_size_list, ymean, '--', marker='o', color=color_list[i], alpha=0.5) 
            plt.fill_between(sample_size_list, ymean-SIGMA*yerr, ymean+SIGMA*yerr,
             alpha=0.1, edgecolor=color_list[i], facecolor=color_list[i])
    elif split is 'test':
        for i in range(len(color_list)):
            ymean = df_data['noise_test_mean'][df_data['model']==model_list[i]]
            yerr = df_data['noise_test_std'][df_data['model']==model_list[i]]
            plt.plot(sample_size_list, ymean, '--', marker='o', color=color_list[i], label=name_list[i], alpha=0.75) 
            plt.fill_between(sample_size_list, ymean-SIGMA*yerr, ymean+SIGMA*yerr,
             alpha=0.3, edgecolor=color_list[i], facecolor=color_list[i])
    else:
        assert False, 'check split (train or test)'
    
    plt.legend(loc='lower right', borderaxespad=0., fontsize=legend_font_size)
    plt.show()    


def print_summary_noise_table(dataset='cifar10', split='test', **kwargs):
    df_data = summarize_noise_results(dataset, **kwargs)
    ncol = 2 + len(df_data['model'].drop_duplicates())
    nrow = 1 + len(df_data['sample'].drop_duplicates())
    model_names = [data.replace('_', '-') for data in df_data['model'].drop_duplicates()]
    line = str(r'\begin{tabular}{|' + 'c|'*ncol + '}\n'
               + r'\hline' + '\n'
              )
    line = line + r'Dataset & Sample size'
    for model_name in model_names:
        line = line + ' & ' + model_name
    line = line + r'\tabularnewline' + '\n'
    line = line + str(r'\hline' + '\n'
                      + r'\hline' + '\n'
                      + r'\multirow{5}{*}{%s}' % df_data['data'][0] + '\n'
                     ) 
    for sample in sorted(df_data['sample'].drop_duplicates()):
        if split is 'test':
            mean_ind, std_ind = 'noise_test_mean', 'noise_test_std'
        elif split is 'train':
            mean_ind, std_ind = 'noise_tr_mean', 'noise_tr_std'
        else:
            assert False, 'split must be train or test'
        line = line + '& %d ' % sample
        for model in df_data['model'].drop_duplicates():
            ind = (df_data['sample'] == sample) & (df_data['model'] == model)
            line = line + r'& $%.1f\pm%.1f$ ' % (df_data[mean_ind][ind], df_data[std_ind][ind])
        line = line + r'\tabularnewline' + '\n'
        # line = line + r'\cline{2-%d}' % ncol + '\n'
    line = line[:-12]
    line = line + r'\hline' + '\n' + r'\end{tabular}'
    print(line)


'''
def show_noise_gap_plot(dataset='cifar10', split='test', **kwargs):
    df_data = summarize_noise_results(dataset=dataset, **kwargs)
    
    plt.figure(figsize=(8,6))
    # plt.xlabel('$log$(# of samples)', fontsize=label_font_size)
    plt.xlabel("Sample size $n$", fontsize=label_font_size)
    plt.ylabel('Accuracy (in %)', fontsize=label_font_size)
    plt.title('Dataset: {}, Noise:{}'.format(dataset, df_data['noise_settings'][0]), fontsize=title_font_size)
    
    if split is 'train':
        for i in range(len(color_list)):
            ymean = df_data['gap_tr_mean'][df_data['model']==model_list[i]]
            yerr = df_data['gap_tr_std'][df_data['model']==model_list[i]]
            plt.plot(sample_size_list, ymean, '--', marker='o',
             color=color_list[i], alpha=0.5, label=name_list[i]) 
            plt.fill_between(sample_size_list, ymean-SIGMA*yerr, ymean+SIGMA*yerr,
             alpha=0.1, edgecolor=color_list[i], facecolor=color_list[i])
    elif split is 'test':
        for i in range(len(color_list)):
            #test plot
            ymean = df_data['gap_test_mean'][df_data['model']==model_list[i]]
            yerr = df_data['gap_test_std'][df_data['model']==model_list[i]]
            plt.plot(sample_size_list, ymean, '--', marker='o',
             color=color_list[i], label=name_list[i], alpha=0.75) 
            plt.fill_between(sample_size_list, ymean-SIGMA*yerr, ymean+SIGMA*yerr,
             alpha=0.3, edgecolor=color_list[i], facecolor=color_list[i])
    else:
        assert False, 'check split (train or test)'
    
    plt.legend(loc='lower right', borderaxespad=0., fontsize=legend_font_size)
    plt.show()
 
'''