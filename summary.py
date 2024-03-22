import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

cur_dir = os.getcwd()
output_dir = os.path.join(cur_dir, 'output')


class Visualization(object):
    def __init__(self, model_list, n_tail=10, show_outliers=False):
        self.model_list = model_list
        self.model_num = len(model_list)
        # self.metric_list = metric_list
        self.n_tail = n_tail
        self.show_outliers = show_outliers
        
        self.df = self.load_progress()
        self.df = self.df.reindex(sorted(self.df.columns), axis=1)
        self.df_exp_mean =  self.df.groupby(['model', 'exp']).mean().reset_index()
        
    
    def load_progress(self):
        df_list = []
        for model in self.model_list:
            model_dir = os.path.join(output_dir, model)
            print(model)
            for exp in os.listdir(model_dir):
                results_path = os.path.join(model_dir, exp, 'progress.csv')
                if os.path.exists(results_path):
                    df_temp = pd.read_csv(results_path)
                    df_temp = df_temp[[col for col in df_temp.columns if col.startswith('metric')]]
                    df_temp = df_temp.dropna()
                    df_temp = df_temp.tail(self.n_tail)
                    df_temp.insert(loc=0, column='exp', value=int(exp.split('_')[-1]))
                    df_temp.insert(loc=0, column='model', value=model)

                else:
                    print(os.path.join(model_dir, exp))
                    print('no result')
                    continue
                # print(df_temp)
                try:
                    df_list.append(df_temp)
                except:
                    print(exp)


        df = pd.concat(df_list).reset_index(drop=True)
        df.columns = [col.split('/')[-1] for col in df.columns]
        return  df


    def filter_outliers(self, group_col, value_col):
        Q1 = self.df.groupby(group_col)[value_col].transform('quantile', 0.25)
        Q3 = self.df.groupby(group_col)[value_col].transform('quantile', 0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        print(self.df)
        return self.df[(self.df[value_col] >= lower_bound) & (self.df[value_col] <= upper_bound)]


    def trimmed_dataframe(self, df, group_col, value_col):
        # Determine the quantiles for trimming
        def trim_group(group):
            Q1 = group[value_col].quantile(0.25)
            Q3 = group[value_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return group[(group[value_col] >= lower_bound) & (group[value_col] <= upper_bound)]
        
        return df.groupby(group_col).apply(trim_group).reset_index(drop=True)
    
    
    def print_metrics(self):
        df = self.df.copy()
        
        metric_avg_list = ['cr', 'mse', 'range']
        
        for metric in metric_avg_list:
            df[metric+'_avg'] = df[[col for col in df.columns if col.startswith(metric)]].mean(axis=1)
        
        save_path = os.path.join(cur_dir,'metrics.txt')
        with open(save_path, 'w') as f:
            
            df_median = df.groupby('model').median().round(5)
            f.write('\n ---------------------------- median -----------------------------\n')
            f.write(df_median.to_string())
            
            for metric in metric_avg_list:
                # print(df[metric+'_avg'])
                df_trimmed = self.trimmed_dataframe(df, 'model', metric+'_avg')
                df_trimmed_mean = df_trimmed.groupby('model')[[col for col in df.columns if col.startswith(metric)]].mean().round(5)
                df_trimmed_std = df_trimmed.groupby('model')[[col for col in df.columns if col.startswith(metric)]].std().round(5)
                f.write('\n ---------------------------- {} average ----------------------------\n'.format(metric))
                f.write(df_trimmed_mean.to_string())
                f.write('\n ----------------------- {} standard deviation -----------------------\n'.format(metric))
                f.write(df_trimmed_std.to_string())  
        f.close()
        
    def create_melt_df(self, metric_name):    
        # col_list = [metric_name+'_E', metric_name+'_N', metric_name+'_W', metric_name+'_S']
        col_list = [metric_name+'_E', metric_name+'_N']
        # act_list = ['E', 'N', 'W', 'S']
        act_list = ['Action E', 'Action N']
        
        print(col_list)
        print(act_list)
        print({col: act for col, act in zip(col_list, act_list)})
        
        df_melt = self.df.melt(
            id_vars = 'model',
            value_vars = col_list,
            var_name = 'columns'
        )
        df_melt['columns'] = df_melt['columns'].replace({col: act for col, act in zip(col_list, act_list)})
        return df_melt
    
        
    def plot_cr_boxplot(self, mapping=None):
        df_melt = self.create_melt_df(metric_name='cr')

        if mapping is None:
            mapping = {}
            for key in df_melt['model'].unique():
                mapping[key] = key
        df_melt['model'] = df_melt['model'].replace(mapping)
        
        plt.figure(figsize=(15, 1 + self.model_num/3))
        ax = sns.boxplot(data=df_melt, hue='columns', x='value', y='model', orient = 'h',
                        order=mapping.values() ,
                            meanprops={"marker":"o",
                        "markerfacecolor":"white", 
                        "markeredgecolor":"black",
                        "markersize":"8"}
                    , boxprops={ "alpha": 0.3}
                    , showfliers=self.show_outliers
                    )
        plt.xlabel('Coverage rate', fontsize=14)
        plt.ylabel('Algorithm', fontsize=14)   
        # plt.title('Boxplot grouped by model')
        plt.tight_layout()     
        
        x_min, x_max = 0.0, 1.0
        x_ticks = np.arange(x_min, x_max, 0.05)
        x_ticks_major = np.arange(x_min, x_max+0.05, 0.1)
        
        ax.set_xticks(x_ticks, minor=True)
        ax.set_xticks(x_ticks_major, minor=False)
        ax.axvline(x=0.95, color='red', linestyle='--', label='95%')
        ax.xaxis.grid(True, which='major', linestyle='-', linewidth=1)
        ax.xaxis.grid(True, which='minor', linestyle='--', linewidth=0.5)
        # legend = plt.legend(fontsize=11, loc='lower right')
        legend = plt.legend(fontsize=11, loc='upper left')
        legend.get_title().set_text('') 
        plt.xticks(fontsize=11)  # Adjust the fontsize as needed for the x-axis
        plt.yticks(fontsize=11)
        # sns.set(font_scale=1.3)
        plt.savefig(os.path.join('boxplot','cr_boxplot.png'))
        print('cr boxplot saved')
        plt.close()
        
        
    def plot_start_cr_boxplot(self, mapping=None):
        df_melt = self.create_melt_df(metric_name='start_cr')

        if mapping is None:
            mapping = {}
            for key in df_melt['model'].unique():
                mapping[key] = key
        df_melt['model'] = df_melt['model'].replace(mapping)
        
        plt.figure(figsize=(15, 1 + self.model_num/3))
        ax = sns.boxplot(data=df_melt, hue='columns', x='value', y='model', orient = 'h',
                        order=mapping.values() ,
                            meanprops={"marker":"o",
                        "markerfacecolor":"white", 
                        "markeredgecolor":"black",
                        "markersize":"8"}
                    , boxprops={ "alpha": 0.3}, showfliers=self.show_outliers)
        plt.xlabel('Coverage rate at start point', fontsize=14)
        plt.ylabel('Algorithm', fontsize=14)   
        # plt.title('Boxplot grouped by model')
        plt.tight_layout()     
        
        x_min, x_max = 0.0, 1.0
        x_ticks = np.arange(x_min, x_max, 0.05)
        x_ticks_major = np.arange(x_min, x_max+0.05, 0.1)
        
        ax.set_xticks(x_ticks, minor=True)
        ax.set_xticks(x_ticks_major, minor=False)
        ax.axvline(x=0.95, color='red', linestyle='--', label='95%')
        ax.xaxis.grid(True, which='major', linestyle='-', linewidth=1)
        ax.xaxis.grid(True, which='minor', linestyle='--', linewidth=0.5)
        legend = plt.legend(fontsize=11, loc='upper left')
        legend.get_title().set_text('') 
        plt.xticks(fontsize=11)  # Adjust the fontsize as needed for the x-axis
        plt.yticks(fontsize=11)
        # sns.set(font_scale=1.3)
        plt.savefig(os.path.join('boxplot','start_cr_boxplot.png'))
        print('start_cr boxplot saved')
        plt.close()
    
    def plot_range_boxplot(self, mapping=None):
        df_melt = self.create_melt_df(metric_name='range')
        
        if mapping is None:
            mapping = {}
            for key in df_melt['model'].unique():
                mapping[key] = key
        df_melt['model'] = df_melt['model'].replace(mapping)
        
        plt.figure(figsize=(15, 1 + self.model_num/3))
        ax = sns.boxplot(data=df_melt, hue='columns', x='value', y='model', orient = 'h',
                        order=mapping.values() ,
                            meanprops={"marker":"o",
                        "markerfacecolor":"white", 
                        "markeredgecolor":"black",
                        "markersize":"8"}, 
                            boxprops={ "alpha": 0.3}, 
                            showfliers=self.show_outliers)
        plt.xlabel('Interval range', fontsize=14)
        plt.ylabel('Algorithm', fontsize=14)   
        # plt.title('Boxplot grouped by model')
        plt.tight_layout()     
        

        ax.xaxis.grid(True, which='major', linestyle='-', linewidth=1)
        ax.xaxis.grid(True, which='minor', linestyle='--', linewidth=0.5)
        legend = plt.legend(fontsize=11, loc='upper right')
        legend.get_title().set_text('') 
        plt.xticks(fontsize=11)  # Adjust the fontsize as needed for the x-axis
        plt.yticks(fontsize=11)
        # plt.xscale('log')
        # sns.set(font_scale=1.3)
        plt.savefig(os.path.join('boxplot','range_boxplot.png'))
        print('range boxplot saved')
        plt.close()
    
    def plot_mse_boxplot(self, mapping=None):
        df_melt = self.create_melt_df(metric_name='mse')
        
        if mapping is None:
            mapping = {}
            for key in df_melt['model'].unique():
                mapping[key] = key
        df_melt['model'] = df_melt['model'].replace(mapping)
        
        plt.figure(figsize=(15, 1 + self.model_num/3))
        ax = sns.boxplot(data=df_melt, hue='columns', x='value', y='model', orient = 'h', 
                        order=mapping.values() ,
                            meanprops={"marker":"o",
                        "markerfacecolor":"white", 
                        "markeredgecolor":"black",
                        "markersize":"8"}
                    , boxprops={ "alpha": 0.3}, showfliers=self.show_outliers)
        
        ax.xaxis.grid(True, which='major', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='x', which='minor', bottom=False)

        plt.xlabel(r'$MSE(\hat{Q})$', fontsize=14)
        plt.ylabel('Algorithm', fontsize=14)  
        plt.tight_layout()     
        plt.xticks(fontsize=11)  # Adjust the fontsize as needed for the x-axis
        plt.yticks(fontsize=11)
        plt.xscale('log')
        # sns.set(font_scale=1.3)
        
        legend = plt.legend(fontsize=11)
        legend.get_title().set_text('') 
        
        plt.savefig(os.path.join('boxplot','mse_boxplot.png'))
        print('mse boxplot saved')
        plt.close()
        
    def plot_start_mse_boxplot(self, mapping=None):
        df_melt = self.create_melt_df(metric_name='start_mse')
        
        if mapping is None:
            mapping = {}
            for key in df_melt['model'].unique():
                mapping[key] = key
        df_melt['model'] = df_melt['model'].replace(mapping)
        
        plt.figure(figsize=(15, 1 + self.model_num/3))
        ax = sns.boxplot(data=df_melt, hue='columns', x='value', y='model', orient = 'h', 
                        order=mapping.values() ,
                            meanprops={"marker":"o",
                        "markerfacecolor":"white", 
                        "markeredgecolor":"black",
                        "markersize":"8"}
                    , boxprops={ "alpha": 0.3}, showfliers=self.show_outliers )
        
        ax.xaxis.grid(True, which='major', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='x', which='minor', bottom=False)

        plt.xlabel(r'$MSE(\hat{Q})$', fontsize=14)
        plt.ylabel('Algorithm', fontsize=14)  
        plt.tight_layout()     
        plt.xticks(fontsize=11)  # Adjust the fontsize as needed for the x-axis
        plt.yticks(fontsize=11)
        plt.xscale('log')
        # sns.set(font_scale=1.3)
        
        legend = plt.legend(fontsize=11, loc='upper right')
        legend.get_title().set_text('') 
        
        plt.savefig(os.path.join('boxplot','start_mse_boxplot.png'))
        print('start_mse boxplot saved')
        plt.close()
        
def plot_cr_curves(model):
    df_list = []
    model_dir = os.path.join(output_dir, model)
    print(model)
    for exp in os.listdir(model_dir):
        results_path = os.path.join(model_dir, exp, 'progress.csv')
        if os.path.exists(results_path):
            df_temp = pd.read_csv(results_path)
            df_temp = df_temp[[col for col in df_temp.columns if col.startswith('metric')]+['rollout/timesteps']]
            df_temp.insert(loc=0, column='exp', value=int(exp.split('_')[-1]))
            df_temp.insert(loc=0, column='model', value=model)
        else:
            print(os.path.join(model_dir, exp))
            print('no result')
            continue
        # print(df_temp)
        try:
            df_list.append(df_temp)
        except:
            print(exp)
            
    df = pd.concat(df_list).reset_index(drop=True)
    df.columns = [col.split('/')[-1] for col in df.columns]
    df = df.dropna()
    df_melted = df.melt(id_vars=['timesteps', 'exp'], value_vars=['cr_E', 'cr_N'], 
                    var_name='Variable', value_name='Value')
    sns.lineplot(x='timesteps', y='Value', hue='Variable', data=df_melted, estimator='median', errorbar='sd')
    plt.axhline(y=0.95, color='red', linestyle='--', label='95%')
    plt.ylim(0.0, 1.0)
    plt.title('Mean Curves of Variables over Time')
    plt.savefig(os.path.join('cr_plot','{}.png'.format(model)))
    plt.close()
    
    

if __name__ == '__main__':

    name_list = [f for f in os.listdir(output_dir) if not f.startswith('.') and os.path.isdir(os.path.join(output_dir, f))]
    name_list = sorted(name_list)
    name_mapping = None
    
    name_mapping = {       
                    'lktd_pp2500': r'LKTD: $\mathcal{N}=2500$',
                    'lktd_pp5000': r'LKTD: $\mathcal{N}=5000$',
                    'lktd_pp10000': r'LKTD: $\mathcal{N}=10000$',
                    
                    
                    'sgld_pp2500': r'SGLD: $\mathcal{N}_t=2500$',
                    'sgld_pp5000': r'SGLD: $\mathcal{N}_t=5000$',
                    'sgld_pp10000': r'SGLD: $\mathcal{N}_t=10000$',     
                              
                    'sghmc_pp2500': r'SGHMC: $\mathcal{N}_t=2500$',
                    'sghmc_pp5000': r'SGHMC: $\mathcal{N}_t=5000$',
                    'sghmc_pp10000': r'SGHMC: $\mathcal{N}_t=10000$',
                    
                    
                    'dqn': r'DQN',
                    # 'boot_dqn_bp05': r'BootDQN: head=5',
                    'boot_dqn_bp05_n10': r'BootDQN',  
                    # 'qrdqn_lr1e-2': r'QRDQN: nq15',
                    # 'qrdqn_lr1e-2_nq30': r'QRDQN: nq30',
                    # 'qrdqn_lr1e-2_nq50': r'QR-DQN',
                    'qrdqn_nq10': r'QR-DQN',

                    'kova': r'KOVA',
                    # 'kova_short': r'KOVA',

    }
    

    boxplot_list = name_list
    
    name_list = [
        'lktd_pp2500',
        'lktd_pp5000',
        'lktd_pp10000',
        
        'sgld_pp2500',
        'sgld_pp5000',
        'sgld_pp10000',
        
        'sghmc_pp2500',
        'sghmc_pp5000',
        'sghmc_pp10000',
        

        'dqn',
        # 'boot_dqn_bp05',
        'boot_dqn_bp05_n10',
        # 'qrdqn_lr1e-2',
        # 'qrdqn_lr1e-2_nq30',
        # 'qrdqn_lr1e-2_nq50',
        'qrdqn_nq10',
        
        'kova',
        # 'kova_short',
    ]
    
    # boxplot_list = [f for f in name_list if f.split('_')[0] in ['LKTD', 'SGLD']]
    
    print(name_list)
    
    visualization = Visualization(name_list, 5, False)
    # print(visualization.df)
    # # print(visualization.df_exp_mean)
    visualization.print_metrics()
    
    visualization.plot_cr_boxplot(name_mapping)
    visualization.plot_start_cr_boxplot(name_mapping)
    visualization.plot_mse_boxplot(name_mapping)
    visualization.plot_start_mse_boxplot(name_mapping)
    visualization.plot_range_boxplot(name_mapping)
    
    
    # Specify the directory
    folder = os.path.join(cur_dir, 'cr_plot')


    # Iterate over each item in the directory
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            # Check if it is a file or a directory
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    for model in name_list:
        plot_cr_curves(model)