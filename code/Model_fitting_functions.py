import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from Models import *
from data_wrangling import *
from itertools import chain
from PyPDF2 import PdfMerger
import inspect
import time
from io import StringIO
import sys
def coef_dict_to_list(coef_dict):
    return list(coef_dict.values())
#the single mutant to be studied
   # function to input paramaters as a list

def get_data_SM(mutation:str):
        df_MT = df
        data = meta_dict["SM"]
        data = data.loc[data['Mutant_ID'] == mutation]
        #WT data missing measurements for inducer = 0.2 so drop last columnn
        data = data[:-1]

        a = re.search("[0-9]",mutation).start()
        mut_col = f"{mutation[:a]}"
        mutation_means = f"{mutation[:a]}_mean"
        df_MT[[mut_col, "Stripe"]] = data[[mutation_means, "Stripe_mean"]].values
        df_MT[["Mutant_ID"]]=mutation
        if mutation.startswith("Output"):
            df_MT["Sensor"]=meta_dict["WT"].Sensor

        return df_MT
#data=pd.read_csv('../data/WT_single.csv')

    #now gonna do the scipy.optimize.minimize

def dict_to_list(params_dict,return_keys=False):
    if return_keys==True:
       a=[list(i.keys()) for i in list(params_dict.values())]
    elif return_keys==False:
        a=[list(i.values()) for i in list(params_dict.values())]

    return list(chain.from_iterable(a))

def dict_to_list2(params_dict,return_keys=False):
    if return_keys==True:
       a=[list(i.keys()) for i in list(params_dict.values())]
    elif return_keys==False:
        a=[list(i) for i in list(params_dict.values())]

    return list(chain.from_iterable(a))

def coef_dict_to_list(coef_dict):
    return list(coef_dict.values())
#the single mutant to be studied
   # function to input paramaters as a list

def get_data_SM(mutation:str):
        df = pd.read_csv('../data/WT_single.csv')
        df_MT = df
        data = meta_dict["SM"]
        data = data.loc[data['Mutant_ID'] == mutation]
        #WT data missing measurements for inducer = 0.2 so drop last columnn
        data = data[:-1]

        a = re.search("[0-9]",mutation).start()
        mut_col = f"{mutation[:a]}"
        mutation_means = f"{mutation[:a]}_mean"
        df_MT[[mut_col, "Stripe"]] = data[[mutation_means, "Stripe_mean"]].values
        df_MT[["Mutant_ID"]]=mutation
        if mutation.startswith("Output"):
            df_MT["Sensor"]=meta_dict["WT"].Sensor

        return df_MT
    #now gonna do the scipy.optimize.minimize
def WT_fit_plot(ax,x, y,params,label:str,data):
        return ax.plot(data[x], y, **params,label=f"{label}")

def WT_fit_plot2(ax,x, y,params,label:str,data, a):
        return ax.plot(data[x], y, **params,label=f"{label}", alpha= a)


    #define scatter plotting function with log scales
def WT_Plotter(ax,x,y, params,data):
    out = ax.scatter(data[x], data[y], **params, marker = 'o')
    xScale = ax.set_xscale('log')
    yScale = ax.set_yscale('log')

    return out, xScale, yScale 

def list_to_dict(old_dict:dict,new_values:list):
    #first check that lengths of lists are compatible
    if len(new_values)==len(dict_to_list(old_dict)):
        i=0
        for key in list(old_dict.keys()):
            for subkey in list(old_dict[key].keys()):
                old_dict[key][subkey]=new_values[i]
                i+=1
    new_dict=old_dict.copy()
    return new_dict
#%%
import seaborn as sns
import re
def Out_to_DF_hill(filepath, model_type, mut_name, all:bool):
    """Converts final outfiles of parameters from ABC_SMC to a dataframe"""
    df_list = []
    with open(filepath) as f:
        for line in f:
            # remove whitespace at the start and the newline at the end
            line = line.strip()
            # split each column on whitespace
            columns = re.split('\s+', line, maxsplit=26)
            cols = [float(x) for x in columns]
            df_list.append(cols)
    dataframe = pd.DataFrame(df_list)
    dataframe.reset_index(drop=True)
    dataframe.index = np.arange(1,len(dataframe)+1)

    if all == False:

        if model_type == model_hill.model_new_WT:
            dataframe.rename(columns={0:'As',1:'Bs',2:'Cs',3:'Ns',4:'Ar',5:'Br',6:'Cr',7:'Nr', 8:'Ao',9:'Bo',10:'Co',11:'Ck',12:'No',13:'Fo'}, inplace=True)
        elif model_type == model_hill.model_muts:
                if mut_name.startswith("Sensor"):
                    dataframe.rename(columns={0:'MAs',1:'MBs',2:'MCs',3:'MNs'}, inplace=True)
                if mut_name.startswith("Regulator"):
                    dataframe.rename(columns={0:'MAr',1:'MBr',2:'MCr',3:'MNr'}, inplace=True)
                if mut_name.startswith("Output"):
                    dataframe.rename(columns={0:'MAo',1:'MBo',2:'MCo',3:'MNo'}, inplace=True)

    else:
        if model_type == model_hill.model_new_WT:
            dataframe.rename(columns={0:'As',1:'Bs',2:'Cs',3:'Ns',4:'Ar',5:'Br',6:'Cr',7:'Nr', 8:'Ao',9:'Bo',10:'Co',11:'Ck',12:'No',13:'Fo'}, inplace=True)
        elif model_type == model_hill.model_muts:
            dataframe.rename(columns={0:'As',1:'Bs',2:'Cs',3:'Ns',4:'MAs',5:'MBs',6:'MCs',7:'MNs',8:'Ar',9:'Br',10:'Cr',11:'Nr',12:'MAr',13:'MBr',14:'MCr',15:'MNr', 16:'Ao',17:'Bo',18:'Co',19: 'Ck', 20:'No',21:'Fo', 22:'MAo',23:'MBo',24:'MCo',25:'MNo'}, inplace=True)
    return dataframe


# %%
from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd

def multivariate_dis(df):
    '''input a dataframe of parameters(cols) and values(rows)'''
    #convert df into matrix
    names = df.keys()
    params = len(df.columns)
    matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        matrix[i] = df[name].to_numpy()
        i = i+1

    #range of parameters as x, means calculated
    mean_list = []
    ranges = np.empty(shape=(params,10), dtype=float)
    j = 0
    
    for m in matrix:
        means = sum(m)
        means = means/len(m)
        mean_list.append(means)
        mini = min(m)
        maxi = max(m)
        temp = np.linspace(mini,maxi,10)
        ranges[j] = temp
        j = j+1
    #generate cov matrix
    cov_matrix = np.cov(matrix, bias = True)
    #generate multivariate normal distribution
    multi_norm_dis = multivariate_normal(
                        mean = mean_list,
                        cov = cov_matrix,
                        allow_singular = True)
    return multi_norm_dis

#%%
#plotting functions
from typing import Any, Dict, List, Optional, Union
def Paired_Density_plot(dataframe, name, save:bool):
    '''Plots joint distribution of parameters from ABC_SMC'''
    sns.set_theme(style="white")

    df = dataframe
    col_names = list(df.columns.values)
        
    g = sns.pairplot(df, vars = col_names, kind = "kde", corner=True)
    #sns.axes[0,0].set_xlim((0,0))
    parlist: List[Dict[str, Union[str, float]]] = {0:{
    'name': 'log_A_s',
    'lower_limit': 2.0,
    'upper_limit': 4.0
    }, 1: {
    'name': 'log_B_s',
    'num': 1,
    'lower_limit': 2.0,
    'upper_limit': 5.0
    }, 2: {
    'name': 'log_C_s',
    'lower_limit': 2.0,
    'upper_limit': 4.0
    }, 3: {
    'name': 'N_s',
    'lower_limit': 1.0,
    'upper_limit': 4.0
    }, 4: {
    'name': 'log_A_r',
    'lower_limit': 2.0,
    'upper_limit': 4.0
    }, 5: {
    'name': 'log_B_r',
    'lower_limit': 2.0,
    'upper_limit': 4.0
    }, 6: {
    'name': 'log_C_r',
    'lower_limit': -4.0,
    'upper_limit': -1.0
    }, 7: {
    'name': 'N_r',
    'lower_limit': 1.0,
    'upper_limit': 4.0
    }, 8: {
    'name': 'log_A_o',
    'lower_limit': 2.0,
    'upper_limit': 4.0
    }, 9: {
    'name': 'log_B_o',
    'lower_limit': 4.0,
    'upper_limit': 8.0
    }, 10: {
    'name': 'log_C_o',
    'lower_limit': -3.0,
    'upper_limit': 0.0
    }, 11: {
    'name': 'N_o',
    'lower_limit': 1.0,
    'upper_limit': 4.0
    }, 12: {
    'name': 'F_o',
    'lower_limit': 0.0,
    'upper_limit': 2.0
    } } 

    #uses the priors from the parameters to act as the range
    g.fig.suptitle(f"Parameter distribution of ({name})")

    count = -1

    for j in range(0,13):
        count = count + 1
        for i in range(count,13):
            g.axes[i,j].set_xlim((parlist[j]['lower_limit'], parlist[j]['upper_limit']))
            g.axes[i,j].set_ylim((parlist[i]['lower_limit'], parlist[i]['upper_limit']))


    if save == True:
        plt.savefig(f'../results/{name}_Hill_Paired_Density.pdf', format="pdf", bbox_inches="tight")
    elif save == False:
        print('plot not saved btw')
    else:
        print('Unclear whether to save plot or not')
    return 

def Paired_Density_plot_compare(dataframe, name, huw, save:bool):
    '''Plots joint distribution of parameters from ABC_SMC'''
    sns.set_theme(style="white")

    df = dataframe
    col_names = list(df.columns)
    new_cols = []
    for i in range(0,len(col_names)-1):
        new_cols.append(col_names[i])

    g = sns.pairplot(df, vars = new_cols, kind = "kde", hue = huw, corner=True)
    #sns.axes[0,0].set_xlim((0,0))
    # parlist: List[Dict[str, Union[str, float]]] = {0:{
    # 'name': 'log_A_s',
    # 'lower_limit': 2.0,
    # 'upper_limit': 4.0
    # }, 1: {
    # 'name': 'log_B_s',
    # 'num': 1,
    # 'lower_limit': 2.0,
    # 'upper_limit': 5.0
    # }, 2: {
    # 'name': 'log_C_s',
    # 'lower_limit': 2.0,
    # 'upper_limit': 4.0
    # }, 3: {
    # 'name': 'N_s',
    # 'lower_limit': 1.0,
    # 'upper_limit': 4.0
    # }, 4: {
    # 'name': 'log_A_r',
    # 'lower_limit': 2.0,
    # 'upper_limit': 4.0
    # }, 5: {
    # 'name': 'log_B_r',
    # 'lower_limit': 2.0,
    # 'upper_limit': 4.0
    # }, 6: {
    # 'name': 'log_C_r',
    # 'lower_limit': -4.0,
    # 'upper_limit': -1.0
    # }, 7: {
    # 'name': 'N_r',
    # 'lower_limit': 1.0,
    # 'upper_limit': 4.0
    # }, 8: {
    # 'name': 'log_A_o',
    # 'lower_limit': 2.0,
    # 'upper_limit': 4.0
    # }, 9: {
    # 'name': 'log_B_o',
    # 'lower_limit': 4.0,
    # 'upper_limit': 8.0
    # }, 10: {
    # 'name': 'log_C_o',
    # 'lower_limit': -3.0,
    # 'upper_limit': 0.0
    # }, 11: {
    # 'name': 'N_o',
    # 'lower_limit': 1.0,
    # 'upper_limit': 4.0
    # }, 12: {
    # 'name': 'F_o',
    # 'lower_limit': 0.0,
    # 'upper_limit': 2.0
    # } } 

    # #uses the priors from the parameters to act as the range
    g.fig.suptitle(f"Parameter distribution of ({name})")

    # count = -1

    # for j in range(0,13):
    #     count = count + 1
    #     for i in range(count,13):
    #         g.axes[i,j].set_xlim((parlist[j]['lower_limit'], parlist[j]['upper_limit']))
    #         g.axes[i,j].set_ylim((parlist[i]['lower_limit'], parlist[i]['upper_limit']))


    if save == True:
        plt.savefig(f'../results/{name}_Hill_Paired_Density.pdf', format="pdf", bbox_inches="tight")
    elif save == False:
        print('plot not saved btw')
    else:
        print('Unclear whether to save plot or not')
    return 

def Paired_Density_plot_mut(dataframe, name, save:bool):
    '''Plots joint distribution of parameters from ABC_SMC'''
    sns.set_theme(style="white")

    df = dataframe
    
    col_names = list(df.columns.values)
        
    g = sns.pairplot(df, vars = col_names, kind = "kde", corner=True)

    # g = sns.pairplot(df, kind = "kde", corner=True)

    #g.setp(axes, xlim=custom_xlim, ylim=custom_ylim)
    
    parlist: List[Dict[str, Union[str, float]]] = {0:{
    'name': 'log_MA',
    'lower_limit': -2.0,
    'upper_limit': 2.0
    }, 1: {
    'name': 'log_MB',
    'lower_limit': -2.0,
    'upper_limit': 2.0
    }, 2: {
    'name': 'log_MC',
    'lower_limit': -1.0,
    'upper_limit': 1.0
    }, 3: {
    'name': 'log_MN',
    'lower_limit': -1.0,
    'upper_limit': 1.0
    }} 
    
    
    count = -1

    for j in range(0,4):
        count = count + 1
        for i in range(count,4):
            g.axes[i,j].set_xlim((parlist[j]['lower_limit'], parlist[j]['upper_limit']))
            g.axes[i,j].set_ylim((parlist[i]['lower_limit'], parlist[i]['upper_limit']))
    
    
    g.fig.suptitle(f"Parameter distribution of Mutant ({name})")



    if save == True:
        plt.savefig(f'../results/{name}_Hill_Paired_Density.pdf', format="pdf", bbox_inches="tight")
    elif save == False:
        print('plot not saved btw')
    else:
        print('Unclear whether to save plot or not')
    # h = sns.pairplot(df, kind = "hist", corner=True)

    return 