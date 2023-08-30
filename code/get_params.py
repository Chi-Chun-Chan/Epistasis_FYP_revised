import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from Models import *
from data_wrangling import *
from Model_fitting_functions import *
from random import seed
import time
from RSS_Scoring import *
import warnings
from Epistasis_calc_functions import *
warnings.simplefilter(action='ignore', category=FutureWarning)
import progressbar as pb
#%%
'''Get predicted pairwise parameters'''

def get_sm_params(mutants:list): 
    '''Obtain single mutant parameters from SMC and sort them into relevant dataframes'''
    mut1 = mutants[0]
    mut2 = mutants[1]
    mut_id = get_mut_ids(mutants)
    path = f'../data/smc_hill_new/{mut1}_smc/all_pars_final.out'
    path2 = f'../data/smc_hill_new/{mut2}_smc/all_pars_final.out'
    df1 = Out_to_DF_hill(path, model_hill.model_muts, mut1, all=True)
    df2 = Out_to_DF_hill(path2, model_hill.model_muts, mut2, all=True)
    WT1_df = df1[['As','Bs','Cs','Ns','Ar','Br','Cr','Nr','Ao','Bo','Co','Ck','No','Fo']]
    WT1_df = np.log10(WT1_df) #convert to log10
    WT1_df.reset_index(drop=True, inplace=True)
    WT2_df = df2[['As','Bs','Cs','Ns','Ar','Br','Cr','Nr','Ao','Bo','Co','Ck','No','Fo']]
    WT2_df = np.log10(WT2_df) #convert to log10
    WT2_df.reset_index(drop=True, inplace=True)

    letters = r'[a-zA-Z]'

    # for file in listdir(folder):

    # Extract from file title, which is the duplet
    mutant_letters = re.findall(letters, mut_id)
    m1_str = mutant_letters[0].lower() # first character of filename string
    m2_str = mutant_letters[1].lower()

    M1_mods_df = df1[[f'MA{m1_str}',f'MB{m1_str}',f'MC{m1_str}',f'MN{m1_str}']]
    M2_mods_df = df2[[f'MA{m2_str}',f'MB{m2_str}',f'MC{m2_str}',f'MN{m2_str}']]
    M1_mods_df = np.log10(M1_mods_df)
    M2_mods_df = np.log10(M2_mods_df)

    # mod_path = f'../data/smc_SM_hill/{mut1}_smc/pars_final.out' 
    # M1_mods_df = Out_to_DF_hill(mod_path, model_hill.model_muts, mut1, all=False)
    M1_mods_df.reset_index(drop=True, inplace=True)
    # mod_path2 = f'../data/smc_SM_hill/{mut2}_smc/pars_final.out' 
    # M2_mods_df = Out_to_DF_hill(mod_path2, model_hill.model_muts, mut2, all=False)
    M2_mods_df.reset_index(drop=True, inplace=True)

    M1_df = pd.concat([M1_mods_df,WT1_df], axis=1)
    M2_df = pd.concat([M2_mods_df, WT2_df,], axis=1)
    Combined_WT = pd.concat([WT1_df,WT2_df], axis=0)

    return Combined_WT, M1_df, M2_df, M1_mods_df, M2_mods_df #returns all parameters in log10 form

def get_dm_params(mutants:list):
    '''Run conditional selection of parameters for a set of single mutants in the form ['Regulator10','Output10']'''
    data = meta_dict['WT']
    combined_wt, m1_wt, m2_wt, m1, m2 = get_sm_params(mutants) #extracts the fitted parameters and corresponding WT params
    names = combined_wt.keys()
    params = len(combined_wt.columns)
    WT_matrix = np.empty(shape=(params,2000), dtype=float)
    i = 0
    for name in names:
        WT_matrix[i] = combined_wt[name].to_numpy() #takes list of single params and appends to array.
        i = i+1

    WT_mean_list = []
    j = 0

    for m in WT_matrix: #calculates the mean of each logged parameter of WT
        means = sum(m)
        means = means/len(m)
        WT_mean_list.append(means)
        j = j+1
    #generate cov matrix
    combined_wt = combined_wt.T
    WT_cov_matrix = np.cov(combined_wt.values) #covariance of each params 14x14 matrix
    #generate multivariate normal distribution
    WT_multi_norm_dis = multivariate_normal(
                        mean = WT_mean_list,
                        cov = WT_cov_matrix,
                        allow_singular = True)
    #accurate = False
    # while accurate == False:
    #     rndint = np.random.randint(low=0, high=1e7)
    #     timeseed = time.time_ns() % 2**16
    #     np.random.seed(rndint+timeseed)
    #     seed(rndint+timeseed)
    #     WT_sample = WT_multi_norm_dis.rvs(size=1, random_state=rndint+timeseed) #generates one wildtype sample from the shared distribution
    #     distance = RSS_Score(param_list= 10**WT_sample, model_type=model_hill, data_=data, model_specs='new_WT')
    #     if distance <= 0.08: #only select good WT params
    #         accurate = True
    rndint = np.random.randint(low=0, high=1e7) #generates random timeseed
    timeseed = time.time_ns() % 2**16
    np.random.seed(rndint+timeseed)
    seed(rndint+timeseed)
    WT_sample = WT_multi_norm_dis.rvs(size=1, random_state=rndint+timeseed) #selects one WT parameter sampled from the combined distribution of params
    #############################################################
    #Sample from modifier parameters (conditional sampling with fixed WT_sample)
    names = m1.keys() #Name of all mutant parameters
    params = len(m1.columns)
    M1_mods_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M1_mods_matrix[i] = m1[name].to_numpy() #convert each modifier into a row in a matrix 
        i = i+1
    
    M1_mean_list = []
    j = 0

    for m in M1_mods_matrix: #calculate the mean of each modifier parameter
        means = sum(m)
        means = means/len(m)
        M1_mean_list.append(means)
        j = j+1

    #get wildtypes for m1
    WT1 = m1_wt.iloc[:,4:] #Calculate the mean for the m1 wildtypes for conditional distribution

    WT1_names = WT1.keys() 
    WT1_params = len(WT1.columns)
    WT1_matrix = np.empty(shape=(WT1_params,1000), dtype=float)
    i = 0
    for name in WT1_names:
        WT1_matrix[i] = WT1[name].to_numpy()
        i = i+1
    WT1_mean_list = []
    j = 0
    for m in WT1_matrix:
        means = sum(m)
        means = means/len(m)
        WT1_mean_list.append(means)
        j = j+1

    #Generate covariance matrix of full mutant params
    names = m1_wt.keys()
    params = len(m1_wt.columns)
    M1_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M1_matrix[i] = m1_wt[name].to_numpy()
        i = i+1
    M1_cov_matrix = np.cov(M1_matrix, bias = True)
    mu1 = M1_mean_list
    mu2 = WT1_mean_list
    C11 = M1_cov_matrix[0:4,0:4]
    C12 = M1_cov_matrix[0:4:,4:]
    C21 = M1_cov_matrix[4:,0:4]
    C22 = M1_cov_matrix[4:,4:]
    C22inv = np.linalg.inv(C22)
    a_minus_mu = (WT_sample - mu2)
    a_minus_mu[:, np.newaxis]
    C12C22inv = np.dot(C12,C22inv.T) 
    temp = np.dot(C12C22inv, a_minus_mu[:, np.newaxis])
    conditional_mu = [x+y for x, y in zip(mu1,temp.flatten().tolist())] 

    conditional_cov = C11 - np.dot(C12C22inv, C21)

    M1_multi_dis = multivariate_normal(mean = conditional_mu,
                                        cov = conditional_cov, 
                                        allow_singular = True
                                                 )
    
    M1_cond_params = M1_multi_dis.rvs(size = 100, random_state=rndint+ timeseed) #sample 100 m1 modifiers from conditional dist.
    M1s = pd.DataFrame(M1_cond_params, columns = m1.keys()) #convert to dataframe with column names as the m1 names
    ######################
    #mut2
    names = m2.keys()
    params = len(m2.columns)
    M2_mods_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M2_mods_matrix[i] = m2[name].to_numpy()
        i = i+1
    
    M2_mean_list = []
    j = 0

    for m in M2_mods_matrix:
        means = sum(m)
        means = means/len(m)
        M2_mean_list.append(means)
        j = j+1

    WT2 = m2_wt.iloc[:,4:]

    WT2_names = WT2.keys() 
    WT2_params = len(WT2.columns)
    WT2_matrix = np.empty(shape=(WT2_params,1000), dtype=float)
    i = 0
    for name in WT2_names:
        WT2_matrix[i] = WT2[name].to_numpy()
        i = i+1
    WT2_mean_list = []
    j = 0
    for m in WT2_matrix:
        means = sum(m)
        means = means/len(m)
        WT2_mean_list.append(means)
        j = j+1

    #Generate covariance matrix of full mutant params
    names = m2_wt.keys()
    params = len(m2_wt.columns)
    M2_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M2_matrix[i] = m2_wt[name].to_numpy()
        i = i+1
    M2_cov_matrix = np.cov(M2_matrix, bias = True)
    mu1 = M2_mean_list
    mu2 = WT2_mean_list
    C11 = M2_cov_matrix[0:4,0:4]
    C12 = M2_cov_matrix[0:4:,4:]
    C21 = M2_cov_matrix[4:,0:4]
    C22 = M2_cov_matrix[4:,4:]
    C22inv = np.linalg.inv(C22)
    a_minus_mu = (WT_sample - mu2)
    a_minus_mu[:, np.newaxis]
    C12C22inv = np.dot(C12,C22inv.T) 
    temp = np.dot(C12C22inv, a_minus_mu[:, np.newaxis])
    conditional_mu = [x+y for x, y in zip(mu1,temp.flatten().tolist())]

    conditional_cov = C11 - np.dot(C12C22inv, C21)

    M2_multi_dis = multivariate_normal(mean = conditional_mu,
                                       cov = conditional_cov,
                                       allow_singular = True
                                                 )
    
    M2_cond_params = M2_multi_dis.rvs(size = 100, random_state=rndint+ timeseed)
    M2s = pd.DataFrame(M2_cond_params, columns = m2.keys())
    # 

    mods_df = pd.DataFrame({"MAs":[], #dataframe for modifier parameters
                        "MBs":[],
                        "MCs":[],
                        "MNs":[],
                        "MAr":[],
                        "MBr":[],
                        "MCr":[],
                        "MNr":[],
                        "MAo":[],
                        "MBo":[],
                        "MCo":[],
                        "MNo":[]})
    
    WT_df = pd.DataFrame(WT_sample).transpose()
    WT_df.columns = ['As','Bs','Cs','Ns','Ar','Br','Cr','Nr','Ao','Bo','Co','Ck','No','Fo']

    mods_df[m1.keys()[0]] = M1s[m1.keys()[0]] #append the correct modifier columns to the mods dataframe
    mods_df[m1.keys()[1]] = M1s[m1.keys()[1]]
    mods_df[m1.keys()[2]] = M1s[m1.keys()[2]]
    mods_df[m1.keys()[3]] = M1s[m1.keys()[3]]
    mods_df[m2.keys()[0]] = M2s[m2.keys()[0]]
    mods_df[m2.keys()[1]] = M2s[m2.keys()[1]]
    mods_df[m2.keys()[2]] = M2s[m2.keys()[2]]
    mods_df[m2.keys()[3]] = M2s[m2.keys()[3]]

    

    mods_df = mods_df.replace(np.nan,0) #replace modifier params of wt node as 0.0 so it becomes 1 when un-logged

    WT_df = WT_df.append([WT_df]*(len(M1s[m1.keys()[0]])-1),ignore_index=True) #make wt_df the same length as mods_df

    log10_pars = pd.concat([WT_df,mods_df], axis=1) #returns full matrix with all WT and modifiers from this round of selection.
      
    return log10_pars
#%%
'''Save all the parameters for all pairwise mutants''' #uncomment and only run if necessary
# prior_mutant = None
# size = 100
# DM_names = DM_stripes['genotype'].tolist()
# DM_names = list(set(DM_names[3:]))
# DM_names.sort()

# if prior_mutant == None: #incase code breaks, put in the genotype of last succesful run e.g R10_S10
#         mutant_range:slice=slice(0,len(DM_names))
#         count = 0
# else:
#     position = DM_names.index(prior_mutant)
#     count = position
#     mutant_range:slice=slice(position+1,len(DM_names)) #runs the next mutant from the previous one if stated

# for genotypes in DM_names[mutant_range]:
#     bar = pb.ProgressBar(maxval = size).start() #import time
#     #get genotypeID
#     mutant_list = get_mut_names(genotypes)
#     log_params = get_dm_params(mutant_list) #100 parameter sets, one wt sample.
#     for i in range(0,size):
#         temp = get_dm_params(mutant_list)
#         bar.update(i+1)
#         log_params = pd.concat([log_params,temp], ignore_index=True)

#     path = f'../results/New_params/Pairwise_params/{genotypes}.csv' #creates dataframe of parameters in the designated path
#     count = count + 1
#     log_params.to_csv(path, index = False)
#     print('mutant ', genotypes, 'completed, ', count, 'out of 300')

#%%

