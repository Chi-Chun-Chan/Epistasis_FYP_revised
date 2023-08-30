'''Epistasis analysis conducted for each pairwise mutant and all pairwise mutants.'''
#%%
import numpy as np 
import seaborn as sns
import sympy as sym
from os import listdir
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy import stats
from Visualise_fits import *
from Epistasis_calc_functions import*
#%%
## Initizalization of dummy variables (parameters of the model) for sympy
symbolnames = ['aR','aS','aO','cS','cR','cO','bR','bS','bO','nR','nS','nO','I']
basepars = ['A','B','C','N'] # nbame of the pars for each mutant
myVars = vars()
for symbolname in symbolnames:
    myVars[symbolname] = sym.symbols(symbolname)
sym.init_printing(use_unicode=True)

def stst(pars):
    ''' 
    returns steady state values of the ODE given a parameter set
    for the three nodes S,R,O. It also returns Oh which is the
    expected output fluorescence in the absence of regulator
    '''
    affS = np.power(pars['Cs']*pars['I'],pars['Ns'])
    Seq = (pars['As'] + pars['Bs']*affS)/(1+affS)

    affR = np.power(pars['Cr']*Seq,pars['Nr'])
    Req = pars['Ar'] + pars['Br']/(1+affR)

    affO = np.power(pars['Co']*(Req+pars['Ck']*Seq),pars['No'])
    Oeq =  (pars['Ao'] + pars['Bo']/(1+affO))*pars['Fo'] #multiply by F_o

    affOh = np.power(pars['Co']*Req,pars['No']) #halfoutput is only sensor
    Oheq = pars['Ao'] + pars['Bo']/(1+affOh)

    return Seq,Req,Oeq,Oheq

#%%
'''Visualise the distribution of fits for pairwise mutant'''
def Epistasis_analysis(mut_list, file):
    Visualise_SM_fit(mut_name=mut_list[0],iter = 'final', plot_num= 50, save=False) #plot each single mutant
    Visualise_SM_fit(mut_name=mut_list[1],iter = 'final', plot_num= 50, save=False)
    Visualise_mut(mutants = mut_list) #plot the predicted fluorescence of pairwise mutant
    depth = 10000 # MAX = 250000, number of parameter sets to use per file
    folder = '../results/New_params/Pairwise_params/'
    epi_model = pd.DataFrame()
    mode_epi = pd.DataFrame()

    letters = r'[a-zA-Z]'
    genotype = get_mut_ids(mut_list)
    DM_df = meta_dict['DM']
    pair_mut_dict = DM_df[DM_df['genotype'] == genotype]
    mut1_data = get_data_SM(mut_list[0])
    mut2_data = get_data_SM(mut_list[1])

    # for file in listdir(folder):

    # Extract from file title, which is the duplet
    mutant_letters = re.findall(letters, file)
    m1_str = mutant_letters[0].lower() # first character of filename string
    m2_str = mutant_letters[1].lower() # first character of filename string
    print('mutant_combo: {} with mutants {} and {}'.format(file,m1_str,m2_str))

    # Calulate WT fluorescence
    data_WT = pd.read_csv(folder+file)
    data_WT = data_WT.head(depth) # if depth<10000 then only a subset of parameters is loaded
    data_WT = 10**data_WT #convert from log10
    data_WT['I'] = 0.0002 # WT peak position
    WT_data = meta_dict['WT']
    WT_fluo = stst(data_WT)[2]
    plt.hist(np.log10(WT_fluo), bins='auto')
    plt.axvline(x=np.log10(WT_data.iloc[5].Stripe), c = 'red', label = 'WT data') #plot stripe medium inducer conc
    plt.xlabel('log10(fluorescence)')
    plt.ylabel('density')
    plt.title('m1_fluo')
    plt.legend()
    plt.show()

    # Creating dataframes for singlets and duplet, in these dataframes the paremets will be modified using the fitted modifiers
    data_mut1 = data_WT.copy()
    data_duplet = data_WT.copy()
    data_mut2 = data_WT.copy()

    # Mutant 1 and duplet part
    for par in basepars: # for each parameter of a node
        data_mut1[par+m1_str] = data_mut1[par+m1_str]*data_mut1['M'+par+m1_str]
        data_duplet[par+m1_str] = data_duplet[par+m1_str]*data_duplet['M'+par+m1_str]
    # if m1_str == 'o': # in the case that 1 of the mutatns is the outout, apply fluorescence correction. Note that this should not affect epistasis
    #     data_mut1['Fo'] = data_mut1['F_o']
    #     data_duplet['Fo'] = data_duplet['F_o']

    m1_fluo = stst(data_mut1)[2]
    plt.hist(np.log10(m1_fluo), bins='auto')
    plt.axvline(x=np.log10(mut1_data.iloc[5].Stripe), c = 'red', label = 'SM data') #plot stripe medium inducer conc
    plt.xlabel('log10(fluorescence)')
    plt.ylabel('density')
    plt.title('m1_fluo')
    plt.legend()
    plt.show()


    # Mutant 2 and duplet part
    for par in basepars:
        data_mut2[par+m2_str] = data_mut2[par+m2_str]*data_mut2['M'+par+m2_str]
        data_duplet[par+m2_str] = data_duplet[par+m2_str]*data_duplet['M'+par+m2_str]
    # if m2_str == 'o':
    #     data_mut2['F_o'] = data_mut2['F_o']*data_mut2['MF_o']
    #     data_duplet['F_o'] = data_duplet['F_o']*data_duplet['MF_o']

    exp_duplet_fluo = pair_mut_dict['obs_fluo_mean']
    m2_fluo = stst(data_mut2)[2] 
    plt.hist(np.log10(m2_fluo), bins='auto')
    plt.axvline(x=np.log10(mut1_data.iloc[5].Stripe), c = 'red',  label = 'SM data')
    plt.title('m2_fluo')
    plt.xlabel('log10(fluorescence)')
    plt.ylabel('density')
    plt.legend()
    plt.show()
    # Duplet
    duplet_fluo = stst(data_duplet)[2]
    #Method1: take the median predicted value and calculate the epistasis from there.
    #   Maybe take only the 5 percent of fluorescent values around the mean and calculate epistasis as those are the more 'accurate' simulations.
    plt.hist(np.log10(duplet_fluo), bins='auto')
    plt.title('duplex_fluo')
    plt.xlabel('log10(fluorescence)')
    plt.ylabel('density')
    plt.axvline(x = np.log10(exp_duplet_fluo.iloc[1]),c = 'r', label = 'Pairwise data')
    plt.legend()
    plt.show()



    #Method2: take the median of the epistasis values and plot them
    #mean of -0.004 where we want a mean of -0.1 ish...

    logG_expected = np.log10(m1_fluo/WT_fluo) + np.log10(m2_fluo/WT_fluo)
    logG_model =  np.log10(duplet_fluo/WT_fluo)
    Epistasi = logG_model - logG_expected

    neg_fluo = []
    for index,epi in enumerate(Epistasi):
        if epi < 0:
            neg_fluo.append(duplet_fluo[index])
    kde = gaussian_kde(neg_fluo)
    x = np.linspace(min(neg_fluo),max(neg_fluo),num=1000)
    y = kde(x)
    mode_index = np.argmax(y)
    neg_mode = x[mode_index]
    plt.hist(np.log10(neg_fluo))
    plt.title('negative eps duplex_fluo')
    plt.xlabel('log10(fluorescence)')
    plt.ylabel('density')
    plt.axvline(x = np.log10(exp_duplet_fluo.iloc[1]),c = 'r', label = 'Pairwise data')
    plt.axvline(x = np.log10(neg_mode),c = 'black', label = 'Mode')
    plt.legend()
    plt.show()



    pos_fluo = []
    for index,epi in enumerate(Epistasi):
        if epi > 0:
            pos_fluo.append(duplet_fluo[index])
    kde = gaussian_kde(pos_fluo)
    x = np.linspace(min(pos_fluo),max(pos_fluo),num=1000)
    y = kde(x)
    mode_index = np.argmax(y)
    pos_mode = x[mode_index]
    plt.hist(np.log10(pos_fluo))
    plt.title('positive eps duplex_fluo')
    plt.xlabel('log10(fluorescence)')
    plt.ylabel('density')
    exp_duplet_fluo = pair_mut_dict['obs_fluo_mean']
    plt.axvline(x = np.log10(exp_duplet_fluo.iloc[1]),c = 'r', label = 'Pairwise data')
    plt.axvline(x = np.log10(pos_mode),c = 'black', label = 'Mode')
    plt.legend()
    plt.show()

    kde = gaussian_kde(Epistasi)
    x = np.linspace(min(Epistasi),max(Epistasi),num=1000)
    y = kde(x)
    mode_index = np.argmax(y)
    mode = pd.DataFrame({'Mode_Eps':[]})
    temp = pd.DataFrame({'Mode_Eps':[x[mode_index]]})
    mode = pd.concat([mode,temp], ignore_index=True)

    mode_epi = pd.concat([mode_epi, mode])
    # epi_model = pd.concat([epi_model,Epistasis])

    # plt.xlim([-0.5,0.5])    
    # plt.hist(epi_model,bins = 100, range=[-0.5,0.5])
    # plt.show()
    plt.xlim([-0.5,0.5])    
    plt.hist(Epistasi,bins = 'auto', range=[-0.5,0.5])
    plt.axvline(x = mode_epi.iloc[0][0], c = 'r', label = 'mode')
    plt.title('Epistasis')
    plt.xlabel('Epistasis')
    plt.ylabel('density')
    plt.legend()
    plt.show()
    return
mut_list = ['Sensor10','Output1']
file = 'S10_O1.csv'
Epistasis_analysis(mut_list,file)

#%%
def Compare_Epistasis():
    depth = 10000 # MAX = 250000, number of parameter sets to use per file
    folder = '../results/New_params/Pairwise_params/'
    epi_model = pd.DataFrame()
    mode_epi = pd.DataFrame()

    letters = r'[a-zA-Z]'
    # genotype = get_mut_ids(mut_list)
    DM_df = meta_dict['DM']
    #pair_mut_dict = DM_df[DM_df['genotype'] == genotype]
    # mut1_data = get_data_SM(mut_list[0])
    # mut2_data = get_data_SM(mut_list[1])

    for file in listdir(folder):

        # Extract from file title, which is the duplet
        mutant_letters = re.findall(letters, file)
        m1_str = mutant_letters[0].lower() # first character of filename string
        m2_str = mutant_letters[1].lower() # first character of filename string
        print('mutant_combo: {} with mutants {} and {}'.format(file,m1_str,m2_str))

        # Calulate WT fluorescence
        data_WT = pd.read_csv(folder+file)
        data_WT = data_WT.head(depth) # if depth<10000 then only a subset of parameters is loaded
        data_WT = 10**data_WT #convert from log10
        data_WT['I'] = 0.0002 # WT peak position
        WT_fluo = stst(data_WT)[2]


        # Creating dataframes for singlets and duplet, in these dataframes the paremets will be modified using the fitted modifiers
        data_mut1 = data_WT.copy()
        data_duplet = data_WT.copy()
        data_mut2 = data_WT.copy()

        # Mutant 1 and duplet part
        for par in basepars: # for each parameter of a node
            data_mut1[par+m1_str] = data_mut1[par+m1_str]*data_mut1['M'+par+m1_str]
            data_duplet[par+m1_str] = data_duplet[par+m1_str]*data_duplet['M'+par+m1_str]
        # if m1_str == 'o': # in the case that 1 of the mutatns is the outout, apply fluorescence correction. Note that this should not affect epistasis
        #     data_mut1['Fo'] = data_mut1['F_o']
        #     data_duplet['Fo'] = data_duplet['F_o']

        m1_fluo = stst(data_mut1)[2]
        # plt.hist(np.log10(m1_fluo), bins='auto')
        # plt.axvline(x=np.log10(mut1_data.iloc[5].Stripe), c = 'red') #plot stripe medium inducer conc
        # plt.title('m1_fluo')
        # plt.show()


        # Mutant 2 and duplet part
        for par in basepars:
            data_mut2[par+m2_str] = data_mut2[par+m2_str]*data_mut2['M'+par+m2_str]
            data_duplet[par+m2_str] = data_duplet[par+m2_str]*data_duplet['M'+par+m2_str]
        # if m2_str == 'o':
        #     data_mut2['F_o'] = data_mut2['F_o']*data_mut2['MF_o']
        #     data_duplet['F_o'] = data_duplet['F_o']*data_duplet['MF_o']

        # exp_duplet_fluo = pair_mut_dict['obs_fluo_mean']
        m2_fluo = stst(data_mut2)[2] 
        # plt.hist(np.log10(m2_fluo), bins='auto')
        # plt.axvline(x=np.log10(mut1_data.iloc[5].Stripe), c = 'red')
        # plt.title('m2_fluo')
        # plt.show()

        # Duplet
        duplet_fluo = stst(data_duplet)[2]
        # #Method1: take the median predicted value and calculate the epistasis from there.
        # #   Maybe take only the 5 percent of fluorescent values around the mean and calculate epistasis as those are the more 'accurate' simulations.
        # plt.hist(np.log10(duplet_fluo), bins='auto')
        # plt.title('duplex_fluo')
        # plt.axvline(x = np.log10(exp_duplet_fluo.iloc[1]),c = 'r')
        # plt.show()



        #Method2: take the median of the epistasis values and plot them
        #mean of -0.004 where we want a mean of -0.1 ish...

        logG_expected = np.log10(m1_fluo/WT_fluo) + np.log10(m2_fluo/WT_fluo)
        logG_model =  np.log10(duplet_fluo/WT_fluo)
        Epistasi = logG_model - logG_expected

        # neg_fluo = []
        # for index,epi in enumerate(Epistasi):
        #     if epi < 0:
        #         neg_fluo.append(duplet_fluo[index])
        # kde = gaussian_kde(neg_fluo)
        # x = np.linspace(min(neg_fluo),max(neg_fluo),num=1000)
        # y = kde(x)
        # mode_index = np.argmax(y)
        # neg_mode = x[mode_index]
        # plt.hist(np.log10(neg_fluo))
        # plt.title('negative eps duplex_fluo')
        # plt.axvline(x = np.log10(exp_duplet_fluo.iloc[1]),c = 'r')
        # plt.axvline(x = np.log10(neg_mode),c = 'black')
        # plt.show()



        # pos_fluo = []
        # for index,epi in enumerate(Epistasi):
        #     if epi > 0:
        #         pos_fluo.append(duplet_fluo[index])
        # kde = gaussian_kde(pos_fluo)
        # x = np.linspace(min(pos_fluo),max(pos_fluo),num=1000)
        # y = kde(x)
        # mode_index = np.argmax(y)
        # pos_mode = x[mode_index]
        # plt.hist(np.log10(pos_fluo))
        # plt.title('positive eps duplex_fluo')
        # exp_duplet_fluo = pair_mut_dict['obs_fluo_mean']
        # plt.axvline(x = np.log10(exp_duplet_fluo.iloc[1]),c = 'r')
        # plt.axvline(x = np.log10(pos_mode),c = 'black')
        # plt.show()

        kde = gaussian_kde(Epistasi)
        x = np.linspace(min(Epistasi),max(Epistasi),num=1000)
        y = kde(x)
        mode_index = np.argmax(y)
        mode = pd.DataFrame({'Mode_Eps':[]})
        temp = pd.DataFrame({'Mode_Eps':[x[mode_index]]})
        mode = pd.concat([mode,temp], ignore_index=True)

        mode_epi = pd.concat([mode_epi, mode])
        epi_model = pd.concat([epi_model,Epistasi])

        # plt.xlim([-0.5,0.5])    
        # plt.hist(epi_model,bins = 100, range=[-0.5,0.5])
        # plt.show()
        # plt.xlim([-0.5,0.5])    
        # plt.hist(Epistasi,bins = 'auto', range=[-0.5,0.5])
        # plt.axvline(x = mode_epi.iloc[0][0], c = 'r')
        # plt.show()
    
    plt.hist(epi_model, bins = 'auto', density = True)
    plt.title('Epistasis of all pairwise mutants')
    plt.xlabel('Epistasis')
    plt.ylabel('Density')
    plt.show()
    plt.hist(mode_epi, bins = 'auto', density = True)
    plt.title('Mode Epistasis of all pairwise mutants')
    plt.xlabel('Mode Epistasis')
    plt.ylabel('Density')
    plt.show()

    return mode_epi, epi_model

mode_epi,epi_model = Compare_Epistasis()
# %%
