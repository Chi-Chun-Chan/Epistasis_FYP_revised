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
from Models import *
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
    plt.title('wt_fluo')
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

    data = pd.read_csv(folder+file)
    data['epi'] = Epistasi

    return data
# mut_list = ['Sensor10','Output10']
# file = 'S10_O10.csv'
# Epistasis_analysis(mut_list,file)

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
        m2_fluo = stst(data_mut2)[2] 
        # Duplet
        duplet_fluo = stst(data_duplet)[2]


        logG_expected = np.log10(m1_fluo/WT_fluo) + np.log10(m2_fluo/WT_fluo)
        logG_model =  np.log10(duplet_fluo/WT_fluo)
        Epistasi = logG_model - logG_expected

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
plt.hist(epi_model, bins = 'auto', density = True)
plt.title('Epistasis of all pairwise mutants')
plt.xlabel('Epistasis')
plt.ylabel('Density')
plt.ylim(0,20)
plt.xlim(-0.5,0.5)
plt.show()
# %%

folder = '../results/New_params/Pairwise_params/'
file = 'S10_O10.csv'

data_WT = pd.read_csv(folder+file)
 # if depth<10000 then only a subset of parameters is loaded
data_WT = data_WT #convert from log10
# Creating dataframes for singlets and duplet, in these dataframes the paremets will be modified using the fitted modifiers
data_mut1 = data_WT.copy()
mut1_df = data_mut1.iloc[:,14:18]
# Paired_Density_plot_mut(mut1_df)
data_duplet = data_WT.copy()
data_mut2 = data_WT.copy()
mut2_df = data_mut2.iloc[:,22:]
wt_df = data_WT.iloc[:,0:13]

col_names = list(wt_df.columns.values)
        
g = sns.pairplot(wt_df, vars = col_names, kind = "kde",diag_kind = "hist" corner=True)
# %%

mut_list = ['Regulator6','Sensor7']
file = 'R6_S7.csv'
df = Epistasis_analysis(mut_list,file)
# %%

Neg_epi_df = df[df['epi']<-0.1].reset_index(drop=True)
Neg_epi_df = Neg_epi_df.iloc[:,:-1]
Neg_epi_df = 10**Neg_epi_df
Pos_epi_df = df[df['epi']>0].reset_index(drop=True)
Pos_epi_df = Pos_epi_df.iloc[:,:-1]
Pos_epi_df = 10**Pos_epi_df

SM_df = get_data_SM('Regulator6')
data_ = SM_df
hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)

fig, ((Sensor, Regulator), (Output, Stripe)) = plt.subplots(2,2, constrained_layout=True)
Stripe.scatter(data_.S, data_.Stripe, c = 'teal')
Stripe.set_xscale('log')
Stripe.set_yscale('log')
Stripe.set_title(r'Full circuit with stripe')
Output.scatter(data_.S, data_.Output, c = 'steelblue')
Output.set_title(r'inducer -> S -| Output (GFP)')
Output.set_xscale('log')
Output.set_yscale('log')
Regulator.scatter(data_.S, data_.Regulator, c = 'dimgrey')
Regulator.set_xscale('log')
Regulator.set_yscale('log')
Regulator.set_title(r'inducer ->S -|R (GFP output)')
Sensor.scatter(data_.S, data_.Sensor, c = 'darkorange')
Sensor.set_xscale('log')
Sensor.set_yscale('log')
Sensor.set_title(r'inducer -> sensor (GFP output)')


#Sets up a parameter set array
np.random.seed(0)  

set_list = []

Neg_par_array = np.empty([50,26])
for i in range(1,50+1):
    #selects 50 random unique parameter sets
    rndint = np.random.randint(low=0, high=1e7)
    timeseed = time.time_ns() % 2**16
    np.random.seed(rndint+timeseed)
    seed(rndint+timeseed)
    rand = np.random.randint(low=0, high=len(Neg_par_array))
    check = set_list.count(rand)
    while check > 0:
        rand = np.random.randint(low=0, high=len(Neg_par_array))
        check = set_list.count(rand)
        if check == 0:
            break
    set_list.append(rand)
    row_list = Neg_epi_df.loc[rand].values.flatten().tolist()
    #convert back to normal
    Neg_par_array[i-1] = row_list

set_list = []

Pos_par_array = np.empty([50,26])
for i in range(1,50+1):
    #selects 50 random unique parameter sets
    rndint = np.random.randint(low=0, high=1e7)
    timeseed = time.time_ns() % 2**16
    np.random.seed(rndint+timeseed)
    seed(rndint+timeseed)
    rand = np.random.randint(low=0, high=len(Pos_par_array))
    check = set_list.count(rand)
    while check > 0:
        rand = np.random.randint(low=0, high=len(Pos_par_array))
        check = set_list.count(rand)
        if check == 0:
            break
    set_list.append(rand)
    row_list = Pos_epi_df.loc[rand].values.flatten().tolist()
    #convert back to normal
    Pos_par_array[i-1] = row_list

for i in range(0,len(Pos_par_array)):
    Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = model_hill.model_single_muts(params_list=Pos_par_array[i], I_conc=data_.S, mutant = 'R')

    Stripe.plot(data_.S, Stripe_est_array, alpha = 0.1, c = 'teal')
    Output.plot(data_.S, Output_est_array, alpha = 0.1, c = 'steelblue')
    Regulator.plot(data_.S, Regulator_est_array,alpha = 0.1, c = 'dimgrey')
    Sensor.plot(data_.S, Sensor_est_array, alpha = 0.1, c = 'darkorange')

for i in range(0,len(Neg_par_array)):
    Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = model_hill.model_single_muts(params_list = Neg_par_array[i], I_conc = data_.S, mutant = 'R')

    Stripe.plot(data_.S, Stripe_est_array, alpha = 0.1, c = 'red')
    Output.plot(data_.S, Output_est_array, alpha = 0.1, c = 'steelblue')
    Regulator.plot(data_.S, Regulator_est_array,alpha = 0.1, c = 'dimgrey')
    Sensor.plot(data_.S, Sensor_est_array, alpha = 0.1, c = 'darkorange')
fig.suptitle(f'R10 Mutant Fitting')
plt.show()

SM_df = get_data_SM('Sensor7')
data_ = SM_df
hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)

fig, ((Sensor, Regulator), (Output, Stripe)) = plt.subplots(2,2, constrained_layout=True)
Stripe.scatter(data_.S, data_.Stripe, c = 'teal')
Stripe.set_xscale('log')
Stripe.set_yscale('log')
Stripe.set_title(r'Full circuit with stripe')
Output.scatter(data_.S, data_.Output, c = 'steelblue')
Output.set_title(r'inducer -> S -| Output (GFP)')
Output.set_xscale('log')
Output.set_yscale('log')
Regulator.scatter(data_.S, data_.Regulator, c = 'dimgrey')
Regulator.set_xscale('log')
Regulator.set_yscale('log')
Regulator.set_title(r'inducer ->S -|R (GFP output)')
Sensor.scatter(data_.S, data_.Sensor, c = 'darkorange')
Sensor.set_xscale('log')
Sensor.set_yscale('log')
Sensor.set_title(r'inducer -> sensor (GFP output)')


#Sets up a parameter set array
np.random.seed(0)  

set_list = []

Neg_par_array = np.empty([50,26])
for i in range(1,50+1):
    #selects 50 random unique parameter sets
    rndint = np.random.randint(low=0, high=1e7)
    timeseed = time.time_ns() % 2**16
    np.random.seed(rndint+timeseed)
    seed(rndint+timeseed)
    rand = np.random.randint(low=0, high=len(Neg_par_array))
    check = set_list.count(rand)
    while check > 0:
        rand = np.random.randint(low=0, high=len(Neg_par_array))
        check = set_list.count(rand)
        if check == 0:
            break
    set_list.append(rand)
    row_list = Neg_epi_df.loc[rand].values.flatten().tolist()
    #convert back to normal
    Neg_par_array[i-1] = row_list

set_list = []

Pos_par_array = np.empty([50,26])
for i in range(1,50+1):
    #selects 50 random unique parameter sets
    rndint = np.random.randint(low=0, high=1e7)
    timeseed = time.time_ns() % 2**16
    np.random.seed(rndint+timeseed)
    seed(rndint+timeseed)
    rand = np.random.randint(low=0, high=len(Pos_par_array))
    check = set_list.count(rand)
    while check > 0:
        rand = np.random.randint(low=0, high=len(Pos_par_array))
        check = set_list.count(rand)
        if check == 0:
            break
    set_list.append(rand)
    row_list = Pos_epi_df.loc[rand].values.flatten().tolist()
    #convert back to normal
    Pos_par_array[i-1] = row_list

for i in range(0,len(Pos_par_array)):
    Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = model_hill.model_single_muts(params_list=Pos_par_array[i], I_conc=data_.S, mutant = 'S')

    Stripe.plot(data_.S, Stripe_est_array, alpha = 0.1, c = 'teal')
    Output.plot(data_.S, Output_est_array, alpha = 0.1, c = 'steelblue')
    Regulator.plot(data_.S, Regulator_est_array,alpha = 0.1, c = 'dimgrey')
    Sensor.plot(data_.S, Sensor_est_array, alpha = 0.1, c = 'darkorange')

for i in range(0,len(Neg_par_array)):
    Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = model_hill.model_single_muts(params_list = Neg_par_array[i], I_conc = data_.S, mutant = 'S')

    Stripe.plot(data_.S, Stripe_est_array, alpha = 0.1, c = 'red')
    Output.plot(data_.S, Output_est_array, alpha = 0.1, c = 'steelblue')
    Regulator.plot(data_.S, Regulator_est_array,alpha = 0.1, c = 'dimgrey')
    Sensor.plot(data_.S, Sensor_est_array, alpha = 0.1, c = 'darkorange')
fig.suptitle(f'Output10 Mutant Fitting')
plt.show()
# %%
#convert from log10
# Creating dataframes for singlets and duplet, in these dataframes the paremets will be modified using the fitted modifiers
data_mut1 = df.copy()
mut1_df = data_mut1.iloc[:,14:18]
col_names = list(mut1_df.columns.values)
mut1_df['epi'] = df['epi'].copy()
# Paired_Density_plot_mut(mut1_df)
data_duplet = df.copy()
data_mut2 = df.copy()
mut2_df = data_mut2.iloc[:,22:-1]
col_names2 = list(mut2_df.columns.values)
mut2_df['epi'] = df['epi'].copy()
wt_df = df.iloc[:,0:13]
col_names3 = list(wt_df.columns.values)
wt_df['epi'] = df['epi'].copy()

# mut1_df.loc[mut1_df['epi'] >0, 'epi'] = 'Pos'
# mut1_df.loc[mut1_df['epi'] <0, 'epi'] = 'Neg'
mut1_df['epi'] = mut1_df['epi'].apply(lambda x: 'Pos' if x > 0 else ('Neg' if x < 0 else x))
g = sns.pairplot(mut1_df, vars = col_names, kind = "kde", diag_kind = "hist", hue = 'epi', corner=True)

mut2_df['epi'] = mut2_df['epi'].apply(lambda x: 'Pos' if x > 0 else ('Neg' if x < 0 else x))
h = sns.pairplot(mut2_df, vars = col_names2, kind = "kde", diag_kind = "hist", hue = 'epi', corner=True)

wt_df['epi'] = wt_df['epi'].apply(lambda x: 'Pos' if x > 0 else ('Neg' if x < 0 else x))
z = sns.pairplot(wt_df.head(1000), vars = col_names3, kind = "kde", diag_kind = "hist", hue = 'epi', corner=True)


# %%
