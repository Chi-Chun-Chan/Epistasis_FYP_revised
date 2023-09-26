import numpy as np
from scipy.integrate import odeint
import pandas as pd
from scipy.optimize import root_scalar
#%%
#we will now create model functions for each of the steady state values of Sensor, Regulator and Output.
#Henceforth referred to as S, R and O

#hill function model
def init_model(I_conc,A_s,B_s,C_s,N_s, A_r,C_r,N_r,A_o,C_o,N_o):
    Sensor= (A_s+B_s*(C_s*I_conc)**N_s)/(1+np.power(C_s*I_conc,N_s))
    Regulator = (A_r)/(1+ np.power(C_r*Sensor,N_r))
    Sens_Reg = Sensor + Regulator
    Output = (A_o)/(1+np.power(C_o*Sensor,N_o))
    Stripe = (A_o)/(1+np.power(C_o*Sens_Reg,N_o))
    return Sensor, Regulator, Output, Stripe


#%%
class model_hill:
    example_dict={"sen_params":{"A_s":1,"B_s":1,"C_s":1,"N_s":1},"reg_params":{"A_r":1,"B_r":1,"C_r":1,"N_r":1},"out_h_params":{},"out_params":{"A_o":1,"B_o":1,"C_o":1,"N_o":1,"F_o":1},"free_params":{}}
    def __init__(self,params_list:list,I_conc):
        self.params_list=params_list
        self.I_conc=I_conc
        #self.example_dict_model_1={"sen_params":{"A_s":1,"B_s":1,"C_s":1,"N_s":1},"reg_params":{"A_r":1,"B_r":1,"C_r":1,"N_r":1},"out_h_params":{"A_h":1,"B_h":1,"C_h":1},"out_params":{"A_o":1,"B_o":1,"C_o":1,"N_o":1},"free_params":{"F_o":1}}
        self.example_dict={"sen_params":{"A_s":1,"B_s":1,"C_s":1,"N_s":1},"reg_params":{"A_r":1,"B_r":1,"C_r":1,"N_r":1},"out_h_params":{},"out_params":{"A_o":1,"B_o":1,"C_o":1,"N_o":1},"free_params":{"F_o":1}}
        
        self.n_parameters_1=16
        self.n_parameters_2=13
    def get_dict(self):
        return self.example_dict_model_2
    @staticmethod    
    def model_new_WT(params_list,I_conc): #reformulated with new paramete C_k
        correct_length=14 
        #S is subscript for parameters corresponding to Sensor
        #R is subscript for parameters corresponding to Regulator
        #H is subscript for parameters corresponding to the half network I->S -| O
        #O is subscript for parameters corresponding to Output
   
        if len(params_list)!=correct_length:
            print("params_list of incorrect length should be of length ",correct_length)
            return None
        #sensor
        A_s=params_list[0] #assume that all parameters are un-logged prior
        B_s=params_list[1]
        C_s=params_list[2]
        N_s=params_list[3]
        #regulator
        A_r=params_list[4]
        B_r=params_list[5]
        C_r=params_list[6]
        N_r=params_list[7]
        #out_half
        
        #output
        A_o=params_list[8]
        B_o=params_list[9]
        C_o=params_list[10]
        C_k=params_list[11]
        N_o=params_list[12]
        
        #free
        F_o=params_list[13]
        
        Sensor = A_s+B_s*np.power(C_s*I_conc,N_s)
        Sensor /= 1+np.power(C_s*I_conc,N_s)

        Regulator = B_r/(1+np.power(C_r*Sensor,N_r))
        Regulator += A_r

        Output_half = B_o/(1+np.power(C_o*Sensor,N_o))
        Output_half += A_o

        # Output = A_o + B_o/(1+np.power(((C_o*Regulator)+(C_k*Sensor)),N_o))
        # Output*=F_o

        Output = A_o + B_o/(1+np.power(C_o*((C_k*Sensor) + Regulator),N_o))
        Output*=F_o
        #I wonder why we describe different repression strengths for repression by LacI_regulator and LacI_sensor?
        return Sensor,Regulator,Output_half, Output
    @staticmethod    
    def model_muts(params_list,I_conc): #Incorporated mutant parameters, requires full list of all parameters
        correct_length=26 
        #S is subscript for parameters corresponding to Sensor
        #R is subscript for parameters corresponding to Regulator
        #H is subscript for parameters corresponding to the half network I->S -| O
        #O is subscript for parameters corresponding to Output
        

        if len(params_list)!=correct_length:
            print("params_list of incorrect length should be of length ",correct_length)
            return None
        #sensor
        A_s=params_list[0]
        B_s=params_list[1]
        C_s=params_list[2]
        N_s=params_list[3]
        MA_s=params_list[4]
        MB_s=params_list[5]
        MC_s=params_list[6]
        MN_s=params_list[7]
        
        #regulator
        A_r=params_list[8]
        B_r=params_list[9]
        C_r=params_list[10]
        N_r=params_list[11]
        MA_r=params_list[12]
        MB_r=params_list[13]
        MC_r=params_list[14]
        MN_r=params_list[15]
    
        #out_half
        
        #output
        A_o=params_list[16]
        B_o=params_list[17]
        C_o=params_list[18]
        C_k=params_list[19]
        N_o=params_list[20]
        F_o=params_list[21]

        MA_o=params_list[22]
        MB_o=params_list[23]
        MC_o=params_list[24]
        MN_o=params_list[25]
        
        Sensor = (A_s*MA_s)+(B_s*MB_s)*np.power((C_s*MC_s)*I_conc,(N_s*MN_s))
        Sensor /= (1+np.power((C_s*MC_s)*I_conc,(N_s*MN_s)))

        Regulator = (MB_r*B_r)/(1+np.power((MC_r*C_r)*Sensor,(MN_r*N_r)))
        Regulator += (MA_r*A_r)

        Output_half = (MB_o*B_o)/(1+np.power((MC_o*C_o)*Sensor,(MN_o*N_o)))
        Output_half += (MA_o*A_o)

        Output = (MA_o*A_o) + (MB_o*B_o)/(1+np.power(((MC_o*C_o)*((C_k*Sensor) + Regulator)),(MN_o*N_o)))
        Output*= F_o

        return Sensor,Regulator,Output_half, Output
    
    @staticmethod    
    def model_muts2(params_list,I_conc): #Used for when the given list of parameters has all modifier parameters at the end.
        correct_length=26 
        #S is subscript for parameters corresponding to Sensor
        #R is subscript for parameters corresponding to Regulator
        #H is subscript for parameters corresponding to the half network I->S -| O
        #O is subscript for parameters corresponding to Output
        

        if len(params_list)!=correct_length:
            print("params_list of incorrect length should be of length ",correct_length)
            return None
        #sensor
        A_s=params_list[0]
        B_s=params_list[1]
        C_s=params_list[2]
        N_s=params_list[3]
        
        #regulator
        A_r=params_list[4]
        B_r=params_list[5]
        C_r=params_list[6]
        N_r=params_list[7]
    
        #out_half
    
        #output
        A_o=params_list[8]
        B_o=params_list[9]
        C_o=params_list[10]
        C_k=params_list[11]
        N_o=params_list[12]
        F_o=params_list[13]

        MA_s=params_list[14]
        MB_s=params_list[15]
        MC_s=params_list[16]
        MN_s=params_list[17]
        MA_r=params_list[18]
        MB_r=params_list[19]
        MC_r=params_list[20]
        MN_r=params_list[21]
        MA_o=params_list[22]
        MB_o=params_list[23]
        MC_o=params_list[24]
        MN_o=params_list[25]
        
        Sensor = (A_s*MA_s)+(B_s*MB_s)*np.power((C_s*MC_s)*I_conc,(N_s*MN_s))
        Sensor /= (1+np.power((C_s*MC_s)*I_conc,(N_s*MN_s)))

        Regulator = (MB_r*B_r)/(1+np.power((MC_r*C_r)*Sensor,(MN_r*N_r)))
        Regulator += (MA_r*A_r)

        Output_half = (MB_o*B_o)/(1+np.power((MC_o*C_o)*Sensor,(MN_o*N_o)))
        Output_half += (MA_o*A_o)

        Output = (MA_o*A_o) + (MB_o*B_o)/(1+np.power(((MC_o*C_o)*((C_k*Sensor) + Regulator)),(MN_o*N_o)))
        Output*= F_o

        return Sensor,Regulator,Output_half, Output

    def model_single_muts(params_list,I_conc,mutant): #Used to plot single mutants only, makes all other modifiers 1 regardless of given list.
        correct_length=26 
        #S is subscript for parameters corresponding to Sensor
        #R is subscript for parameters corresponding to Regulator
        #H is subscript for parameters corresponding to the half network I->S -| O
        #O is subscript for parameters corresponding to Output
        

        if len(params_list)!=correct_length:
            print("params_list of incorrect length should be of length ",correct_length)
            return None
        #sensor
        A_s=params_list[0]
        B_s=params_list[1]
        C_s=params_list[2]
        N_s=params_list[3]
        
        #regulator
        A_r=params_list[4]
        B_r=params_list[5]
        C_r=params_list[6]
        N_r=params_list[7]
    
        #out_half
    
        #output
        A_o=params_list[8]
        B_o=params_list[9]
        C_o=params_list[10]
        C_k=params_list[11]
        N_o=params_list[12]
        F_o=params_list[13]

        MA_s=params_list[14]
        MB_s=params_list[15]
        MC_s=params_list[16]
        MN_s=params_list[17]
        MA_r=params_list[18]
        MB_r=params_list[19]
        MC_r=params_list[20]
        MN_r=params_list[21]
        MA_o=params_list[22]
        MB_o=params_list[23]
        MC_o=params_list[24]
        MN_o=params_list[25]

        if mutant == 'S':
            MA_o = 1
            MB_o = 1
            MC_o = 1
            MN_o = 1
            MF_o = 1
            MA_r = 1
            MB_r = 1
            MC_r = 1
            MN_r = 1
        elif mutant == 'R':
            MA_o = 1
            MB_o = 1
            MC_o = 1
            MN_o = 1
            MF_o = 1
            MA_s = 1
            MB_s = 1
            MC_s = 1
            MN_s = 1
        else:
            MA_s = 1
            MB_s = 1
            MC_s = 1
            MN_s = 1
            MA_r = 1
            MB_r = 1
            MC_r = 1
            MN_r = 1
        
        Sensor = (A_s*MA_s)+(B_s*MB_s)*np.power((C_s*MC_s)*I_conc,(N_s*MN_s))
        Sensor /= (1+np.power((C_s*MC_s)*I_conc,(N_s*MN_s)))

        Regulator = (MB_r*B_r)/(1+np.power((MC_r*C_r)*Sensor,(MN_r*N_r)))
        Regulator += (MA_r*A_r)

        Output_half = (MB_o*B_o)/(1+np.power((MC_o*C_o)*Sensor,(MN_o*N_o)))
        Output_half += (MA_o*A_o)

        Output = (MA_o*A_o) + (MB_o*B_o)/(1+np.power(((MC_o*C_o)*((C_k*Sensor) + Regulator)),(MN_o*N_o)))
        Output*= F_o

        return Sensor,Regulator,Output_half, Output

#%%