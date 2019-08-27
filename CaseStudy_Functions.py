#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 12:07:30 2018

@author: maxencedutreix
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from math import exp
from math import sqrt
from math import pi
from scipy.special import erf
from scipy.integrate import quad
from scipy.stats import norm
from matplotlib import rc
import timeit
import bisect
import sys
import igraph
import scipy.sparse as sparse
import scipy.sparse.csgraph
sys.setrecursionlimit(10000)
#from operator import itemgetter
#from prioritydictionary import priorityDictionary
#from graph import DiGraph

plt.rc('font', family='serif')


LOW_1 = 0.0
UP_1 = 4.0
LOW_2 = 0.0
UP_2 = 4.0
sigma1 = sqrt(0.1)
sigma2 = sqrt(0.1)
mu1 = -0.3
mu2 = -0.3
Gaussian_Width_1 = 0.2
Gaussian_Width_2 = 0.2
Semi_Width_1 = Gaussian_Width_1/2.0
Semi_Width_2 = Gaussian_Width_2/2.0

Time_Step = 0.05

def Initial_Partition_Plot(Space):
    
    #Plots the initial state space before verification/synthesis/refinement
       
    fig = plt.figure('Partition P')
    plt.title(r'Initial Partition P', fontsize=25)
        
    plt.plot([0, 4], [0, 0], color = 'k')
    plt.plot([0, 4], [1.0,1.0], color = 'k')
    plt.plot([0, 4], [2.0,2.0], color = 'k')
    plt.plot([0, 4], [3.0,3.0], color = 'k')
    plt.plot([0, 4], [4,4], color = 'k')
    plt.plot([0, 0], [0 ,4], color = 'k')
    plt.plot([1.0, 1.0], [0,4], color = 'k')
    plt.plot([2.0, 2.0], [0,4], color = 'k')
    plt.plot([3.0, 3.0], [0,4], color = 'k')
    plt.plot([4.0, 4.0], [0,4], color = 'k')
    
    ax = plt.gca()
#    
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    ax.set_xlabel('x1', fontsize=20)
    ax.set_ylabel('x2', fontsize=20)
    plt.savefig('Partition.pdf', bbox_inches='tight')
    
    return 1    



def Reachable_Sets_Verification(State):
    
    #Compute the reachable sets for all states in a rectangular partition for mixed-monotone dynamics        
    R_sets = []    
    a = 1.3
    b = 0.25
    
    for i in range(len(State)):
    
        R_set = State[i].copy()    
        R_set[0][0] = State[i][0][0] + (-a*State[i][0][0] + State[i][0][1] )* Time_Step
        R_set[0][1] = State[i][0][1] + (((State[i][0][0]**2)/((State[i][0][0]**2) + 1)) - b* State[i][0][1] ) * Time_Step
        
        R_set[1][0] = State[i][1][0] + (-a*State[i][1][0] + State[i][1][1] )* Time_Step
        R_set[1][1] = State[i][1][1] + (((State[i][1][0]**2)/((State[i][1][0]**2) + 1)) - b* State[i][1][1] ) * Time_Step
        
        R_sets.append(R_set)

    return R_sets






def Probability_Interval_Computation_Verification(R_Set, Target_Set):
    
    #Computes the lower and upper bound probabilities of transition from state
    #to state using the reachable sets in R_set and the target sets in Target_Set
       
    Lower = np.array(np.zeros((len(R_Set),Target_Set.shape[0])))
    Upper = np.array(np.zeros((len(R_Set),Target_Set.shape[0])))
    Reachable_States = [[] for x in range(len(R_Set))]
    Pre_States = [[] for x in range(len(R_Set))]
    Is_Bridge_State = np.zeros(len(R_Set))
    Bridge_Transitions = [[] for x in range(len(R_Set))]
    

    Z1 = (erf(Semi_Width_1/sigma1)/sqrt(2)) - (erf(-Semi_Width_1/sigma1)/sqrt(2))
    Z2 = (erf(Semi_Width_2/sigma2)/sqrt(2)) - (erf(-Semi_Width_2/sigma2)/sqrt(2))
    
    
           
    
    for j in range(len(R_Set)):
        
        
        
        r0 = R_Set[j][0][0]
        r1 = R_Set[j][1][0]
        r2 = R_Set[j][0][1]
        r3 = R_Set[j][1][1]
        
        
        for h in range(len(Target_Set)):

                     
            q0 = Target_Set[h][0][0]
            q1 = Target_Set[h][1][0]
            q2 = Target_Set[h][0][1]
            q3 = Target_Set[h][1][1]
            

            if q0 == LOW_1 and r0 + mu1 - Semi_Width_1 < LOW_1:
                q0 = r0 + mu1 - Semi_Width_1
                    
            if q1 == UP_1 and r1 + mu1 + Semi_Width_1 > UP_1:
                q1 = r1 + mu1 + Semi_Width_1
                    
            if q2 == LOW_2 and r2 + mu2 - Semi_Width_2 < LOW_2:
                q2 = r2 + mu2 - Semi_Width_2
                    
            if q3 == UP_2 and r3 + mu2 + Semi_Width_2 > UP_2:
                q3 = r3 + mu2 + Semi_Width_2                          
            
            
            if (r0 >= q1 + Semi_Width_1 - mu1) or (r1 <= q0 - Semi_Width_1 - mu1) or (r2 >= q3 + Semi_Width_2 - mu2) or (r3 <= q2 - Semi_Width_2 - mu2):
                Lower[j][h] = 0.0
                Upper[j][h] = 0.0
                continue
                
            
            Reachable_States[j].append(h)
            Pre_States[h].append(j)
            
            a1_Opt = ((q0 + q1)/2.0) - mu1
            a2_Opt = ((q2 + q3)/2.0) - mu2
            
            
            
            if (r1 < a1_Opt): 
                a1_Max = r1
                a1_Min = r0
            elif(r0 > a1_Opt): 
                a1_Max = r0
                a1_Min = r1
            else: 
                a1_Max = a1_Opt       
                if (a1_Opt <= (r1+r0)/2.0):
                    a1_Min = r1
                else:
                    a1_Min = r0
                
                
            
            if (r2 > a2_Opt): 
                a2_Max = r2
                a2_Min = r3
            elif(r3 < a2_Opt): 
                a2_Max = r3
                a2_Min = r2
            else: 
                a2_Max = a2_Opt
                if (a2_Opt <= (r2+r3)/2.0):
                    a2_Min = r3
                else:
                    a2_Min = r2
          
                
                
             
                
        
            if a1_Max + mu1 - Semi_Width_1  > q0 and a1_Max + mu1 + Semi_Width_1 < q1 and a2_Max + mu2 - Semi_Width_2 > q2 and a2_Max + mu2 + Semi_Width_2 < q3:
                H = 1.0
            else:
                
                if q0 < a1_Max + mu1 - Semi_Width_1:
                    b0 = a1_Max + mu1 - Semi_Width_1
                else:
                    b0 = q0
                    
                if q1 > a1_Max + mu1 + Semi_Width_1:
                    b1 = a1_Max + mu1 + Semi_Width_1
                else:
                    b1 = q1
                    
                if q2 < a2_Max + mu2 - Semi_Width_2:
                    b2 = a2_Max + mu2 - Semi_Width_2
                else:
                    b2 = q2
                    
                if q3 > a2_Max + mu2 + Semi_Width_2:
                    b3 = a2_Max + mu2 + Semi_Width_2
                else:
                    b3 = q3    
                    
                
                H = ( ( (erf((b1 - a1_Max - mu1)/sigma1)/sqrt(2)) - (erf((b0 - a1_Max - mu1)/sigma1)/sqrt(2)) ) / (Z1)) * ( (erf((b3 - a2_Max - mu2)/sigma2)/sqrt(2)) - (erf((b2 - a2_Max - mu2)/sigma2)/sqrt(2)) ) / ( Z2 )
        
                if H > 1:
                    H = 1.0
                    
                
                
            if (a1_Min + mu1 + Semi_Width_1 <= q0) or (a1_Min + mu1 - Semi_Width_1 >= q1) or (a2_Min + mu2 + Semi_Width_2 <= q2) or (a2_Min + mu2 - Semi_Width_2 >= q3):
                Is_Bridge_State[j] = 1
                Bridge_Transitions[j].append(h)
                Lower[j][h] = 0.0
                Upper[j][h] = H
                continue                
            
            
            else:
                
                if q0 < a1_Min + mu1 - Semi_Width_1:
                    b0 = a1_Min + mu1 - Semi_Width_1
                else:
                    b0 = q0
                    
                if q1 > a1_Min + mu1 + Semi_Width_1:
                    b1 = a1_Min + mu1 + Semi_Width_1
                else:
                    b1 = q1
                    
                if q2 < a2_Min + mu2 - Semi_Width_2:
                    b2 = a2_Min + mu2 - Semi_Width_2
                else:
                    b2 = q2
                    
                if q3 > a2_Min + mu2 + Semi_Width_2:
                    b3 = a2_Min + mu2 + Semi_Width_2
                else:
                    b3 = q3   
                
                
                
                L = ( ( (erf((b1 - a1_Min - mu1)/sigma1)/sqrt(2)) - (erf((b0 - a1_Min - mu1)/sigma1)/sqrt(2)) ) / (Z1)) * ( (erf((b3 - a2_Min - mu2)/sigma2)/sqrt(2)) - (erf((b2 - a2_Min - mu2)/sigma2)/sqrt(2)) ) / ( Z2 )
        
        
                if L < 0:
                    L = 0.0

            if L == 0.0:
                Is_Bridge_State[j] = 1
                Bridge_Transitions[j].append(h)
            
            Lower[j][h] = L
            Upper[j][h] = H
          
    Is_Bridge_State = Is_Bridge_State.astype(int)        
            
    return (Lower,Upper, Reachable_States, Is_Bridge_State, Bridge_Transitions, Pre_States)



    
def Build_Product_IMC(T_l, T_u, A, L, Acc, Reachable_States, Is_Bridge_State, Bridge_Transitions):
    
    #Constructs the product between an IMC (defined by lower transition matrices
    # T_l and T_u) and an Automata A according to Labeling function L
      
    
    IA_l = np.zeros((T_l.shape[0]*len(A),T_l.shape[0]*len(A)))
    IA_u = np.zeros((T_l.shape[0]*len(A),T_l.shape[0]*len(A)))
    Is_A = np.zeros(T_l.shape[0]*len(A))
    Is_N_A = np.zeros(T_l.shape[0]*len(A))
    Which_A = [[] for x in range(T_l.shape[0]*len(A))]
    Which_N_A = [[] for x in range(T_l.shape[0]*len(A))]
    New_Reachable_States = [[] for x in range(T_l.shape[0]*len(A))]
    New_Is_Bridge_State = np.zeros(T_l.shape[0]*len(A))
    New_Bridge_Transitions = [[] for x in range(T_l.shape[0]*len(A))]
    Init = np.zeros((T_l.shape[0]))
    Init = Init.astype(int) #Saves "true" initial state of automaton, accounts for initial state label
    

    
    for x in range(len(Acc)):
        for i in range(len(Acc[x][0])):
            for j in range(T_l.shape[0]):
                Is_N_A[len(A)*j + Acc[x][0][i]] = 1
                Which_N_A[len(A)*j + Acc[x][0][i]].append(x)
        
        for i in range(len(Acc[x][1])):
            for j in range(T_l.shape[0]):
                Is_A[len(A)*j + Acc[x][1][i]] = 1
                Which_A[len(A)*j + Acc[x][1][i]].append(x)            

   
    for i in range(T_l.shape[0]):
        for j in range(len(A)):            
            for k in range(T_l.shape[0]):
                for l in range(len(A)):
                    
                    
                    
                    if L[k] in A[j][l]:
                        
                        if j == 0:
                            Init[k] = l
                                                    
                        IA_l[len(A)*i+j, len(A)*k+l] = T_l[i,k]
                        IA_u[len(A)*i+j, len(A)*k+l] = T_u[i,k]
                        

                        
                        if T_u[i,k] > 0:
                            New_Reachable_States[len(A)*i+j].append(len(A)*k+l)
                            if T_l[i,k] == 0:
                                New_Is_Bridge_State[len(A)*i+j] = 1
                                New_Bridge_Transitions[len(A)*i+j].append(len(A)*k+l)
                        
                    else:
                        IA_l[len(A)*i+j, len(A)*k+l] = 0.0
                        IA_u[len(A)*i+j, len(A)*k+l] = 0.0
    
                 

    Is_A = Is_A.astype(int)
    Is_N_A = Is_N_A.astype(int)
    New_Is_Bridge_State = New_Is_Bridge_State.astype(int)                         

    return (IA_l, IA_u, Is_A, Is_N_A, Which_A, Which_N_A, New_Reachable_States, New_Is_Bridge_State, New_Bridge_Transitions, Init) 





def Find_Largest_BSCCs_One_Pair(I_l, I_u, Acc, N_State_Auto, Is_A_State, Is_N_A_State, Which_A_Pair, Which_N_A_Pair, Reachable_States, Is_Bridge_State, Bridge_Transition, Leaky_States_L_Accepting, Leaky_States_L_Non_Accepting, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Is_in_P, List_I_UA, List_I_UN, Previous_A_BSCC, Previous_Non_A_BSCC, First_Verif):
    
    #Search Algorithm when the Rabin Automata has only 2 Rabin Pairs
    
    #I_l and I_u are respectively the lower and upper bound transition matrices
    #of the product IMC, Acc contains the Rabin Pairs of the Automata, N_State is the
    #number of states in the original system
       
              
    G = np.zeros((I_l.shape[0],I_l.shape[1]))
       
    for i in range(I_l.shape[0]):
        for j in range(I_l.shape[1]):
            if I_u[i,j] > 0:
                G[i,j] = 1

    G_prime = np.copy(G)
    
    
    if First_Verif == 0:
        Deleted_States = []
        Prev_A = set().union(*Previous_A_BSCC)
        Prev_N = set().union(*Previous_Non_A_BSCC)
        Deleted_States.extend(list(set(range(G.shape[0])) - set(Prev_A) - set(Prev_N)))
        
        Ind = list(set(Prev_A)|set(Prev_N))
        Ind.sort()
        G = np.delete(np.array(G),Deleted_States,axis=0)
        G = np.delete(np.array(G),Deleted_States,axis=1)
    else:
        Ind = range(G.shape[0])
        
    First_Verif = 0   
    
                        
    C,n = SSCC(G)
    
       
    SCC_Status = [0]*n ###Each SCC has 'status': 0 normal, 1: sub-SCCs of a largest N-BSCC, 2: sub-SCCs of a largest A-BSCC

    tag = 0
    m = 0

    List_UN = []
    List_UA = []

    Is_In_L_A = np.zeros(I_l.shape[0]) #Is the state in the largest potential accepting BSCC?
    Is_In_L_N_A = np.zeros(I_l.shape[0]) #Is the state in the largest potential non-accepting BSCC?
    Which_A = np.zeros(I_l.shape[0]) #Keeps track of which accepting BSCC does each state belong to (if applicable)
    Which_N_A = np.zeros(I_l.shape[0])
    Which_A.astype(int)
    Which_N_A.astype(int)
    Is_In_L_A.astype(int)
    Is_In_L_N_A.astype(int)
    Potential_Permanent_Accepting = [] #Stores the potential permanent BSCCs until we can check whether it contains a potential component of the other acceptance status
    Potential_Permanent_Accepting_Bridge_States = [] #Stores the potential permanent BSCCs until we can check whether it contains a potential component of the other acceptance status
    Potential_Permanent_Non_Accepting = [] #Stores the potential permanent BSCCs until we can check whether it contains a potential component of the other acceptance status
    Potential_Permanent_Non_Accepting_Bridge_States = [] #Stores the potential permanent BSCCs until we can check whether it contains a potential component of the other acceptance status
        
    Bridge_A = []
    Bridge_N_A = []
        
    while tag == 0:
        
        
        SCC = C[m]

              
        
        #Converts back the SCC's to the indices of the original graph to check if BSCC
        Orig_SCC = []
        for k in range(len(SCC)):
            Orig_SCC.append(Ind[SCC[k]])

        BSCC = 1

     
        Leak = []
        Check_Tag = 1
        Reach_in_R = [[] for x in range(len(Orig_SCC))]
        Pre = [[] for x in range(len(Orig_SCC))]
        All_Leaks = []
        Check_Orig_SCC = np.zeros(len(Orig_SCC), dtype=int)
        
        
        while (len(Leak) != 0 or Check_Tag == 1):
            
            
            ind_leak = []
            Leak = []
            
            for i in range(len(Orig_SCC)):
                
                if Check_Orig_SCC[i] == -1 : continue
                
                Set_All_Leaks = set(Orig_SCC) - set(All_Leaks)
                Diff_List1 = list(set(Reachable_States[Orig_SCC[i]]) - Set_All_Leaks)
                Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[Orig_SCC[i]]))
                
                if Check_Tag == 1:
                    
                    Reach_in_R[i].extend(list(set(Reachable_States[Orig_SCC[i]]) - set(Diff_List1)))
                    for j in range(len(Reach_in_R[i])):
                        Pre[Orig_SCC.index(Reach_in_R[i][j])].append(Orig_SCC[i])
               
                if (len(Diff_List2) != 0) or (sum(I_u[Orig_SCC[i], Reach_in_R[i]])<1) :
                    Leak.append(Orig_SCC[i])
                    ind_leak.append(i)
   
            
            if len(Leak) != 0:
                All_Leaks.extend(Leak)
                BSCC = 0
                for i in range(len(Leak)):
                    Check_Orig_SCC[ind_leak[i]] = -1
                    for j in range(len(Pre[ind_leak[i]])):
                        Reach_in_R[Orig_SCC.index(Pre[ind_leak[i]][j])].remove(Leak[i])
                    
            Check_Tag = 0
           
        if BSCC != 1:    
            SCC = list(set(Orig_SCC) - set(All_Leaks))
            for k in range(len(SCC)):
                SCC[k] = Ind.index(SCC[k])
            
            if len(SCC) != 0:
                SCC = sorted(SCC, key=int)                
                New_G = G[np.ix_(SCC,SCC)]
                
                C_new, n_new = SSCC(New_G)
                for j in range(len(C_new)):
                    for k in range(len(C_new[j])):
                        C_new[j][k] = SCC[C_new[j][k]] 
                    C.append(C_new[j])
                    SCC_Status.append(SCC_Status[m])
                    
            
            
        else:  
            
            Bridge_States = []
                
            if SCC_Status[m] == 0:
                
                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                Inevitable = 1 #Tag to see if BSCC is an inevitable BSCC
                
                
                #First, we go through all the states to check their acceptance status and to see if they eventually leak outside


                for j in range(len(SCC)):
                    
                    
                    if Is_A_State[Ind[SCC[j]]] == 1:
                        acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Which_A_Pair[Ind[SCC[j]]])):
                            indices.append(Which_A_Pair[Ind[SCC[j]]][n])
                        ind_acc.append(indices) 

                    if Is_N_A_State[Ind[SCC[j]]] == 1:
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Which_N_A_Pair[Ind[SCC[j]]])):
                            indices.append(Which_N_A_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         
                        
                      
                    if Is_Bridge_State[Ind[SCC[j]]] == 1:
                                                                                                             
                        Diff_List = np.setdiff1d(Reachable_States[Ind[SCC[j]]], Orig_SCC)                            
                        if len(Diff_List) != 0:
                            Inevitable = 0 
                        Bridge_States.append(Ind[SCC[j]])

                        

                
                
                if Inevitable == 1:
                    #If a BSCC that cannot leak contains no accepting state, then it has to be a permanent non-accepting BSCC                            
                    if len(acc_states) == 0:                        
                        List_I_UN.append([])
                        Leaky_States_P_Non_Accepting.append([])
                        for j in range(len(SCC)):
                            List_I_UN[-1].append(Ind[SCC[j]])                            
                            
                    else:
                        
                        Accept = []
                        Acc_Tag = 0
                        Non_Accept_Remove = [[] for x in range(len(Acc))] #Contains all non-accepting states which prevent the bscc to be accepting for all pairs
                        
                        if len(non_acc_states) == 0:
                            Acc_Tag = 1
                            for j in range(len(acc_states)):
                                Accept.append(acc_states[j])
                        else:        
                            for j in range(len(ind_acc)):
                                for l in range(len(ind_acc[j])):
                                    Check_Tag = 0
                                    Keep_Going = 0
                                    for w in range(len(ind_non_acc)):  
                                        if ind_acc[j][l] in ind_non_acc[w]:
                                            Check_Tag = 1
                                            if len(Non_Accept_Remove[ind_acc[j][l]]) == 0:
                                                Non_Accept_Remove[ind_acc[j][l]].append(non_acc_states[w])
                                                Keep_Going = 1                                
                                            elif Keep_Going == 0:
                                                break                                
                                    if Check_Tag == 0:                                
                                        Accept.append(acc_states[j])
                                        Acc_Tag = 1  
                        
                        if Acc_Tag == 1:
                            
                            Potential_Permanent_Accepting.append([])
                            Potential_Permanent_Accepting_Bridge_States.append([])                                        
                            for n in range(len(SCC)):
                                Potential_Permanent_Accepting[-1].append(Ind[SCC[n]])
                            for n in range(len(Bridge_States)):
                                Potential_Permanent_Accepting_Bridge_States[-1].append(Bridge_States[n])                                                                                    
                              
                        
                            SCC_bis = [x for x in SCC if x not in Accept]
                            if len(SCC_bis) != 0:
                                SCC_bis = sorted(SCC_bis, key=int)           
                                New_G = G[np.ix(SCC_bis, SCC_bis)]
            
                                C_new, n_new = SSCC(New_G)
                                
                                for j in range(len(C_new)):
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC[C_new[j][k]]
                                    C.append(C_new[j])
                                    SCC_Status.append(2)
                                        
                                                                                
                                                                                                              
                        if Acc_Tag == 0: 
                            
                            Potential_Permanent_Non_Accepting.append([])
                            Potential_Permanent_Non_Accepting_Bridge_States.append([])                                        
                            for n in range(len(SCC)):
                                Potential_Permanent_Non_Accepting[-1].append(Ind[SCC[n]])
                            for n in range(len(Bridge_States)):
                                Potential_Permanent_Non_Accepting_Bridge_States[-1].append(Bridge_States[n])                                                                                    
                            
                            for l in range(len(Non_Accept_Remove)):
                               if len(Non_Accept_Remove[l]) != 0:
                                    SCC_bis = [x for x in SCC if x not in Non_Accept_Remove[l]]
                                    if len(SCC_bis) != 0:
                                        SCC_bis = sorted(SCC_bis, key=int)           
                                        New_G = G[np.ix(SCC_bis, SCC_bis)]
                                        C_new, n_new = SSCC(New_G)
                                        for j in range(len(C_new)):
                                            for k in range(len(C_new[j])):
                                                C_new[j][k] = SCC_bis[C_new[j][k]]   
                                            C.append(C_new[j])
                                            SCC_Status.append(1)                            

                #If the BSCC can leak for some induced product MC, then it is not a permanent BSCC                
                elif len(acc_states) == 0:
                    List_UN.append([])
                    Bridge_N_A.append([])
                    Leaky_States_L_Non_Accepting.append([])
                    for j in range(len(SCC)):
                        List_UN[-1].append(Ind[SCC[j]])
                        Which_N_A[Ind[SCC[j]]] = len(List_UN) - 1
                        Is_In_L_N_A[Ind[SCC[j]]] = 1
                    for x in range(len(Bridge_States)):
                        Bridge_N_A[-1].append(Bridge_States[x])
                            
          

                
                else:
                   
                    Acc_Tag = 0
                    Accept = [] #Contains unmatched accepting states
                    
                                        
                    if len(non_acc_states) == 0:
                        Acc_Tag = 1
                        for j in range(len(acc_states)):
                            Accept.append(acc_states[j])
                    
                    else:
                        Non_Accept_Remove = [[] for x in range(len(Acc))] #Contains all non-accepting states which prevent the bscc to be accepting for all pairs
                        
                        for j in range(len(ind_acc)):
                            for l in range(len(ind_acc[j])):
                                Check_Tag = 0
                                Keep_Going = 0
                                for w in range(len(ind_non_acc)):  
                                    if ind_acc[j][l] in ind_non_acc[w]:
                                        Check_Tag = 1
                                        if len(Non_Accept_Remove[ind_acc[j][l]]) == 0:
                                            Non_Accept_Remove[ind_acc[j][l]].append(non_acc_states[w])
                                            Keep_Going = 1                                
                                        elif Keep_Going == 0:
                                            break                                
                                if Check_Tag == 0:                                
                                    Accept.append(acc_states[j])
                                    Acc_Tag = 1  
                    
                    
                    #The BSCC is a potential accepting BSCC. Need to remove all unmatched accepting states to check if it contains potential non-accepting BSCCs
                    if Acc_Tag == 1:
                        List_UA.append([])
                        Bridge_A.append([])
                        Leaky_States_L_Accepting.append([])
                        for n in range(len(SCC)):
                            List_UA[-1].append(Ind[SCC[n]])
                            Which_A[Ind[SCC[n]]] = len(List_UA) - 1
                            Is_In_L_A[Ind[SCC[n]]] = 1
                        
                        for x in range(len(Bridge_States)):
                            Bridge_A[-1].append(Bridge_States[x])
                        

                        SCC_bis = [x for x in SCC if x not in Accept]
                        if len(SCC_bis) != 0:
                            SCC_bis = sorted(SCC_bis, key=int)           

                            New_G = G[np.ix(SCC_bis,SCC_bis)]
        
                            C_new, n_new = SSCC(New_G)
                            
                            for j in range(len(C_new)):
                                for k in range(len(C_new[j])):
                                    C_new[j][k] = SCC[C_new[j][k]]
                                C.append(C_new[j])
                                SCC_Status.append(2)
                                
                    #The BSCC is a potential non-accepting BSCC. Need to remove all matched non-accepting states to check if it contains potential accepting BSCCs            
                    else:
                        
                        List_UN.append([])
                        Bridge_N_A.append([])
                        Leaky_States_L_Non_Accepting.append([])
                        for n in range(len(SCC)):
                            List_UN[-1].append(Ind[SCC[n]])
                            Which_N_A[Ind[SCC[n]]] = len(List_UN) - 1
                            Is_In_L_N_A[Ind[SCC[n]]] = 1
                       
                        for x in range(len(Bridge_States)):
                            Bridge_N_A[-1].append(Bridge_States[x])
                        
                        for l in range(len(Non_Accept_Remove)):
                           if len(Non_Accept_Remove[l]) != 0:
                                SCC_bis = [x for x in SCC if x not in Non_Accept_Remove[l]]
                                if len(SCC_bis) != 0:
                                    SCC_bis = sorted(SCC_bis, key=int)           

                                    New_G = G[np.ix_(SCC_bis, SCC_bis)]
                                    C_new, n_new = SSCC(New_G)
                                    for j in range(len(C_new)):
                                        for k in range(len(C_new[j])):
                                            C_new[j][k] = SCC_bis[C_new[j][k]]   
                                        C.append(C_new[j])
                                        SCC_Status.append(1)
                        
  
              
            if SCC_Status[m] == 1: ###This BSCC is part of a Potentially Larger Non-A BSCC, Want to check if it contains a potential accepting BSCC 
                 
                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                                        
                for j in range(len(SCC)):
                    
                    if Is_A_State[Ind[SCC[j]]] == 1:
                        acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Which_A_Pair[Ind[SCC[j]]])):
                            indices.append(Which_A_Pair[Ind[SCC[j]]][n])
                        ind_acc.append(indices) 

                    if Is_N_A_State[Ind[SCC[j]]] == 1:
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Which_N_A_Pair[Ind[SCC[j]]])):
                            indices.append(Which_N_A_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices) 
                             
                    if Is_Bridge_State[Ind[SCC[j]]] == 1:
                        Bridge_States.append(Ind[SCC[j]])
                
                
                if len(acc_states) > 0:
                    Acc_Tag = 0
                    
                    if len(non_acc_states) == 0:
                        Acc_Tag = 1
                    else:     
                        Non_Accept_Remove = [[] for x in range(len(Acc))] #Contains all non-accepting states which prevent the bscc to be accepting for all pairs
                        
                        for j in range(len(ind_acc)):
                            for l in range(len(ind_acc[j])):
                                Check_Tag = 0
                                Keep_Going = 0
                                for w in range(len(ind_non_acc)):  
                                    if ind_acc[j][l] in ind_non_acc[w]:
                                        Check_Tag = 1
                                        if len(Non_Accept_Remove[ind_acc[j][l]]) == 0:
                                            Non_Accept_Remove[ind_acc[j][l]].append(non_acc_states[w])
                                            Keep_Going = 1                                
                                        elif Keep_Going == 0:
                                            break                                
                                if Check_Tag == 0:                                
                                    Acc_Tag = 1  
                                                                    
                    if Acc_Tag == 1:
                        
                        
                        List_UA.append([])
                        Bridge_A.append([])
                        Leaky_States_L_Accepting.append([])
                        for j in range(len(SCC)):
                            List_UA[-1].append(Ind[SCC[j]])
                            Which_A[Ind[SCC[j]]] = len(List_UA) - 1
                            Is_In_L_A[Ind[SCC[j]]] = 1
                        for x in range(len(Bridge_States)):
                            Bridge_A[-1].append(Bridge_States[x])  
                            
                    else:    
                        
                        for l in range(len(Non_Accept_Remove)):
                           if len(Non_Accept_Remove[l]) != 0:
                                SCC_bis = [x for x in SCC if x not in Non_Accept_Remove[l]]
                                if len(SCC_bis) != 0:
                                    SCC_bis = sorted(SCC_bis, key=int)           

                                    New_G = G[np.ix_(SCC_bis, SCC_bis)]
                                    C_new, n_new = SSCC(New_G)
                                    for j in range(len(C_new)):
                                        for k in range(len(C_new[j])):
                                            C_new[j][k] = SCC_bis[C_new[j][k]]   
                                        C.append(C_new[j])
                                        SCC_Status.append(1)
             
            if SCC_Status[m] == 2: ###This SCC is part of a Potentially Larger A BSCC, Want to check if it contains a potential non-accepting BSCC
                 

                ind_acc = []
                ind_non_acc = []
                acc_states = []
                non_acc_states = []
                
                
                for j in range(len(SCC)):
                    
                    
                    if Is_A_State[Ind[SCC[j]]] == 1:
                        acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Which_A_Pair[Ind[SCC[j]]])):
                            indices.append(Which_A_Pair[Ind[SCC[j]]][n])
                        ind_acc.append(indices) 

                    if Is_N_A_State[Ind[SCC[j]]] == 1:
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Which_N_A_Pair[Ind[SCC[j]]])):
                            indices.append(Which_N_A_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         

                    if Is_Bridge_State[Ind[SCC[j]]] == 1:                                         
                            Bridge_States.append(Ind[SCC[j]])

                                      
                
                if len(acc_states) == 0:
                    
                    
                    List_UN.append([])
                    Bridge_N_A.append([])
                    Leaky_States_L_Non_Accepting.append([])
                    for j in range(len(SCC)):
                        List_UN[-1].append(Ind[SCC[j]])
                        Which_N_A[Ind[SCC[j]]] = len(List_UN) - 1
                        Is_In_L_N_A[Ind[SCC[j]]] = 1
                    for x in range(len(Bridge_States)):
                        Bridge_N_A[-1].append(Bridge_States[x])
                    
                
                else:   
                    
                        Acc_Tag = 0
                        Accept = [] #Contains unmatched accepting states
                        for j in range(len(ind_acc)):                            
                            for l in range(len(ind_acc[j])):
                                if Accept[-1] == acc_states[j]:
                                    break
                                Check_Tag = 0
                                for w in range(len(ind_non_acc)):  
                                    if ind_acc[j][l] in ind_non_acc[w]:
                                        Check_Tag = 1
                                        break                                
                                if Check_Tag == 0:                                
                                    Accept.append(acc_states[j])
                                    Acc_Tag = 1                         
    
    
                        if Acc_Tag == 0:
                            
                            
                            List_UN.append([])
                            Bridge_N_A.append([])
                            Leaky_States_L_Non_Accepting.append([])
                            for j in range(len(SCC)):
                                List_UN[-1].append(Ind[SCC[j]])
                                Which_N_A[Ind[SCC[j]]] = len(List_UN) - 1
                                Is_In_L_N_A[Ind[SCC[j]]] = 1
                            for x in range(len(Bridge_States)):
                                Bridge_N_A[-1].append(Bridge_States[x])
                            
                        else:
                             
                            SCC_bis = [x for x in SCC if x not in Accept]
                            if len(SCC_bis) != 0:
                                SCC_bis = sorted(SCC_bis, key=int)           
                                New_G = G[np.ix_(SCC_bis, SCC_bis)]
            
                                C_new, n_new = SSCC(New_G)
                                
                                for j in range(len(C_new)):
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC[C_new[j][k]]
                                    C.append(C_new[j])
                                    SCC_Status.append(2)
                
                
        m +=1
        if m == len(C): tag = 1
    
    
    for i in range(len(Potential_Permanent_Accepting)):
        Check = 0
        for j in range(len(Potential_Permanent_Accepting[i])):
            if Is_In_L_N_A[Potential_Permanent_Accepting[i][j]] == 1:
                Check = 1
                break
        
        if Check == 0: 
            List_I_UA.append(Potential_Permanent_Accepting[i])
            Leaky_States_P_Accepting.append([])                       

        
        else: 
            List_UA.append([])
            Bridge_A.append(Potential_Permanent_Accepting_Bridge_States[i])
            Leaky_States_L_Accepting.append([])
            for n in range(len(Potential_Permanent_Accepting[i])):
                List_UA[-1].append(Potential_Permanent_Accepting[i][n])
                Which_A[Potential_Permanent_Accepting[i][n]] = len(List_UA) - 1
                Is_In_L_A[Potential_Permanent_Accepting[i][n]] = 1
    
            
    for i in range(len(Potential_Permanent_Non_Accepting)):
        Check = 0       
        for j in range(len(Potential_Permanent_Non_Accepting[i])):
            if Is_In_L_A[Potential_Permanent_Non_Accepting[i][j]] == 1:
                Check = 1
                break        
        if Check == 0:
            List_I_UN.append(Potential_Permanent_Non_Accepting[i])
            Leaky_States_P_Non_Accepting.append([])                     
      
        else:            
            List_UN.append([])
            Bridge_N_A.append(Potential_Permanent_Non_Accepting_Bridge_States[i])
            Leaky_States_L_Non_Accepting.append([])
            for n in range(len(Potential_Permanent_Non_Accepting[i])):
                List_UN[-1].append(Potential_Permanent_Non_Accepting[i][n])
                Which_N_A[Potential_Permanent_Non_Accepting[i][n]] = len(List_UN) - 1
                Is_In_L_N_A[Potential_Permanent_Non_Accepting[i][n]] = 1

    Which_A = Which_A.astype(int)
    Which_N_A = Which_N_A.astype(int)
    Is_In_L_A = Is_In_L_A.astype(int)
    Is_In_L_N_A = Is_In_L_N_A.astype(int)
    
    return (List_UN, List_UA, List_I_UN, List_I_UA, Is_In_L_A, Is_In_L_N_A, Which_A, Which_N_A, Bridge_A, Bridge_N_A, G_prime, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Leaky_States_L_Accepting, Leaky_States_L_Non_Accepting)





def Find_Winning_Losing_Components(I_l, I_u, List_L_N_A, List_L_A, List_I_N_A, List_I_A, Reachable_States, Is_Bridge_State, Bridge_Transition, G, Bridge_Acc, Bridge_N_Acc, Is_State_In_L_A, Is_State_In_L_N_A, Is_in_P, Leaky_States_L_Accepting, Leaky_States_L_Non_Accepting, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Leaky_L_Accepting_Previous, Leaky_L_Non_Accepting_Previous, Previous_A_BSCC, Previous_Non_A_BSCC):

    WC_L = [[] for i in range(len(List_L_A))]
    WC_I = []
    Is_in_WC_L = np.zeros(G.shape[0])
    Which_WC_L = [[] for i in range(G.shape[0])] #First number in list tells you which BSCC, second number tells you which component around the BSCC
    Bridge_WC_L = [[] for i in range(len(List_L_A))]
    Is_in_LC_L = np.zeros(G.shape[0])
    Which_LC_L = [[] for i in range(G.shape[0])] #First number in list tells you which BSCC, second number tells you which component around the BSCC
    Bridge_LC_L = [[] for i in range(len(List_L_N_A))]
    
    
    Is_in_WC_P = np.zeros(G.shape[0]) #Is the state in a potential winning component around a permanent winning component?
    Which_WC_P = [[] for i in range(G.shape[0])]
    Bridge_WC_P = [[] for i in range(len(List_I_A))]
    Is_in_LC_P = np.zeros(G.shape[0]) #Is the state in a potential losing component around a permanent losing component?
    Which_LC_P = [[] for i in range(G.shape[0])]
    Bridge_LC_P = [[] for i in range(len(List_I_N_A))]
    LC_L = [[] for i in range(len(List_L_N_A))]
    LC_I = []
    
    
    G_original = np.copy(G)
    
    
    for n in range(len(List_I_A)):
             
        
        T = List_I_A[n]
        G = np.copy(G_original)
   
        C = []
        W = []
        m = 0
        
        if len(Leaky_States_P_Accepting[n]) == 0:
            for q in range(len(Previous_A_BSCC)):
                if T[0] in Previous_A_BSCC[q]:
                    Leaky_States_P_Accepting[n] = list(Leaky_L_Accepting_Previous[q])  
                    break        

        Ind = [x for x in (range(G.shape[0])) if x not in Leaky_States_P_Accepting[n]]      

        Indices = np.zeros(G_original.shape[0], dtype=int)
        for i in range(len(Ind)):
            Indices[Ind[i]] = i
        
        G = np.delete(np.asarray(G), Leaky_States_P_Accepting[n], axis = 0)
        G = np.delete(np.asarray(G), Leaky_States_P_Accepting[n], axis = 1)
        
        while len(C) != 0 or m == 0:                        
            Gr = igraph.Graph.Adjacency(G.tolist())
            
            if m != 0:
                R_prev = list(R)             
            
            R = []
            
            for q in range(len(T)):                 
                Res = Gr.subcomponent(Indices[T[q]], mode="IN")
                R2 = [x for x in Res if x not in R]
                R.extend(R2)
 
            for q in range(len(R)):
                R[q] = Ind[R[q]] #Converting back to original indices 
            
            if m == 0:             
                Tr = set(range(G_original.shape[0])) - set(R)                   

            else:
                Tr = Tr | (set(R_prev) - set(R))
                
            R2 = list(R)
            R = list( set(R) - set(T) )
            
            C = []
            Is_In_C = np.zeros(G_original.shape[0])
            Leaks = list(R2)
            
            if m == 0:
           
                Ind_Original_R = np.zeros(G_original.shape[0], dtype=int)
                for i in range(len(R)):
                    Ind_Original_R[R[i]] = i
                Reach_in_R = [[] for x in range(len(R))]
                Pre = [[] for x in range(len(R))]
                W = []
            
            
            ind_leak = []
            C = []

            
            for i in range(len(R)):
                                
                Diff_List1 = set(Reachable_States[R[i]]) - set(R2)
                Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[R[i]]))
                
                if m == 0:
                    
                    Reach_in_R[i].extend(list(set(Reachable_States[R[i]]) - Diff_List1 - set(T) ))
                    for j in range(len(Reach_in_R[i])):
                        Pre[R.index(Reach_in_R[i][j])].append(R[i])
                    Reach_in_R[i].extend(set.intersection(set(Reachable_States[R[i]]), set(T)))   
               
                if (len(Diff_List2) != 0) or (sum(I_u[R[i], Reach_in_R[Ind_Original_R[R[i]]]])<1) :
                    C.append(R[i])
                    ind_leak.append(i)
   
            
            if len(C) != 0:
                W.extend(C)
                for i in range(len(C)):
                    for j in range(len( Pre[Ind_Original_R[C[i]]] )):
                        Reach_in_R[Ind_Original_R[Pre[Ind_Original_R[C[i]]][j]]].remove(C[i])
                                        
#           Declare those states as dead by creating a self loop 
            C_tran  = []
            for i in range(len(C)):
                C_tran.append(Ind.index(C[i]))
            
            for i in range(len(C_tran)):
                G[C_tran[i],:] = np.zeros(G.shape[1])
                G[C_tran[i],C_tran[i]] = 1
            
            List = list(set(range(G.shape[0])) - set(C_tran))
            
            for i in range(len(List)):
                for j in range(len(C_tran)):
                    G[List[i],C_tran[j]] = 0
            
            m += 1
        
        Leaky_States_P_Accepting[n] = list(set(Tr) | set(W))       
        R = list(set(range(G_original.shape[0])) - set(W))
        
        R = list(set(R) - set(Tr))
        B = list(R)
        
        for q in range(len(R)):
            Is_in_WC_L[R[q]] = 1
       
        #Below, we want to check whether two sets of Winning components are disjoint        
        
        # Find sets of disjoint Components W around the BSCC
        
        #Remove potential non-accepting states to find permanent components
            
        Remove = []    
        for i in range(len(R)): 
            if Is_State_In_L_N_A[R[i]] == 1:
                Remove.append(R[i])      
        R = list(set(R) - set(Remove))
        
        C = []
        m = 0
        Is_In_C = np.zeros(G_original.shape[0])

        for i in range(len(R)):
            
            if R[i] in (C+T): continue           
            leak = 0
            Leaky_State = []
            Diff_List = np.setdiff1d(Reachable_States[R[i]], R)
                    
            if (len(Diff_List) != 0):
                leak = 1
                C.append(R[i])
                Is_In_C[R[i]] = 1
                Leaky_State.append(R[i])

                                                                                     
            if leak == 1:
                
                tag2 = 0
                k = 0
                while tag2 == 0:
                                             
                    for j in range(len(R)):
                        if Is_In_C[R[j]] == 0:
                            if Leaky_State[k] in Reachable_States[R[j]]:
                                C.append(R[j])
                                Is_In_C[R[j]] = 1
                                Leaky_State.append(R[j])
                                               
                    k += 1                   
                    if k == len(Leaky_State): tag2 = 1
                   

                
        List_Permanents = np.setdiff1d(R, C).tolist()
        

        for i in range(len(List_Permanents)):
            Is_in_P[List_Permanents[i]] = 1
        
        List_I_A[n] = list(List_Permanents)
        
        
        WC_I = WC_I + List_Permanents
            
        B = list(set(B) - set(List_Permanents))                                                       
        
        G = np.copy(G_original)
        G = G[np.ix_(B, B)]
        
        num, D = scipy.sparse.csgraph.connected_components(G, directed = False)
        W = [[] for i in range(num) ]
        for q in range(len(D)):
            for i in range(num):
                if D[q] == i:
                    W[i].append(B[q])
                    break
        
        for q in range(len(W)):
            Set_C = W[q]            
            Bridges = []
            for l in range(len(Set_C)):
                Is_in_WC_P[Set_C[l]] = 1
                Which_WC_P[Set_C[l]].append([n,q])
                if Is_Bridge_State[Set_C[l]] == 1:
                    Bridges.append(Set_C[l])
            Bridge_WC_P[n].append(Bridges) 
            
            
            
    
    for n in range(len(List_L_A)):
        
        
        T = List_L_A[n]
             
        Already_Check = 1 #Checks if the BSCC already belongs to a larger potential component. If that is the case, the BSCC doesn't need to be rechecked
        for q in range(len(T)):
            if Is_in_WC_L[T[q]] == 1:
                Already_Check = 0
                break
                   
        if Already_Check == 0:
            continue
        
        for q in range(len(Previous_A_BSCC)):
            if T[0] in Previous_A_BSCC[q]:
                Leaky_States_L_Accepting[n] = list(Leaky_L_Accepting_Previous[q])
                break
         
                        
        G = np.copy(G_original)
        C = []
        W = []
        m = 0
        
        Ind = [x for x in (range(G.shape[0])) if x not in Leaky_States_L_Accepting[n]]            
        Indices = np.zeros(G_original.shape[0], dtype=int)
        for i in range(len(Ind)):
            Indices[Ind[i]] = i 
            
            
        G = np.delete(np.asarray(G), Leaky_States_L_Accepting[n], axis = 0)
        G = np.delete(np.asarray(G), Leaky_States_L_Accepting[n], axis = 1)
                
        while len(C) != 0 or m == 0:                        
            Gr = igraph.Graph.Adjacency(G.tolist())
            
            if m != 0:
                R_prev = list(R)
            
            R = []
            
            for q in range(len(T)):                 
                Res = Gr.subcomponent(Indices[T[q]], mode="IN")
                R2 = [x for x in Res if x not in R]
                R.extend(R2)

 
            for q in range(len(R)):

                R[q] = Ind[R[q]] #Converting back to original indices 
            
            if m == 0:             
                Tr = set(range(G_original.shape[0])) - set(R)                   
            else:
                Tr = Tr | (set(R_prev) - set(R))
                
            R2 = list(R)
            R = list( set(R) - set(T) )
            
            C = []
            Is_In_C = np.zeros(G_original.shape[0])
            Leaks = list(R2)
            
            if m == 0:
           
                Ind_Original_R = np.zeros(G_original.shape[0], dtype=int)
                for i in range(len(R)):
                    Ind_Original_R[R[i]] = i
                Reach_in_R = [[] for x in range(len(R))]
                Pre = [[] for x in range(len(R))]
                W = []
            
            
            ind_leak = []
            C = []

            
            for i in range(len(R)):
                                
                Diff_List1 = set(Reachable_States[R[i]]) - set(R2)
                Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[R[i]]))
                
                if m == 0:
                    
                    Reach_in_R[i].extend(list(set(Reachable_States[R[i]]) - Diff_List1 - set(T) ))
                    for j in range(len(Reach_in_R[i])):
                        Pre[R.index(Reach_in_R[i][j])].append(R[i])
                    Reach_in_R[i].extend(set.intersection(set(Reachable_States[R[i]]), set(T)))   
               
                if (len(Diff_List2) != 0) or (sum(I_u[R[i], Reach_in_R[Ind_Original_R[R[i]]]])<1) :
                    C.append(R[i])
                    ind_leak.append(i)
   
            
            if len(C) != 0:
                W.extend(C)
                for i in range(len(C)):
                    for j in range(len( Pre[Ind_Original_R[C[i]]] )):
                        Reach_in_R[Ind_Original_R[Pre[Ind_Original_R[C[i]]][j]]].remove(C[i])
                
                
#             Declare those states as dead by creating a self loop 
            C_tran  = []
            for i in range(len(C)):
                C_tran.append(Ind.index(C[i]))
            
            for i in range(len(C_tran)):
                G[C_tran[i],:] = np.zeros(G.shape[1])
                G[C_tran[i],C_tran[i]] = 1
            
            List = list(set(range(G.shape[0])) - set(C_tran))
            
            for i in range(len(List)):
                for j in range(len(C_tran)):
                    G[List[i],C_tran[j]] = 0
            
            m += 1        
        
        
        Leaky_States_L_Accepting[n] = list(Tr | set(W))
        L = set(range(G_original.shape[0])) - set(W)
        List_Comp = list(L - set(Tr))
        


        WC_L[n].append(List_Comp)
        
        for q in range(len(WC_L[n][-1])):
            Is_in_WC_L[WC_L[n][-1][q]] = 1
        
        WC_L[n].pop()
        
        for q in range(len(T)):
            Which_WC_L[T[q]].append([n,0])
        WC_L[n].append(T)
        Bridge_WC_L[n].append(Bridge_Acc[n])
                
        #Below, we want to check whether two sets of Winning components are disjoint        
        Win_Comp = list(set(List_Comp) - set(T)) 
        
        # Find sets of disjoint Components W around the BSCC
        
        
        G = np.copy(G_original)
        G = G[np.ix_(Win_Comp, Win_Comp)] 

        num, D = scipy.sparse.csgraph.connected_components(G, directed = False)
        W = [[] for i in range(num)]

        for q in range(len(D)):
            for i in range(num):
                if D[q] == i:
                    W[i].append(Win_Comp[q])
                    break


        for q in range(len(W)):
            Set_C = W[q]           
            WC_L[n].append(Set_C)
            Bridges = list(Bridge_Acc[n])
            for l in range(len(Set_C)):
                Which_WC_L[Set_C[l]].append([n,q+1])
                if Is_Bridge_State[Set_C[l]] == 1:
                    Bridges.append(Set_C[l])
            Bridge_WC_L[n].append(Bridges)
            
        
     
        
    for n in range(len(List_I_N_A)):
        
        T = List_I_N_A[n]
        G = np.copy(G_original)

        if len(Leaky_States_P_Non_Accepting[n]) == 0:
            for q in range(len(Previous_Non_A_BSCC)):
                if T[0] in Previous_Non_A_BSCC[q]:
                    Leaky_States_P_Non_Accepting[n] = list(Leaky_L_Non_Accepting_Previous[q])
                    break

        Ind = [x for x in (range(G.shape[0])) if x not in Leaky_States_P_Non_Accepting[n]]            
        
        Indices = np.zeros(G_original.shape[0], dtype=int)
        for i in range(len(Ind)):
            Indices[Ind[i]] = i         
        
        G = np.delete(np.asarray(G), Leaky_States_P_Non_Accepting[n], axis = 0)
        G = np.delete(np.asarray(G), Leaky_States_P_Non_Accepting[n], axis = 1)
        
        C = []
        W = []
        m = 0
                
        while len(C) != 0 or m == 0:                        
            Gr = igraph.Graph.Adjacency(G.tolist())
            
            if m != 0:
                R_prev = list(R)             
            
            R = []
            
            for q in range(len(T)):                 
                Res = Gr.subcomponent(Indices[T[q]], mode="IN")
                R2 = [x for x in Res if x not in R]
                R.extend(R2)

 
            for q in range(len(R)):
                R[q] = Ind[R[q]] #Converting back to original indices 
            
            if m == 0:             
                Tr = set(range(G_original.shape[0])) - set(R)                   

            else:
                Tr = Tr | (set(R_prev) - set(R))
                
            R2 = list(R)
            R = list( set(R) - set(T) )
            
            C = []
            Is_In_C = np.zeros(G_original.shape[0])
            Leaks = list(R2)
            
            if m == 0:
           
                Ind_Original_R = np.zeros(G_original.shape[0], dtype=int)
                for i in range(len(R)):
                    Ind_Original_R[R[i]] = i
                Reach_in_R = [[] for x in range(len(R))]
                Pre = [[] for x in range(len(R))]
                W = []
            
            
            ind_leak = []
            C = []

            
            for i in range(len(R)):
                                
                Diff_List1 = set(Reachable_States[R[i]]) - set(R2)
                Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[R[i]]))
                
                if m == 0:
                    
                    Reach_in_R[i].extend(list(set(Reachable_States[R[i]]) - Diff_List1 - set(T) ))
                    for j in range(len(Reach_in_R[i])):
                        Pre[R.index(Reach_in_R[i][j])].append(R[i])
                    Reach_in_R[i].extend(set.intersection(set(Reachable_States[R[i]]), set(T)))   
               
                if (len(Diff_List2) != 0) or (sum(I_u[R[i], Reach_in_R[Ind_Original_R[R[i]]]])<1) :
                    C.append(R[i])
                    ind_leak.append(i)
   
            
            if len(C) != 0:
                W.extend(C)

                for i in range(len(C)):
                    for j in range(len( Pre[Ind_Original_R[C[i]]] )):
                        Reach_in_R[Ind_Original_R[Pre[Ind_Original_R[C[i]]][j]]].remove(C[i])                    
                    
                
#             Declare those states as dead by creating a self loop 
            C_tran  = []
            for i in range(len(C)):
                C_tran.append(Ind.index(C[i]))
            
            for i in range(len(C_tran)):
                G[C_tran[i],:] = np.zeros(G.shape[1])
                G[C_tran[i],C_tran[i]] = 1
            
            List = list(set(range(G.shape[0])) - set(C_tran))
            
            for i in range(len(List)):
                for j in range(len(C_tran)):
                    G[List[i],C_tran[j]] = 0
            
                                                                    
            m += 1
        
        Leaky_States_P_Non_Accepting[n] = list(set(Tr) | set(W))       
        R = list(set(range(G_original.shape[0])) - set(W))
        R = list(set(R) - set(Tr))
        B = list(R)    #Save the largest potential components around the BSCC    

        for q in range(len(R)):
            Is_in_LC_L[R[q]] = 1            


        #Remove states from R that belong to a potential accepting BSCC    
        Remove = []    
        for i in range(len(R)): 
            if Is_State_In_L_A[R[i]] == 1:
                Remove.append(R[i])
        
        R = list(set(R) - set(Remove))
            
        C = []
        m = 0
        Is_In_C = np.zeros(G_original.shape[0])
               
        for i in range(len(R)):
            
            if R[i] in (C+T): continue           
            leak = 0
            Leaky_State = []
            Diff_List = np.setdiff1d(Reachable_States[R[i]], R)

            if (len(Diff_List) != 0):
                leak = 1
                C.append(R[i])
                Is_In_C[R[i]] = 1
                Leaky_State.append(R[i])
                
                                                                                     
            if leak == 1:
                
                tag2 = 0
                k = 0
                while tag2 == 0:
                                             
                    for j in range(len(R)):
                        if Is_In_C[R[j]] == 0:
                            if Leaky_State[k] in Reachable_States[R[j]]:
                                C.append(R[j])
                                Is_In_C[R[j]] = 1
                                Leaky_State.append(R[j])
                                               
                    k += 1                   
                    if k == len(Leaky_State): tag2 = 1
                   
         
        List_Permanents = np.setdiff1d(R, C).tolist()   
        for i in range(len(List_Permanents)):
            Is_in_P[List_Permanents[i]] = 1                  
            
        List_I_N_A[n] = list(List_Permanents)
            
        LC_I = LC_I + List_Permanents 
        
        #Below, we want to check whether two sets of Winning components are disjoint         
        
        # Find sets of disjoint Components W around the BSCC
        B = list(set(B) - set(List_Permanents))

        G = np.copy(G_original)
        G = G[np.ix_(B, B)]
        
        num, D = scipy.sparse.csgraph.connected_components(G, directed = False)
        W = [[] for i in range(num) ]
        for q in range(len(D)):
            for i in range(num):
                if D[q] == i:
                    W[i].append(B[q])
                    break
        
        for q in range(len(W)):
            Set_C = W[q]            
            Bridges = []
            for l in range(len(Set_C)):
                Is_in_LC_P[Set_C[l]] = 1
                Which_LC_P[Set_C[l]].append([n,q])
                if Is_Bridge_State[Set_C[l]] == 1:
                    Bridges.append(Set_C[l])
            Bridge_LC_P[n].append(Bridges)             

    for n in range(len(List_L_N_A)):
        
        T = List_L_N_A[n]

        
        Already_Check = 1 #Checks if the BSCC already belongs to a larger potential component. If that is the case, the BSCC doesn't need to be rechecked
        for q in range(len(T)):
            if Is_in_LC_L[T[q]] == 1:
                Already_Check = 0
                break
       
        if Already_Check == 0:
            continue

        Indices = np.zeros(G_original.shape[0], dtype=int)
        for i in range(len(Ind)):
            Indices[Ind[i]] = i 
       
        for q in range(len(Previous_Non_A_BSCC)):
            if T[0] in Previous_Non_A_BSCC[q]:
                Leaky_States_L_Non_Accepting[n] = list(Leaky_L_Non_Accepting_Previous[q])
                break
                  
        G = np.copy(G_original)
        C = []
        W = []
        m = 0
        
        Ind = [x for x in (range(G.shape[0])) if x not in Leaky_States_L_Non_Accepting[n]]            
        
        Indices = np.zeros(G_original.shape[0], dtype=int)
        for i in range(len(Ind)):
            Indices[Ind[i]] = i
            
        G = np.delete(np.asarray(G), Leaky_States_L_Non_Accepting[n], axis = 0)
        G = np.delete(np.asarray(G), Leaky_States_L_Non_Accepting[n], axis = 1)
        
        while len(C) != 0 or m == 0:    
                    
            Gr = igraph.Graph.Adjacency(G.tolist())           
            if m != 0:
                R_prev = list(R)
            
            R = []
            
            for q in range(len(T)):                 
                Res = Gr.subcomponent(Indices[T[q]], mode="IN")
                R2 = [x for x in Res if x not in R]
                R.extend(R2)
 
            for q in range(len(R)):
                R[q] = Ind[R[q]] #Converting back to original indices 
            
            if m == 0:             
                Tr = set(range(G_original.shape[0])) - set(R)                   
            else:
                Tr = Tr | (set(R_prev) - set(R))
                
            R2 = list(R)
            R = list( set(R) - set(T) )
            
            C = []
            Is_In_C = np.zeros(G_original.shape[0])
            
            if m == 0:
           
                Ind_Original_R = np.zeros(G_original.shape[0], dtype=int)
                for i in range(len(R)):
                    Ind_Original_R[R[i]] = i
                Reach_in_R = [[] for x in range(len(R))]
                Pre = [[] for x in range(len(R))]
                W = []
            
            
            ind_leak = []
            C = []

            
            for i in range(len(R)):
                                
                Diff_List1 = set(Reachable_States[R[i]]) - set(R2)
                Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[R[i]]))
                
                if m == 0:
                    
                    Reach_in_R[i].extend(list(set(Reachable_States[R[i]]) - Diff_List1 - set(T) ))
                    for j in range(len(Reach_in_R[i])):
                        Pre[R.index(Reach_in_R[i][j])].append(R[i])
                    Reach_in_R[i].extend(set.intersection(set(Reachable_States[R[i]]), set(T)))   
               
                if (len(Diff_List2) != 0) or (sum(I_u[R[i], Reach_in_R[Ind_Original_R[R[i]]]])<1) :
                    C.append(R[i])
                    ind_leak.append(i)
   
            
            if len(C) != 0:
                W.extend(C)
                for i in range(len(C)):
                    for j in range(len( Pre[Ind_Original_R[C[i]]] )):
                        Reach_in_R[Ind_Original_R[Pre[Ind_Original_R[C[i]]][j]]].remove(C[i])
                
                
#             Declare those states as dead by creating a self loop 
            C_tran  = []
            for i in range(len(C)):
                C_tran.append(Ind.index(C[i]))
            
            for i in range(len(C_tran)):
                G[C_tran[i],:] = np.zeros(G.shape[1])
                G[C_tran[i],C_tran[i]] = 1
            
            List = list(set(range(G.shape[0])) - set(C_tran))
            
            for i in range(len(List)):
                for j in range(len(C_tran)):
                    G[List[i],C_tran[j]] = 0
            
            m += 1        
        
        
        Leaky_States_L_Non_Accepting[n] = list(Tr | set(W))
        L = set(range(G_original.shape[0])) - set(W)
        List_Comp = list(L - set(Tr))       
       
        
        LC_L[n].append(List_Comp)
        
        for q in range(len(LC_L[n][-1])):
            Is_in_LC_L[LC_L[n][-1][q]] = 1
        

        LC_L[n].pop()
        
        for q in range(len(T)):
            Which_LC_L[T[q]].append([n,0])
        LC_L[n].append(T)
        Bridge_LC_L[n].append(Bridge_N_Acc[n])
        
        
        #Below, we want to check whether sets of Losing components are disjoint        
        Los_Comp = list(set(List_Comp) - set(T)) 
        
        # Find sets of disjoint Components W around the BSCC
        
        
        G = np.copy(G_original)
        G = G[np.ix_(Los_Comp, Los_Comp)]        
        num, D = scipy.sparse.csgraph.connected_components(G, directed = False)
        W = [[] for i in range(num) ]

        for q in range(len(D)):
            for i in range(num):
                if D[q] == i:
                    W[i].append(Los_Comp[q])
                    break

        for q in range(len(W)):
            Set_C = W[q]            
            LC_L[n].append(Set_C)
            Bridges = list(Bridge_N_Acc[n])
            for l in range(len(Set_C)):
                Which_LC_L[Set_C[l]].append([n,q+1])
                if Is_Bridge_State[Set_C[l]] == 1:
                    Bridges.append(Set_C[l])
            Bridge_LC_L[n].append(Bridges)
  

        
    All_WC_L = []   
    All_LC_L = []      
        
    for n in range(Is_in_WC_L.shape[0]):
        if Is_in_WC_L[n] == 1:
            All_WC_L.append(n)
    for n in range(Is_in_LC_L.shape[0]):
        if Is_in_LC_L[n] == 1:
            All_LC_L.append(n)
    
            
    for n in range(len(LC_I)):
        Is_in_LC_L[LC_I[n]] = 0
        
    for n in range(len(WC_I)):
        Is_in_WC_L[WC_I[n]] = 0
        
    for n in range(len(Is_in_WC_P)):
        if Is_in_WC_P[n] == 1:
            Is_in_WC_L[n] = 0
    
        if Is_in_LC_P[n] == 1:
            Is_in_LC_L[n] = 0 
            
            
    Previous_A_BSCC = copy.deepcopy(List_L_A)
    Previous_Non_A_BSCC = copy.deepcopy(List_L_N_A)
    Leaky_L_Accepting_Previous = copy.deepcopy(Leaky_States_L_Accepting)
    Leaky_L_Non_Accepting_Previous = copy.deepcopy(Leaky_States_L_Non_Accepting)
    

    return WC_L, WC_I, LC_L, LC_I, All_WC_L, All_LC_L, Is_in_WC_L, Which_WC_L, Bridge_WC_L, Is_in_LC_L, Which_LC_L, Bridge_LC_L, Is_in_WC_P, Which_WC_P, Bridge_WC_P, Is_in_LC_P, Which_LC_P, Bridge_LC_P, Is_in_P, List_I_A, List_I_N_A, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Leaky_L_Accepting_Previous, Leaky_L_Non_Accepting_Previous, Previous_A_BSCC, Previous_Non_A_BSCC




def Reachability_Upper(IA_l, IA_u, Q1, Q0, Num_States, Automata_size, Reach, Init):
    
    #Q1 are the sets whose reachability needs to be computed
    #Q0 are the sets whose reachability is already decided (Inevitable BSCCs)
    
    Descending_Order = []
    Index_Vector = np.zeros((IA_l.shape[0],1))  
      
    for k in range(IA_l.shape[0]):
                             
        if k in Q1:       

            Index_Vector[k,0] = 1.0
            Descending_Order.insert(0,k)
            
        elif k not in Q0:
            
            Index_Vector[k,0] = 0.0
            Descending_Order.append(k) 
            
    for k in range(len(Q0)):
        Descending_Order.append(Q0[k])
    

    
    d = {k:v for v,k in enumerate(Descending_Order)} 
    Sort_Reach = []

 
    for i in range(len(Reach)):
        Reach[i].sort(key=d.get)
        Sort_Reach.append(Reach[i])
    
 

    Phi_Max = Phi_Computation_Upper(IA_u, IA_l, Descending_Order, Q1, Q0, Reach, Sort_Reach)
    Steps_High = np.dot(Phi_Max, Index_Vector)
   
    for i in range(len(Q1)):    
        Steps_High[Q1[i]][0] = 1.0
    for i in range(len(Q0)): 
        Steps_High[Q0[i]][0] = 0.0
               
    Success_Intervals = []    
    for i in range(IA_l.shape[0]):       
        Success_Intervals.append(Steps_High[i][0])   
    
    Terminate_Check = 0
    Convergence_threshold = 0.000000001
      
    while Terminate_Check == 0:
        
        
        Previous_List = copy.copy(Descending_Order)         
              
        
        for i in range(len(Q0)):
            Success_Intervals[Q0[i]] = 0.0
        
        for i in range(len(Q1)):
            Success_Intervals[Q1[i]] = 1.0
       
        Descending_Order = np.array(range(len(Success_Intervals)))
        Success_Array = np.array(Success_Intervals)
        Descending_Order = list(Descending_Order[(-Success_Array).argsort()])
        
        d = {k:v for v,k in enumerate(Descending_Order)}
        Sort_Reach = []
        for i in range(len(Reach)):
            Reach[i].sort(key=d.get)
            Sort_Reach.append(Reach[i])
            
        
        if Previous_List != Descending_Order:
            Phi_Max = Phi_Computation_Upper(IA_u, IA_l, Descending_Order, Q1, Q0, Reach, Sort_Reach)

         
        Steps_High = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Max), Steps_High)

    
        for i in range(len(Q1)):  
            Steps_High[Q1[i]][0] = 1.0

            
        for i in range(len(Q0)):   
            Steps_High[Q0[i]][0] = 0.0 
            
   
                  
        Max_Difference = 0
               
        
        for i in range(IA_l.shape[0]):
           
            Max_Difference = max(Max_Difference, abs(Success_Intervals[i] - Steps_High[i][0]))            
            Success_Intervals[i] = Steps_High[i][0]
            
        if Max_Difference < Convergence_threshold:       
            Terminate_Check = 1
     

            
    Bounds = []
    Prod_Bounds = []

    
    for i in range(Num_States):

        Bounds.append(Success_Intervals[i*Automata_size+Init[i]])
        
    for i in range(len(Success_Intervals)):

        Prod_Bounds.append(Success_Intervals[i]) 
       
    return (Bounds, Prod_Bounds, Phi_Max)




def Phi_Computation_Upper(Upper, Lower, Order_D, q1, q0, Reach, Reach_Sort):

    Phi_max = np.zeros((Upper.shape[0], Upper.shape[1]))   
    
             
    
    for j in range(Upper.shape[0]):       
        if j in q1 or j in q0:
            continue      
        else:   
            
            Up = Upper[j][:]
            Low = Lower[j][:]          
            Sum_1_D = 0.0
            Sum_2_D = sum(Low[Reach[j]])
            Phi_max[j][Reach_Sort[j][0]] = min(Low[Reach_Sort[j][0]] + 1 - Sum_2_D, Up[Reach_Sort[j][0]]) 
                             
            for i in range(1, len(Reach_Sort[j])):  
                               
                Sum_1_D = Sum_1_D + Phi_max[j][Reach_Sort[j][i-1]]               
                if Sum_1_D >= 1:
                    break     
                                       
                Sum_2_D = Sum_2_D - Low[Reach_Sort[j][i-1]]                                     
                Phi_max[j][Reach_Sort[j][i]] = min(Low[Reach_Sort[j][i]] + 1 - (Sum_1_D+Sum_2_D), Up[Reach_Sort[j][i]])
             
    
    
    return Phi_max




def Update_Ordering_Upper(State_Space, Q0, Q1, Int):
         
    Descending_Order = []
    First_State = 0 
    
    


    for k in range(State_Space.shape[0]):
              
        if k not in (Q0 + Q1):
            
            if First_State == 0:
                
                Descending_Order.append(k)
                First_State = 1
                
            else:
                
                for l in range(len(Descending_Order)):
                    
                    if (Int[Descending_Order[l]][0] < Int[k][0]):
                        
                        Descending_Order.insert(l, k)
                        break
                    
                    if l == len(Descending_Order) - 1:
                        Descending_Order.append(k)                        
                    
                       
    for k in range(State_Space.shape[0]):
                                        
        if k in Q1:
            Descending_Order.insert(0,k)
            
        if k in Q0:
            Descending_Order.append(k)
            
    return Descending_Order





def Reachability_Lower(IA_l, IA_u, Q1, Q0, Num_States, Automata_size, Reach, Init):
    
    #Q1 are the sets whose reachability needs to be computed
    #Q0 are the sets whose reachability is already decided (Inevitable BSCCs)
    
    Ascending_Order = []
    Index_Vector = np.zeros((IA_l.shape[0],1))  
    
    for k in range(IA_l.shape[0]):
                                
        if k in Q1:
            
            Index_Vector[k,0] = 1.0
            Ascending_Order.append(k)
           
        elif k not in Q0:
            
            Index_Vector[k,0] = 0.0
            Ascending_Order.insert(0,k)

    for k in range(len(Q0)): 
        Ascending_Order.insert(0,Q0[k])
        

    d = {k:v for v,k in enumerate(Ascending_Order)} 
    Sort_Reach = []

    for i in range(len(Reach)):
        Reach[i].sort(key=d.get)
        Sort_Reach.append(Reach[i])        
        
                
    Phi_Min = Phi_Computation_Lower(IA_u, IA_l, Ascending_Order, Q1, Q0, Reach, Sort_Reach)
    Steps_Low = np.dot(Phi_Min, Index_Vector)
    
    for i in range(len(Q1)):    
        Steps_Low[Q1[i]][0] = 1.0
    for i in range(len(Q0)):
        Steps_Low[Q0[i]][0] = 0.0

    
    Success_Intervals = []
         
    for i in range(IA_l.shape[0]):       

        Success_Intervals.append(Steps_Low[i][0])
              
    Terminate_Check = 0
    Convergence_threshold = 0.000001
    Previous_Max_Difference = 1
      
    while Terminate_Check == 0:
                   
        Previous_List = copy.copy(Ascending_Order)
        
        for i in range(len(Q0)):
            Success_Intervals[Q0[i]] = 0.0
        
        for i in range(len(Q1)):
            Success_Intervals[Q1[i]] = 1.0
       
        Ascending_Order = np.array(range(len(Success_Intervals)))
        Success_Array = np.array(Success_Intervals)
        Ascending_Order = list(Ascending_Order[(Success_Array).argsort()]) 
        
        d = {k:v for v,k in enumerate(Ascending_Order)} 
        Sort_Reach = []

        for i in range(len(Reach)):
            Reach[i].sort(key=d.get)
            Sort_Reach.append(Reach[i]) 
        
        if Previous_List != Ascending_Order:
            Phi_Min = Phi_Computation_Lower(IA_u, IA_l, Ascending_Order, Q1, Q0, Reach, Sort_Reach)
            
        Steps_Low = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Min), Steps_Low)
        
        for i in range(len(Q1)):   
            Steps_Low[Q1[i]][0] = 1.0
  
        for i in range(len(Q0)):
            Steps_Low[Q0[i]][0] = 0.0 
 

              
           
        Max_Difference = 0
        
               
        for i in range(IA_l.shape[0]):
           
                       
            Max_Difference = max(Max_Difference, abs(Success_Intervals[i] - Steps_Low[i][0]))        
                    
    
            Success_Intervals[i] = Steps_Low[i][0]
         
        if Max_Difference < Convergence_threshold:       
            Terminate_Check = 1
                   
    Bounds = []
    Prod_Bounds = []
    
    for i in range(Num_States):

        Bounds.append(Success_Intervals[i*Automata_size+Init[i]])
    
    for i in range(len(Success_Intervals)):

        Prod_Bounds.append(Success_Intervals[i])
    
       
    return (Bounds, Prod_Bounds, Phi_Min)




def Phi_Computation_Lower(Upper, Lower, Order_A, q1, q0, Reach, Reach_Sort):

    Phi_min = np.zeros((Upper.shape[0], Upper.shape[1]))


   
    for j in range(Upper.shape[0]):
        
        if j in q1 or j in q0:
            continue
        else:
    
            Up = Upper[j][:]
            Low = Lower[j][:]                 
            Sum_1_A = 0.0
            Sum_2_A = sum(Low[Reach[j]])
            Phi_min[j][Reach_Sort[j][0]] = min(Low[Reach_Sort[j][0]] + 1 - Sum_2_A, Up[Reach_Sort[j][0]])  
      
            for i in range(1, len(Reach_Sort[j])):
                             
                Sum_1_A = Sum_1_A + Phi_min[j][Reach_Sort[j][i-1]]
                if Sum_1_A >= 1:
                    break
                Sum_2_A = Sum_2_A - Low[Reach_Sort[j][i-1]]
                Phi_min[j][Reach_Sort[j][i]] = min(Low[Reach_Sort[j][i]] + 1 - (Sum_1_A+Sum_2_A), Up[Reach_Sort[j][i]])  
        
    return Phi_min




def Update_Ordering_Lower(State_Space, Q0, Q1, Int):
       
    Ascending_Order = []
    First_State = 0
    
    for k in range(State_Space.shape[0]):
              
        if k not in (Q0 + Q1):
            
            if First_State == 0:
                
                Ascending_Order.append(k)
                First_State = 1
                
            else:
                
                for l in range(len(Ascending_Order)):
                    
                    if (Int[Ascending_Order[-1-l]][0] < Int[k][0]):
                        
                        Ascending_Order.insert(-l, k)
                        break
                    
                    if l == len(Ascending_Order) - 1:
                        Ascending_Order.insert(0, k)
                                             
                                        
    for k in range(State_Space.shape[0]):
                                       
        
        if k in Q1:           
            Ascending_Order.append(k)

        elif k in Q0:          
            Ascending_Order.insert(0,k)
            
    return Ascending_Order




def SSCC(graph):
    
    #Search for all Strongly Connected Components in a Graph

    #set of visited vertices
    used = set()
    
    #call first depth-first search
    list_vector = [] #vertices in topological sorted order
    for vertex in range(len(graph)):
       if vertex not in used:
          (list_vector,used) = first_dfs(vertex, graph, used, list_vector)              
    list_vector.reverse()
    
    #preparation for calling second depth-first search
    graph_t = reverse_graph(graph)
    used = set()
    
    #call second depth-first search
    components= []
    list_components = [] #strong-connected components
    scc_quantity = 0 #quantity of strong-connected components 
    for vertex in list_vector:
        if vertex not in used:
            scc_quantity += 1
            list_components = []
            (list_components, used) = second_dfs(vertex, graph_t, list_components, list_vector, used)
            components.append(list_components)
            
    
    return components, scc_quantity



def first_dfs(vertex, graph, used, list_vector):
    used.add(vertex)
    for v in range(len(graph)):   
        if graph[vertex][v] == 1 and v not in used:   
            (list_vector, used) = first_dfs(v, graph, used, list_vector)
    list_vector.append(vertex)
    return(list_vector, used)

    
def second_dfs(vertex, graph_t, list_components, list_vector, used):
    used.add(vertex)
    for v in list_vector:   
        if graph_t[vertex][v] == 1 and v not in used:   
            (list_components, used) = second_dfs(v, graph_t, list_components, list_vector, used)
    list_components.append(vertex)
    return(list_components, used)
    		                   
    
def reverse_graph(graph):
    graph_t = list(zip(*graph))
    return graph_t


def State_Space_Plot(Space, Y, N, M, Tag):
    
    
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    
    if Tag == 0:
        fig = plt.figure('Initial State Space')
        plt.title(r'Initial State Space', fontsize=25)
        
    else:
        fig = plt.figure('Final State Space')
        plt.title(r'Final State Space', fontsize=25)
    
    plt.xlim([0,4])
    plt.ylim([0,4])

    ax=plt.gca()
    
    
    
    for i in range(Space.shape[0]):      
        if i in N:
            
            pol = plt.Rectangle((Space[i][0][0], Space[i][0][1]), Space[i][1][0] - Space[i][0][0], Space[i][1][1] - Space[i][0][1], facecolor='red', edgecolor='k', linewidth = 0.1)
            
        elif i in Y:            
            pol = plt.Rectangle((Space[i][0][0], Space[i][0][1]), Space[i][1][0] - Space[i][0][0], Space[i][1][1] - Space[i][0][1], facecolor='green', edgecolor='k', linewidth = 0.1)
        
        else:           
            pol = plt.Rectangle((Space[i][0][0], Space[i][0][1]), Space[i][1][0] - Space[i][0][0], Space[i][1][1] - Space[i][0][1], facecolor='yellow', edgecolor='k', linewidth = 0.1)
        
        ax.add_artist(pol)
    
    ax1 = plt.gca()
    ax1.set_xlabel('x1', fontsize=20)
    ax1.set_ylabel('x2', fontsize=20)
    
    plt.savefig('results.pdf', bbox_inches='tight')
      
    return 1



def Refinement_IMC_Global_Path(S_Space, Y_States, N_States, M_States, Best_Markov_Chain, Worst_Markov_Chain, Product_Intervals, Success_Intervals, Id, p_thresh, Ord, Len_Automata, Low_Bound_Matrix, Upp_Bound_Matrix, Automaton, Automaton_Acc, L_map, IA1_u, IA1_l, List_Largest_Non_A_BSCC, List_Largest_A_BSCC, List_Inevitable_Non_A_BSCC, List_Inevitable_A_BSCC, Bridge_A, Bridge_N_A, Is_In_L_N_A, Is_In_L_A, Which_A, Which_N_A, Reachable_States, Is_Bridge_State, Bridge_Transitions, Pre_States, Product_Reachable_States,  Is_in_WC_L, Which_WC_L, Bridge_WC_L, Is_in_LC_L, Which_LC_L, Bridge_LC_L, Is_in_WC_P, Which_WC_P, Bridge_WC_P, Is_in_LC_P, Which_LC_P, Bridge_LC_P, Is_in_P, V_Stop, V_Total, Leaky_States_L_Accepting, Leaky_States_L_Non_Accepting, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Leaky_L_Accepting_Previous, Leaky_L_Non_Accepting_Previous, Previous_A_BSCC, Previous_Non_A_BSCC):
    
    Tag = 0
    Max_I = 0.0
    V_Maybe = 1.0
     
    while ((V_Maybe > V_Stop) and len(M_States) != 0) or Tag == 0:
                     
        Tag += 1      
        Set_Refinement = []
        Maybe_States_Product = []
        
        for j in range(len(M_States)):                           
            Maybe_States_Product.append(Len_Automata*M_States[j])
              
         

        Weights = Uncertainty_Ranking_Path(Best_Markov_Chain, Worst_Markov_Chain, Maybe_States_Product, Product_Intervals, IA1_u, IA1_l, Is_in_WC_L, Which_WC_L, Bridge_WC_L, Is_in_LC_L, Which_LC_L, Bridge_LC_L, Is_in_WC_P, Which_WC_P, Bridge_WC_P, Is_in_LC_P, Which_LC_P, Bridge_LC_P, Is_in_P, S_Space, V_Total, Len_Automata)
    
        # The lines below convert uncertainty in the product IMC into uncertainty in the original system

        Weights_Original_S = [0.0 for n in range(S_Space.shape[0])]
        
        for j in range(len(Weights)):
            Weights_Original_S[j/Len_Automata] += Weights[j]            
        Uncertainty_Ranking_Original_S = list(reversed(sorted(range(len(Weights_Original_S)), key=lambda x: Weights_Original_S[x])))
            
        Weights_Original_S = list(reversed(sorted(Weights_Original_S)))

        Index_Stop = -1   
        Cut_off_Percentage = 0.10
        for j in range(len(Weights_Original_S)):
            if (Weights_Original_S[j]/Weights_Original_S[0]) < Cut_off_Percentage:
                Index_Stop = j
                break
            if Weights_Original_S[j] <= 0.0:
                Index_Stop = j
                break
    
        if Index_Stop == -1: Index_Stop = len(Weights_Original_S)        
        Index_Stop1 = (S_Space.shape[0]) #Parameter that sets the maximum number of states to be refined       
        Num_States = min(len(Uncertainty_Ranking_Original_S), Index_Stop, Index_Stop1)

        for j in range(Num_States):       
            Set_Refinement.append(Uncertainty_Ranking_Original_S[j])
        

               
        Set_Refinement.sort()            
        New_States = Raw_Refinement(Set_Refinement, S_Space)

        
        
        
        """ Deleting previous state and replacing it with new states, as well as updating index for some lists """
    
        First_Loop = 1
        First_Loop2 = 1
        
        Index_Stop = np.zeros(S_Space.shape[0]+len(Set_Refinement))
        Index_Stop = Index_Stop.astype(int)
        Index_Stop2 = np.zeros(S_Space.shape[0]+len(Set_Refinement))
        Index_Stop2 = Index_Stop2.astype(int)        
        Index_Stop3 = np.zeros(len(Leaky_States_P_Accepting))
        Index_Stop3 = Index_Stop3.astype(int)
        Index_Stop4 = np.zeros(len(Leaky_States_P_Non_Accepting))
        Index_Stop4 = Index_Stop4.astype(int)
        Index_Stop5 = np.zeros(len(Leaky_L_Accepting_Previous))
        Index_Stop5 = Index_Stop5.astype(int)       
        Index_Stop6 = np.zeros(len(Leaky_L_Non_Accepting_Previous))
        Index_Stop6 = Index_Stop6.astype(int)
        Index_Stop7 = np.zeros(len(Previous_A_BSCC))
        Index_Stop7 = Index_Stop7.astype(int)
        Index_Stop8 = np.zeros(len(Previous_Non_A_BSCC))
        Index_Stop8 = Index_Stop8.astype(int)
        
        Copy_List_Inevitable_Non_A_BSCC = []
        for i in range(len(List_Inevitable_Non_A_BSCC)):
            Copy_List_Inevitable_Non_A_BSCC.append([])
            for j in range(len(List_Inevitable_Non_A_BSCC[i])):
                Copy_List_Inevitable_Non_A_BSCC[-1].append(List_Inevitable_Non_A_BSCC[i][j])
                
        Copy_List_Inevitable_A_BSCC = [] 
        for i in range(len(List_Inevitable_A_BSCC)):
            Copy_List_Inevitable_A_BSCC.append([])
            for j in range(len(List_Inevitable_A_BSCC[i])):
                Copy_List_Inevitable_A_BSCC[-1].append(List_Inevitable_A_BSCC[i][j]) 
        
        
        Copy_Leaky_States_P_Accepting = [] 
        for i in range(len(Leaky_States_P_Accepting)):
            Copy_Leaky_States_P_Accepting.append([])
            Leaky_States_P_Accepting[i].sort()
            for j in range(len(Leaky_States_P_Accepting[i])):
                Copy_Leaky_States_P_Accepting[-1].append(Leaky_States_P_Accepting[i][j])        


        Copy_Leaky_States_P_Non_Accepting = [] 
        for i in range(len(Leaky_States_P_Non_Accepting)):
            Copy_Leaky_States_P_Non_Accepting.append([])
            Leaky_States_P_Non_Accepting[i].sort()
            for j in range(len(Leaky_States_P_Non_Accepting[i])):
                Copy_Leaky_States_P_Non_Accepting[-1].append(Leaky_States_P_Non_Accepting[i][j])  

                
        Copy_Leaky_L_Accepting_Previous = [] 
        for i in range(len(Leaky_L_Accepting_Previous)):
            Copy_Leaky_L_Accepting_Previous.append([])
            Leaky_L_Accepting_Previous[i].sort()
            for j in range(len(Leaky_L_Accepting_Previous[i])):
                Copy_Leaky_L_Accepting_Previous[-1].append(Leaky_L_Accepting_Previous[i][j])  
 
        Copy_Leaky_L_Non_Accepting_Previous = [] 
        for i in range(len(Leaky_L_Non_Accepting_Previous)):
            Copy_Leaky_L_Non_Accepting_Previous.append([])
            Leaky_L_Non_Accepting_Previous[i].sort()
            for j in range(len(Leaky_L_Non_Accepting_Previous[i])):
                Copy_Leaky_L_Non_Accepting_Previous[-1].append(Leaky_L_Non_Accepting_Previous[i][j])  


        Copy_Previous_A_BSCC = [] 
        for i in range(len(Previous_A_BSCC)):
            Copy_Previous_A_BSCC.append([])
            Previous_A_BSCC[i].sort()
            for j in range(len(Previous_A_BSCC[i])):
                Copy_Previous_A_BSCC[-1].append(Previous_A_BSCC[i][j])  


        Copy_Previous_Non_A_BSCC = [] 
        for i in range(len(Previous_Non_A_BSCC)):
            Copy_Previous_Non_A_BSCC.append([])
            Previous_Non_A_BSCC[i].sort()
            for j in range(len(Previous_Non_A_BSCC[i])):
                Copy_Previous_Non_A_BSCC[-1].append(Previous_Non_A_BSCC[i][j])  
                                                                       

    

        for m in range(len(Set_Refinement)):
                       
            
            S_Space = np.insert(S_Space, Set_Refinement[m]+1+m , np.asarray(New_States[2*m]), 0)
            S_Space = np.insert(S_Space, Set_Refinement[m]+1+m , np.asarray(New_States[2*m+1]), 0)           
            S_Space = np.delete(S_Space, Set_Refinement[m]+m, 0) 
            
            Low_Bound_Matrix = np.insert(Low_Bound_Matrix, Set_Refinement[m]+1+m, np.zeros(Low_Bound_Matrix.shape[1]), axis = 0)
            Upp_Bound_Matrix = np.insert(Upp_Bound_Matrix, Set_Refinement[m]+1+m, np.zeros(Upp_Bound_Matrix.shape[1]), axis = 0)
            Low_Bound_Matrix = np.insert(Low_Bound_Matrix, Set_Refinement[m]+1+m, np.zeros(Low_Bound_Matrix.shape[0]), axis = 1)
            Upp_Bound_Matrix = np.insert(Upp_Bound_Matrix, Set_Refinement[m]+1+m, np.zeros(Upp_Bound_Matrix.shape[0]), axis = 1)
              
            L_map.insert(Set_Refinement[m]+m+1, L_map[Set_Refinement[m]+m])
            Is_in_P = np.insert(Is_in_P, (Set_Refinement[m]+m+1)*Len_Automata, [0]*Len_Automata)
            
            (First_Loop, Reachable_States, Set_Refinement, Index_Stop, m) = Index_Update(First_Loop, Reachable_States, Set_Refinement, Index_Stop, m)
            (First_Loop2, Bridge_Transitions, Set_Refinement, Index_Stop2, m) = Index_Update(First_Loop2, Bridge_Transitions, Set_Refinement, Index_Stop2, m)
            
            for l in range(len(List_Inevitable_Non_A_BSCC)):
                for k in range(len(List_Inevitable_Non_A_BSCC[l])):
                    if Copy_List_Inevitable_Non_A_BSCC[l][k]/Len_Automata > Set_Refinement[m]:

                        List_Inevitable_Non_A_BSCC[l][k] += Len_Automata
                    

            for l in range(len(List_Inevitable_A_BSCC)):
                for k in range(len(List_Inevitable_A_BSCC[l])):
                    if Copy_List_Inevitable_A_BSCC[l][k]/Len_Automata > Set_Refinement[m]:
                        List_Inevitable_A_BSCC[l][k] += Len_Automata


            (Leaky_States_P_Accepting, Copy_Leaky_States_P_Accepting, Index_Stop3) = Index_Update_Product(Leaky_States_P_Accepting, Copy_Leaky_States_P_Accepting, Set_Refinement, Index_Stop3, Len_Automata, m)
            (Leaky_States_P_Non_Accepting, Copy_Leaky_States_P_Non_Accepting, Index_Stop4) = Index_Update_Product(Leaky_States_P_Non_Accepting, Copy_Leaky_States_P_Non_Accepting, Set_Refinement, Index_Stop4, Len_Automata, m)
            (Leaky_L_Accepting_Previous, Copy_Leaky_L_Accepting_Previous, Index_Stop5) = Index_Update_Product(Leaky_L_Accepting_Previous, Copy_Leaky_L_Accepting_Previous, Set_Refinement, Index_Stop5, Len_Automata, m)
            (Leaky_L_Non_Accepting_Previous, Copy_Leaky_L_Non_Accepting_Previous, Index_Stop6) = Index_Update_Product(Leaky_L_Non_Accepting_Previous, Copy_Leaky_L_Non_Accepting_Previous, Set_Refinement, Index_Stop6, Len_Automata, m)
            (Previous_A_BSCC, Copy_Previous_A_BSCC, Index_Stop7) = Index_Update_Product(Previous_A_BSCC, Copy_Previous_A_BSCC, Set_Refinement, Index_Stop7, Len_Automata, m)
            (Previous_Non_A_BSCC, Copy_Previous_Non_A_BSCC, Index_Stop8) = Index_Update_Product(Previous_Non_A_BSCC, Copy_Previous_Non_A_BSCC, Set_Refinement, Index_Stop8, Len_Automata, m)



        Is_Bridge_State = np.zeros(len(Bridge_Transitions))
        Is_Bridge_State = Is_Bridge_State.astype(int)
        
        for j in range(len(Bridge_Transitions)):
            if len(Bridge_Transitions[j]) != 0:
                Is_Bridge_State[j] = 1

        Is_New_State = np.zeros(S_Space.shape[0])
        Set_Refinement2 = []
               
        for j in range(len(Set_Refinement)):
            Is_New_State[Set_Refinement[j] + j] = 1
            Is_New_State[Set_Refinement[j] + j + 1] = 1
            Set_Refinement2.append(Set_Refinement[j]+j+1)
            Set_Refinement[j] = Set_Refinement[j] + j
            
        
        Reachable_Set = Reachable_Sets_Verification(S_Space)

         
        
        (Low_Bound_Matrix, Upp_Bound_Matrix, Reachable_States, Is_Bridge_State, Bridge_Transitions) = Probability_Interval_Computation_Refinement(Reachable_Set, S_Space, Set_Refinement, Set_Refinement2, Low_Bound_Matrix, Upp_Bound_Matrix, Is_New_State, Reachable_States, Is_Bridge_State, Bridge_Transitions)

        
        #Constructs the product between the IMC and the Automata
        (IA1_l, IA1_u, Is_Acc, Is_Non_Acc, Which_Acc_S, Which_Non_Acc_S, Product_Reachable_States, Product_Is_Bridge_State, Product_Bridge_Transitions, Init) = Build_Product_IMC(Low_Bound_Matrix, Upp_Bound_Matrix, Automaton, L_map, Automaton_Acc, Reachable_States, Is_Bridge_State, Bridge_Transitions)

        
        #Contains the interval of satisfactions for all states
        Success_Intervals = [[] for n in range(S_Space.shape[0])]
        
        #Contains the probability of reaching an accepting BSCC from all states in the product IMC
        Product_Intervals = [[] for n in range(IA1_l.shape[0])]
        
        
        #Case when the Automata has one Rabin Pair

        Leaky_States_L_Accepting = [] 
        Leaky_States_L_Non_Accepting = [] 
        First_Verif = 0


        (List_Largest_Non_A_BSCC, List_Largest_A_BSCC, List_Inevitable_Non_A_BSCC, List_Inevitable_A_BSCC, Is_In_L_A, Is_In_L_N_A, Which_A, Which_N_A, Bridge_A, Bridge_N_A, Graph, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Leaky_States_L_Accepting, Leaky_States_L_Non_Accepting) = Find_Largest_BSCCs_One_Pair(IA1_l, IA1_u, Automaton_Acc, Low_Bound_Matrix.shape[0], Is_Acc, Is_Non_Acc, Which_Acc_S, Which_Non_Acc_S, Product_Reachable_States, Product_Is_Bridge_State, Product_Bridge_Transitions, Leaky_States_L_Accepting, Leaky_States_L_Non_Accepting, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Is_in_P, List_Inevitable_A_BSCC, List_Inevitable_Non_A_BSCC, Previous_A_BSCC, Previous_Non_A_BSCC, First_Verif)

        (List_Largest_Winning_Components, Inevitable_Winning_Components, List_Largest_Losing_Components, Inevitable_Losing_Components, Largest_Winning_Components, Largest_Losing_Components,  Is_in_WC_L, Which_WC_L, Bridge_WC_L, Is_in_LC_L, Which_LC_L, Bridge_LC_L, Is_in_WC_P, Which_WC_P, Bridge_WC_P, Is_in_LC_P, Which_LC_P, Bridge_LC_P, Is_in_P, List_Inevitable_A_BSCC, List_Inevitable_Non_A_BSCC, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Leaky_L_Accepting_Previous, Leaky_L_Non_Accepting_Previous, Previous_A_BSCC, Previous_Non_A_BSCC) = Find_Winning_Losing_Components(IA1_l, IA1_u, List_Largest_Non_A_BSCC, List_Largest_A_BSCC, List_Inevitable_Non_A_BSCC, List_Inevitable_A_BSCC, Product_Reachable_States, Product_Is_Bridge_State, Product_Bridge_Transitions, Graph, Bridge_A, Bridge_N_A, Is_In_L_A, Is_In_L_N_A, Is_in_P, Leaky_States_L_Accepting, Leaky_States_L_Non_Accepting, Leaky_States_P_Accepting, Leaky_States_P_Non_Accepting, Leaky_L_Accepting_Previous, Leaky_L_Non_Accepting_Previous, Previous_A_BSCC, Previous_Non_A_BSCC)
        
    
        
        if len(Largest_Losing_Components) == 0: 
            #If the product IMC cannot create a non-accepting BSCC, then the probability 
            #of success is trivially one for all states.
            for i in range(len(Success_Intervals)):
                Success_Intervals[i].append(1.0)
                Success_Intervals[i].append(1.0)
                
        elif len(Largest_Winning_Components) == 0:
    
            #If the product IMC cannot create an accepting BSCC, then the probability 
            #of success is trivially zero for all states.
            for i in range(len(Success_Intervals)):
                Success_Intervals[i].append(0.0)
                Success_Intervals[i].append(0.0)
                
            for i in range(len(Product_Intervals)):
                Product_Intervals[i].append(0.0)
                Product_Intervals[i].append(0.0)    
                
        else:
            
            if len(Inevitable_Winning_Components) ==  0:
                Low_Bounds = [0.0 for j in range(S_Space.shape[0])]
                Low_Bounds_Prod = [0.0 for j in range(IA1_u.shape[0]) ]
                Worst_Markov_Chain = Random_Markov_Chain_IMC(IA1_u, IA1_l)
            else:  
                (Prob_Reach, Low_Bounds_Prod, Worst_Markov_Chain) = Reachability_Upper(IA1_l, IA1_u, Largest_Losing_Components, Inevitable_Winning_Components, Low_Bound_Matrix.shape[0], len(Automaton), Product_Reachable_States, Init)
                Low_Bounds = [i - j for i, j in zip([1.0]*len(Prob_Reach), Prob_Reach)]                    
                Low_Bounds_Prod = [i - j for i, j in zip([1.0]*len(Low_Bounds_Prod), Low_Bounds_Prod)]

            
            if len(Inevitable_Losing_Components) == 0:
                Upp_Bounds = [1.0 for j in range(S_Space.shape[0])]
                Upp_Bounds_Prod = [1.0 for j in range(IA1_u.shape[0]) ]
                Best_Markov_Chain = Random_Markov_Chain_IMC(IA1_u, IA1_l)
            else:
                (Upp_Bounds, Upp_Bounds_Prod, Best_Markov_Chain) = Reachability_Upper(IA1_l, IA1_u, Largest_Winning_Components, Inevitable_Losing_Components, Low_Bound_Matrix.shape[0], len(Automaton), Product_Reachable_States, Init)

                
            for i in range(len(Success_Intervals)):  

                Success_Intervals[i].append(Low_Bounds[i])
                Success_Intervals[i].append(Upp_Bounds[i])
            
            for i in range(len(Product_Intervals)):
                Product_Intervals[i].append(Low_Bounds_Prod[i])
                Product_Intervals[i].append(Upp_Bounds_Prod[i])

             
                               
        Y_States = []
        N_States = []
        M_States = []
        Max_I = 0.0
    
        if Ord == 1:
            ### Probability <=
            
            for i in range(len(Success_Intervals)): 
                if Success_Intervals[i][1] <= p_thresh:     
                    Y_States.append(i)
                    
                elif Success_Intervals[i][0] > p_thresh:       
                    N_States.append(i)
                    
                else:
                    Max_I = max(Max_I, Success_Intervals[i][1] - Success_Intervals[i][0])
                    M_States.append(i)
            
        elif Ord == 2: 
            ### Probability >=
            
            for i in range(len(Success_Intervals)):  
                if Success_Intervals[i][1] < p_thresh:     
                    N_States.append(i)
                    
                elif Success_Intervals[i][0] >= p_thresh:       
                    Y_States.append(i)
                    
                else:
                    Max_I = max(Max_I, Success_Intervals[i][1] - Success_Intervals[i][0])
                    M_States.append(i)
                    
            
        elif Ord == 3:   
            ### Probability <
        
            for i in range(len(Success_Intervals)):       
                if Success_Intervals[i][1] < p_thresh:     
                    Y_States.append(i)
                    
                elif Success_Intervals[i][0] >= p_thresh:       
                    N_States.append(i)
                    
                else:
                    
                    Max_I = max(Max_I, Success_Intervals[i][1] - Success_Intervals[i][0])
                    M_States.append(i)
        else:      
            ### Probability >
        
            for i in range(len(Success_Intervals)):     
                if Success_Intervals[i][1] <= p_thresh:     
                    N_States.append(i)
                    
                elif Success_Intervals[i][0] > p_thresh:       
                    Y_States.append(i)
                    
                else:
                    Max_I = max(Max_I, Success_Intervals[i][1] - Success_Intervals[i][0])
                    M_States.append(i)    



        V_Maybe = 0.0
        for i in range(len(M_States)):
            V_Maybe += (S_Space[M_States[i]][1][0] - S_Space[M_States[i]][0][0])* (S_Space[M_States[i]][1][1] - S_Space[M_States[i]][0][1])
        V_Maybe = V_Maybe/V_Total

        print 'Refinement Step', Tag
        
        print 'V_Maybe'
        print V_Maybe
        
        
        #Set Tag to number of desired refinement steps, otherwise comment line below
        if Tag == 5: break
   
    return (S_Space, Y_States, N_States, M_States, Success_Intervals)




def Uncertainty_Ranking_Path(Upper, Lower, Maybe, Success_Int, IA_u, IA_l, Is_in_WC_L, Which_WC_L, Bridge_WC_L, Is_in_LC_L, Which_LC_L, Bridge_LC_L, Is_in_WC_P, Which_WC_P, Bridge_WC_P, Is_in_LC_P, Which_LC_P, Bridge_LC_P, Is_in_P, S_Space, V_Total, Len_A):
    
    #Computes an Uncertainty Ranking based on the states satisfiability intervals
    
       
    PATH_LENGTH = (Upper.shape[0]/6.0) #Parameter determining the maximum length of accepting path that are being inspected
    Difference_Success = np.zeros(Upper.shape[0])
    Prob_Threshold = 0.001 #Threshold probability at which the current path is 'dropped'
    
    Reachable_Edges_Upper = [[] for x in range(Upper.shape[0])] #List all _ vertices from a given vertex
    Reachable_Check = np.zeros(Upper.shape[0])# This list keeps track of the vertices whose reachable sets have been computed
        
    for i in range(Difference_Success.shape[0]):
        Difference_Success[i] = Success_Int[i][1] - Success_Int[i][0]
    
    Weights = np.zeros(Upper.shape[0])
    
    for j in range(len(Maybe)):
        
        V_Weight = 1.0
        
        Check = 0
        Bridge_Check = np.zeros(len(Weights)) #Checks whether a score has previously been added to a given state
        
        if Is_in_WC_L[Maybe[j]] == 1:
            Current_A = Which_WC_L[Maybe[j]]
            Max_Probability_Gain = V_Weight*Difference_Success[Maybe[j]]
            for q in range(len(Current_A)):
                for i in range(len(Bridge_WC_L[Current_A[q][0]][Current_A[q][1]])):                   
                    if  Bridge_Check[Bridge_WC_L[Current_A[q][0]][Current_A[q][1]][i]] == 0:
                        Bridge_Check[Bridge_WC_L[Current_A[q][0]][Current_A[q][1]][i]] = 1
                        Weights[Bridge_WC_L[Current_A[q][0]][Current_A[q][1]][i]] += Max_Probability_Gain                      
            Check = 1

        if Is_in_WC_P[Maybe[j]] == 1:
            Current_A = Which_WC_P[Maybe[j]]
            Max_Probability_Gain = V_Weight*Difference_Success[Maybe[j]]
            for q in range(len(Current_A)):
                for i in range(len(Bridge_WC_P[Current_A[q][0]][Current_A[q][1]])):
                    if  Bridge_Check[Bridge_WC_P[Current_A[q][0]][Current_A[q][1]][i]] == 0:
                        Bridge_Check[Bridge_WC_P[Current_A[q][0]][Current_A[q][1]][i]] = 1
                        Weights[Bridge_WC_P[Current_A[q][0]][Current_A[q][1]][i]] += Max_Probability_Gain                      
            Check = 1        
        
        if Is_in_LC_L[Maybe[j]] == 1:
            Current_N_A = Which_LC_L[Maybe[j]]
            Max_Probability_Gain = V_Weight*Difference_Success[Maybe[j]]
            for q in range(len(Current_N_A)):
                for i in range(len(Bridge_LC_L[Current_N_A[q][0]][Current_N_A[q][1]])):
                    if  Bridge_Check[Bridge_LC_L[Current_N_A[q][0]][Current_N_A[q][1]][i]] == 0:
                        Bridge_Check[Bridge_LC_L[Current_N_A[q][0]][Current_N_A[q][1]][i]] = 1                    
                        Weights[Bridge_LC_L[Current_N_A[q][0]][Current_N_A[q][1]][i]] += Max_Probability_Gain                      
            Check = 1
        
        if Is_in_LC_P[Maybe[j]] == 1:
            Current_N_A = Which_LC_P[Maybe[j]]
            Max_Probability_Gain = V_Weight*Difference_Success[Maybe[j]]
            for q in range(len(Current_N_A)):
                for i in range(len(Bridge_LC_P[Current_N_A[q][0]][Current_N_A[q][1]])):
                    if  Bridge_Check[Bridge_LC_P[Current_N_A[q][0]][Current_N_A[q][1]][i]] == 0:
                        Bridge_Check[Bridge_LC_P[Current_N_A[q][0]][Current_N_A[q][1]][i]] = 1
                        Weights[Bridge_LC_P[Current_N_A[q][0]][Current_N_A[q][1]][i]] += Max_Probability_Gain                      
            Check = 1 
        
        if Check == 1:
            continue
                       
        Visited_Edges = []
        Current_Path = [Maybe[j]]
        Current_Path_Is_Visited = np.zeros(Upper.shape[0]) #Checks whether a state has already been visited along a path or not
        Current_Path_Is_Visited[Maybe[j]] = 1
        Current_Path_Probability = 1.0 #Current_Path_Best_Case_Probability
        Current_Path_Probability_Lower = 1.0
        
        while (len(Current_Path) != 0):
            
            
            if Current_Path_Probability < Prob_Threshold:
                Current_Path_Probability = Current_Path_Probability/(Upper[Current_Path[-2],Current_Path[-1]])
                Current_Path_Is_Visited[Current_Path[-1]] = 0
                Current_Path.pop()                
                continue
            #This if statement considers the case where the current state belongs to an Inevitable A BSCC
            if  len(Current_Path) != 1:                    
                if Is_in_P[Current_Path[-1]] == 1:
                    Current_Path_Probability = Current_Path_Probability/(Upper[Current_Path[-2],Current_Path[-1]])

                    Current_Path_Is_Visited[Current_Path[-1]] = 0
                    Current_Path.pop()
                    continue
            
            Check = 0
            Bridge_Check = np.zeros(len(Weights)) 
                
            if Is_in_LC_L[Current_Path[-1]] == 1:
                
                Max_Probability_Gain = V_Weight*Current_Path_Probability*Difference_Success[Current_Path[-1]]

                Current_N_A = Which_LC_L[Current_Path[-1]]
                
                for q in range(len(Current_N_A)):
                    for i in range(len(Bridge_LC_L[Current_N_A[q][0]][Current_N_A[q][1]])):
                        if Bridge_Check[Bridge_LC_L[Current_N_A[q][0]][Current_N_A[q][1]][i]] == 0:
                            Bridge_Check[Bridge_LC_L[Current_N_A[q][0]][Current_N_A[q][1]][i]] = 1
                            Weights[Bridge_LC_L[Current_N_A[q][0]][Current_N_A[q][1]][i]] += Max_Probability_Gain
                    
                Check = 1


            if Is_in_LC_P[Current_Path[-1]] == 1:
                
                Max_Probability_Gain = V_Weight*Current_Path_Probability*Difference_Success[Current_Path[-1]]
        
                Current_N_A = Which_LC_P[Current_Path[-1]]

                for q in range(len(Current_N_A)):
                    for i in range(len(Bridge_LC_P[Current_N_A[q][0]][Current_N_A[q][1]])):
                        if Bridge_Check[Bridge_LC_P[Current_N_A[q][0]][Current_N_A[q][1]][i]] == 0:
                            Bridge_Check[Bridge_LC_P[Current_N_A[q][0]][Current_N_A[q][1]][i]] = 1                        
                            Weights[Bridge_LC_P[Current_N_A[q][0]][Current_N_A[q][1]][i]] += Max_Probability_Gain
                    
                Check = 1
                                              
            if Is_in_WC_L[Current_Path[-1]] == 1:
                Max_Probability_Gain = V_Weight*Current_Path_Probability*Difference_Success[Current_Path[-1]]
                
                Current_A = Which_WC_L[Current_Path[-1]]

                for q in range(len(Current_A)):                   
                    for i in range(len(Bridge_WC_L[Current_A[q][0]][Current_A[q][1]])):
                        if Bridge_Check[Bridge_WC_L[Current_A[q][0]][Current_A[q][1]][i]] == 0:
                            Bridge_Check[Bridge_WC_L[Current_A[q][0]][Current_A[q][1]][i]] = 1
                            Weights[Bridge_WC_L[Current_A[q][0]][Current_A[q][1]][i]] += Max_Probability_Gain
                    
                Check = 1
               

            if Is_in_WC_P[Current_Path[-1]] == 1:
                Max_Probability_Gain = V_Weight*Current_Path_Probability*Difference_Success[Current_Path[-1]]
                
                Current_A = Which_WC_P[Current_Path[-1]]

                for q in range(len(Current_A)): 
                    for i in range(len(Bridge_WC_P[Current_A[q][0]][Current_A[q][1]])):
                        if Bridge_Check[Bridge_WC_P[Current_A[q][0]][Current_A[q][1]][i]] == 0:
                            Bridge_Check[Bridge_WC_P[Current_A[q][0]][Current_A[q][1]][i]] = 1                        
                            Weights[Bridge_WC_P[Current_A[q][0]][Current_A[q][1]][i]] += Max_Probability_Gain                    
                Check = 1

            if Check == 1:
                Current_Path_Probability = Current_Path_Probability/(Upper[Current_Path[-2],Current_Path[-1]])
                Current_Path_Is_Visited[Current_Path[-1]] = 0
                Current_Path.pop()
                continue                   
            
            if len(Visited_Edges) < len(Current_Path):
                Visited_Edges.append([])
                                   
            if Reachable_Check[Current_Path[-1]] == 0:
                for i in range(Upper.shape[1]):
                    if Upper[Current_Path[-1],i] > 0:
                        Reachable_Edges_Upper[Current_Path[-1]].append(i)                        
                Reachable_Check[Current_Path[-1]] = 1
                        
                
            if  len(Visited_Edges[-1]) == 0:  
                for i in range(len(Reachable_Edges_Upper[Current_Path[-1]])):                        
                    Weights[Current_Path[-1]] += V_Weight*Current_Path_Probability*(Upper[Current_Path[-1], Reachable_Edges_Upper[Current_Path[-1]][i]]*Success_Int[Reachable_Edges_Upper[Current_Path[-1]][i]][1] - Lower[Current_Path[-1], Reachable_Edges_Upper[Current_Path[-1]][i]]*Success_Int[Reachable_Edges_Upper[Current_Path[-1]][i]][0])    
                    if  Current_Path_Is_Visited[Reachable_Edges_Upper[Current_Path[-1]][i]] == 1:
                        a = Reachable_Edges_Upper[Current_Path[-1]][i]
                        Visited_Edges[-1].append(a)
                        Reachable_Edges_Upper[Current_Path[-1]].pop(i)
                        Reachable_Edges_Upper[Current_Path[-1]].insert(0,a)
                    

            if (len(Visited_Edges[-1]) == len(Reachable_Edges_Upper[Current_Path[-1]])):
                if len(Current_Path) != 1:
                    Current_Path_Probability = Current_Path_Probability/(Upper[Current_Path[-2],Current_Path[-1]])
                    
                    Current_Path_Is_Visited[Current_Path[-1]] = 0
                    Current_Path.pop()
                    Visited_Edges.pop()
                else:
                    Current_Path_Is_Visited[Current_Path[-1]] = 0
                    Current_Path.pop()
                                
            else:
                
                #Add some condition for which an edge does not need to be visited
                Current_Path_Probability = Current_Path_Probability*Upper[Current_Path[-1],Reachable_Edges_Upper[Current_Path[-1]][len(Visited_Edges[-1])]]
                                                      
                Current_Path.append(Reachable_Edges_Upper[Current_Path[-1]][len(Visited_Edges[-1])])
                Current_Path_Is_Visited[Current_Path[-1]] = 1
                Visited_Edges[-1].append(Reachable_Edges_Upper[Current_Path[-2]][len(Visited_Edges[-1])])
                
    return Weights



def Raw_Refinement(State, Space): 
       
    New_St = []       
    for i in range(len(State)):

       a1 = Space[State[i]][1][0] - Space[State[i]][0][0]
       a2 = Space[State[i]][1][1] - Space[State[i]][0][1]
    
       if a1 > a2:
                               
           New_St.append([(Space[State[i]][0][0],Space[State[i]][0][1]),((Space[State[i]][1][0] + Space[State[i]][0][0])/2.0,Space[State[i]][1][1])])
           New_St.append([((Space[State[i]][1][0] + Space[State[i]][0][0])/2.0 , Space[State[i]][0][1]),(Space[State[i]][1][0],Space[State[i]][1][1])])
  
       else:
           
           New_St.append([(Space[State[i]][0][0] , (Space[State[i]][1][1]+Space[State[i]][0][1])/2.0),(Space[State[i]][1][0],Space[State[i]][1][1])])
           New_St.append([(Space[State[i]][0][0] , Space[State[i]][0][1]),(Space[State[i]][1][0],(Space[State[i]][1][1]+Space[State[i]][0][1])/2.0)])
   
    return New_St



def Probability_Interval_Computation_Refinement(R_Set, Target_Set, Set_Ref1, Set_Ref2, L_Bound_Matrix, U_Bound_Matrix, Is_New, Reachable_States, Is_Bridge_State, Bridge_Transitions):
    
            
    Z1 = (erf(Semi_Width_1/sigma1)/sqrt(2)) - (erf(-Semi_Width_1/sigma1)/sqrt(2))
    Z2 = (erf(Semi_Width_2/sigma2)/sqrt(2)) - (erf(-Semi_Width_2/sigma2)/sqrt(2))

    
    for j in range(len(R_Set)):
        
        r0 = R_Set[j][0][0]
        r1 = R_Set[j][1][0]
        r2 = R_Set[j][0][1]
        r3 = R_Set[j][1][1]
        
        
            
        if Is_New[j] == 1:
            
            for h in range(Target_Set.shape[0]):
        
                q0 = Target_Set[h][0][0]
                q1 = Target_Set[h][1][0]
                q2 = Target_Set[h][0][1]
                q3 = Target_Set[h][1][1]
                
                if q0 == LOW_1 and r0 + mu1 - Semi_Width_1 < LOW_1:
                    q0 = r0 + mu1 - Semi_Width_1
                    
                if q1 == UP_1 and r1 + mu1 + Semi_Width_1 > UP_1:
                    q1 = r1 + mu1 + Semi_Width_1
                    
                if q2 == LOW_2 and r2 + mu2 - Semi_Width_2 < LOW_2:
                    q2 = r2 + mu2 - Semi_Width_2
                    
                if q3 == UP_2 and r3 + mu2 + Semi_Width_2 > UP_2:
                    q3 = r3 + mu2 + Semi_Width_2
                
                if (r0 >= q1 + Semi_Width_1 - mu1) or (r1 <= q0 - Semi_Width_1 - mu1) or (r2 >= q3 + Semi_Width_2 - mu2) or (r3 <= q2 - Semi_Width_2 - mu2):
                    L_Bound_Matrix[j][h] = 0.0
                    U_Bound_Matrix[j][h] = 0.0
                    continue

                bisect.insort(Reachable_States[j], h)                                                                                                
                
                a1_Opt = ((q0 + q1)/2.0) - mu1
                a2_Opt = ((q2 + q3)/2.0) - mu2
                                
                
                if (r1 < a1_Opt): 
                    a1_Max = r1
                    a1_Min = r0
                elif(r0 > a1_Opt): 
                    a1_Max = r0
                    a1_Min = r1
                else: 
                    a1_Max = a1_Opt       
                    if (a1_Opt <= (r1+r0)/2.0):
                        a1_Min = r1
                    else:
                        a1_Min = r0
                                                        
                if (r2 > a2_Opt): 
                    a2_Max = r2
                    a2_Min = r3
                elif(r3 < a2_Opt): 
                    a2_Max = r3
                    a2_Min = r2
                else: 
                    a2_Max = a2_Opt
                    if (a2_Opt <= (r2+r3)/2.0):
                        a2_Min = r3
                    else:
                        a2_Min = r2
                                                                                           
            
                if a1_Max + mu1 - Semi_Width_1  > q0 and a1_Max + mu1 + Semi_Width_1 < q1 and a2_Max + mu2 - Semi_Width_2 > q2 and a2_Max + mu2 + Semi_Width_2 < q3:
                    H = 1.0
                else:
                    
                    if q0 < a1_Max + mu1 - Semi_Width_1:
                        b0 = a1_Max + mu1 - Semi_Width_1
                    else:
                        b0 = q0
                        
                    if q1 > a1_Max + mu1 + Semi_Width_1:
                        b1 = a1_Max + mu1 + Semi_Width_1
                    else:
                        b1 = q1
                        
                    if q2 < a2_Max + mu2 - Semi_Width_2:
                        b2 = a2_Max + mu2 - Semi_Width_2
                    else:
                        b2 = q2
                        
                    if q3 > a2_Max + mu2 + Semi_Width_2:
                        b3 = a2_Max + mu2 + Semi_Width_2
                    else:
                        b3 = q3    
                        
                    
                    H = ( ( (erf((b1 - a1_Max - mu1)/sigma1)/sqrt(2)) - (erf((b0 - a1_Max - mu1)/sigma1)/sqrt(2)) ) / (Z1)) * ( (erf((b3 - a2_Max - mu2)/sigma2)/sqrt(2)) - (erf((b2 - a2_Max - mu2)/sigma2)/sqrt(2)) ) / ( Z2 )
            
                    if H > 1:
                        H = 1.0
                        
                    
                    
                if (a1_Min + mu1 + Semi_Width_1 <= q0) or (a1_Min + mu1 - Semi_Width_1 >= q1) or (a2_Min + mu2 + Semi_Width_2 <= q2) or (a2_Min + mu2 - Semi_Width_2 >= q3):                   
                    Is_Bridge_State[j] = 1
                    bisect.insort(Bridge_Transitions[j],h)
                    L_Bound_Matrix[j][h] = 0.0
                    U_Bound_Matrix[j][h] = H
                    continue                
                
                
                else:
                    
                    if q0 < a1_Min + mu1 - Semi_Width_1:
                        b0 = a1_Min + mu1 - Semi_Width_1
                    else:
                        b0 = q0
                        
                    if q1 > a1_Min + mu1 + Semi_Width_1:
                        b1 = a1_Min + mu1 + Semi_Width_1
                    else:
                        b1 = q1
                        
                    if q2 < a2_Min + mu2 - Semi_Width_2:
                        b2 = a2_Min + mu2 - Semi_Width_2
                    else:
                        b2 = q2
                        
                    if q3 > a2_Min + mu2 + Semi_Width_2:
                        b3 = a2_Min + mu2 + Semi_Width_2
                    else:
                        b3 = q3   
                    

                    
                    L = ( ( (erf((b1 - a1_Min - mu1)/sigma1)/sqrt(2)) - (erf((b0 - a1_Min - mu1)/sigma1)/sqrt(2)) ) / (Z1)) * ( (erf((b3 - a2_Min - mu2)/sigma2)/sqrt(2)) - (erf((b2 - a2_Min - mu2)/sigma2)/sqrt(2)) ) / ( Z2 )
            
            
                    if L < 0:
                        L = 0.0
                    
                L_Bound_Matrix[j][h] = L
                U_Bound_Matrix[j][h] = H

        else:
            
            
            
            for h in range(len(Set_Ref1)):
 
                q0 = Target_Set[Set_Ref1[h]][0][0]
                q1 = Target_Set[Set_Ref1[h]][1][0]
                q2 = Target_Set[Set_Ref1[h]][0][1]
                q3 = Target_Set[Set_Ref1[h]][1][1]
                

                if q0 == LOW_1 and r0 + mu1 - Semi_Width_1 < LOW_1:
                    q0 = r0 + mu1 - Semi_Width_1
                    
                if q1 == UP_1 and r1 + mu1 + Semi_Width_1 > UP_1:
                    q1 = r0 + mu1 + Semi_Width_1
                    
                if q2 == LOW_2 and r2 + mu2 - Semi_Width_2 < LOW_2:
                    q2 = r2 + mu2 - Semi_Width_2
                    
                if q3 == UP_2 and r3 + mu2 + Semi_Width_2 > UP_2:
                    q3 = r3 + mu2 + Semi_Width_2                      
                
                if (r0 >= q1 + Semi_Width_1 - mu1) or (r1 <= q0 - Semi_Width_1 - mu1) or (r2 >= q3 + Semi_Width_2 - mu2) or (r3 <= q2 - Semi_Width_2 - mu2):
                    L_Bound_Matrix[j][Set_Ref1[h]] = 0.0
                    U_Bound_Matrix[j][Set_Ref1[h]] = 0.0
                    continue 

                bisect.insort(Reachable_States[j],Set_Ref1[h])                                                                                               
                
                a1_Opt = ((q0 + q1)/2.0) - mu1
                a2_Opt = ((q2 + q3)/2.0) - mu2
                                
                
                if (r1 < a1_Opt): 
                    a1_Max = r1
                    a1_Min = r0
                elif(r0 > a1_Opt): 
                    a1_Max = r0
                    a1_Min = r1
                else: 
                    a1_Max = a1_Opt       
                    if (a1_Opt <= (r1+r0)/2.0):
                        a1_Min = r1
                    else:
                        a1_Min = r0
                                                        
                if (r2 > a2_Opt): 
                    a2_Max = r2
                    a2_Min = r3
                elif(r3 < a2_Opt): 
                    a2_Max = r3
                    a2_Min = r2
                else: 
                    a2_Max = a2_Opt
                    if (a2_Opt <= (r2+r3)/2.0):
                        a2_Min = r3
                    else:
                        a2_Min = r2
                                                                                           
            
                if a1_Max + mu1 - Semi_Width_1  > q0 and a1_Max + mu1 + Semi_Width_1 < q1 and a2_Max + mu2 - Semi_Width_2 > q2 and a2_Max + mu2 + Semi_Width_2 < q3:
                    H = 1.0
                else:
                    
                    if q0 < a1_Max + mu1 - Semi_Width_1:
                        b0 = a1_Max + mu1 - Semi_Width_1
                    else:
                        b0 = q0
                        
                    if q1 > a1_Max + mu1 + Semi_Width_1:
                        b1 = a1_Max + mu1 + Semi_Width_1
                    else:
                        b1 = q1
                        
                    if q2 < a2_Max + mu2 - Semi_Width_2:
                        b2 = a2_Max + mu2 - Semi_Width_2
                    else:
                        b2 = q2
                        
                    if q3 > a2_Max + mu2 + Semi_Width_2:
                        b3 = a2_Max + mu2 + Semi_Width_2
                    else:
                        b3 = q3    
                        
                    
                    H = ( ( (erf((b1 - a1_Max - mu1)/sigma1)/sqrt(2)) - (erf((b0 - a1_Max - mu1)/sigma1)/sqrt(2)) ) / (Z1)) * ( (erf((b3 - a2_Max - mu2)/sigma2)/sqrt(2)) - (erf((b2 - a2_Max - mu2)/sigma2)/sqrt(2)) ) / ( Z2 )
            
                    if H > 1:
                        H = 1.0
                        
                    
                    
                if (a1_Min + mu1 + Semi_Width_1 <= q0) or (a1_Min + mu1 - Semi_Width_1 >= q1) or (a2_Min + mu2 + Semi_Width_2 <= q2) or (a2_Min + mu2 - Semi_Width_2 >= q3):
                    Is_Bridge_State[j] = 1
                    bisect.insort(Bridge_Transitions[j],Set_Ref1[h])                   
                    L_Bound_Matrix[j][Set_Ref1[h]] = 0.0
                    U_Bound_Matrix[j][Set_Ref1[h]] = H
                    continue

                else:
                    
                    if q0 < a1_Min + mu1 - Semi_Width_1:
                        b0 = a1_Min + mu1 - Semi_Width_1
                    else:
                        b0 = q0
                        
                    if q1 > a1_Min + mu1 + Semi_Width_1:
                        b1 = a1_Min + mu1 + Semi_Width_1
                    else:
                        b1 = q1
                        
                    if q2 < a2_Min + mu2 - Semi_Width_2:
                        b2 = a2_Min + mu2 - Semi_Width_2
                    else:
                        b2 = q2
                        
                    if q3 > a2_Min + mu2 + Semi_Width_2:
                        b3 = a2_Min + mu2 + Semi_Width_2
                    else:
                        b3 = q3   
                    
                    
                    
                    L = ( ( (erf((b1 - a1_Min - mu1)/sigma1)/sqrt(2)) - (erf((b0 - a1_Min - mu1)/sigma1)/sqrt(2)) ) / (Z1)) * ( (erf((b3 - a2_Min - mu2)/sigma2)/sqrt(2)) - (erf((b2 - a2_Min - mu2)/sigma2)/sqrt(2)) ) / ( Z2 )
            
            
                    if L < 0:
                        L = 0.0
                    
                    
                L_Bound_Matrix[j][Set_Ref1[h]] = L
                U_Bound_Matrix[j][Set_Ref1[h]] = H
                
                
                
                
            for h in range(len(Set_Ref2)):
                

                q0 = Target_Set[Set_Ref2[h]][0][0]
                q1 = Target_Set[Set_Ref2[h]][1][0]
                q2 = Target_Set[Set_Ref2[h]][0][1]
                q3 = Target_Set[Set_Ref2[h]][1][1]
                
                if q0 == LOW_1 and r0 + mu1 - Semi_Width_1 < LOW_1:
                    q0 = r0 + mu1 - Semi_Width_1
                    
                if q1 == UP_1 and r1 + mu1 + Semi_Width_1 > UP_1:
                    q1 = r0 + mu1 + Semi_Width_1
                    
                if q2 == LOW_2 and r2 + mu2 - Semi_Width_2 < LOW_2:
                    q2 = r2 + mu2 - Semi_Width_2
                    
                if q3 == UP_2 and r3 + mu2 + Semi_Width_2 > UP_2:
                    q3 = r3 + mu2 + Semi_Width_2  
                    
                
                if (r0 >= q1 + Semi_Width_1 - mu1) or (r1 <= q0 - Semi_Width_1 - mu1) or (r2 >= q3 + Semi_Width_2 - mu2) or (r3 <= q2 - Semi_Width_2 - mu2):
                    L_Bound_Matrix[j][Set_Ref2[h]] = 0.0
                    U_Bound_Matrix[j][Set_Ref2[h]] = 0.0
                    continue                                                                                                
                
                bisect.insort(Reachable_States[j], Set_Ref2[h])
                
                a1_Opt = ((q0 + q1)/2.0) - mu1
                a2_Opt = ((q2 + q3)/2.0) - mu2
                                
                
                if (r1 < a1_Opt): 
                    a1_Max = r1
                    a1_Min = r0
                elif(r0 > a1_Opt): 
                    a1_Max = r0
                    a1_Min = r1
                else: 
                    a1_Max = a1_Opt       
                    if (a1_Opt <= (r1+r0)/2.0):
                        a1_Min = r1
                    else:
                        a1_Min = r0
                                                        
                if (r2 > a2_Opt): 
                    a2_Max = r2
                    a2_Min = r3
                elif(r3 < a2_Opt): 
                    a2_Max = r3
                    a2_Min = r2
                else: 
                    a2_Max = a2_Opt
                    if (a2_Opt <= (r2+r3)/2.0):
                        a2_Min = r3
                    else:
                        a2_Min = r2
                                                                                           
            
                if a1_Max + mu1 - Semi_Width_1  > q0 and a1_Max + mu1 + Semi_Width_1 < q1 and a2_Max + mu2 - Semi_Width_2 > q2 and a2_Max + mu2 + Semi_Width_2 < q3:
                    H = 1.0
                else:
                    
                    if q0 < a1_Max + mu1 - Semi_Width_1:
                        b0 = a1_Max + mu1 - Semi_Width_1
                    else:
                        b0 = q0
                        
                    if q1 > a1_Max + mu1 + Semi_Width_1:
                        b1 = a1_Max + mu1 + Semi_Width_1
                    else:
                        b1 = q1
                        
                    if q2 < a2_Max + mu2 - Semi_Width_2:
                        b2 = a2_Max + mu2 - Semi_Width_2
                    else:
                        b2 = q2
                        
                    if q3 > a2_Max + mu2 + Semi_Width_2:
                        b3 = a2_Max + mu2 + Semi_Width_2
                    else:
                        b3 = q3    
                        
                    
                    H = ( ( (erf((b1 - a1_Max - mu1)/sigma1)/sqrt(2)) - (erf((b0 - a1_Max - mu1)/sigma1)/sqrt(2)) ) / (Z1)) * ( (erf((b3 - a2_Max - mu2)/sigma2)/sqrt(2)) - (erf((b2 - a2_Max - mu2)/sigma2)/sqrt(2)) ) / ( Z2 )
            
                    if H > 1:
                        H = 1.0
                        
                    
                    
                if (a1_Min + mu1 + Semi_Width_1 <= q0) or (a1_Min + mu1 - Semi_Width_1 >= q1) or (a2_Min + mu2 + Semi_Width_2 <= q2) or (a2_Min + mu2 - Semi_Width_2 >= q3):
                    Is_Bridge_State[j] = 1
                    bisect.insort(Bridge_Transitions[j],Set_Ref2[h])   
                    L_Bound_Matrix[j][Set_Ref2[h]] = 0.0
                    U_Bound_Matrix[j][Set_Ref2[h]] = H
                    continue

                else:
                    
                    if q0 < a1_Min + mu1 - Semi_Width_1:
                        b0 = a1_Min + mu1 - Semi_Width_1
                    else:
                        b0 = q0
                        
                    if q1 > a1_Min + mu1 + Semi_Width_1:
                        b1 = a1_Min + mu1 + Semi_Width_1
                    else:
                        b1 = q1
                        
                    if q2 < a2_Min + mu2 - Semi_Width_2:
                        b2 = a2_Min + mu2 - Semi_Width_2
                    else:
                        b2 = q2
                        
                    if q3 > a2_Min + mu2 + Semi_Width_2:
                        b3 = a2_Min + mu2 + Semi_Width_2
                    else:
                        b3 = q3   
                    
                    
                    
                    L = ( ( (erf((b1 - a1_Min - mu1)/sigma1)/sqrt(2)) - (erf((b0 - a1_Min - mu1)/sigma1)/sqrt(2)) ) / (Z1)) * ( (erf((b3 - a2_Min - mu2)/sigma2)/sqrt(2)) - (erf((b2 - a2_Min - mu2)/sigma2)/sqrt(2)) ) / ( Z2 )
            
            
                    if L < 0:
                        L = 0.0
                    
                    
                L_Bound_Matrix[j][Set_Ref2[h]] = L
                U_Bound_Matrix[j][Set_Ref2[h]] = H
                              
                
    
    return (L_Bound_Matrix, U_Bound_Matrix, Reachable_States, Is_Bridge_State, Bridge_Transitions)


def Random_Markov_Chain_IMC(Upp, Low):
    
    Random_Chain = np.zeros((Upp.shape[0], Upp.shape[0]))
    
    for i in range(Upp.shape[0]):
        Sum = 0.0
        for j in range(Upp.shape[0]):
            Random_Chain[i,j] = Low[i,j]
            Sum += Low[i,j]
            
        for j in range(Upp.shape[0]):
            if Sum >= 1.0: break
            Random_Chain[i,j] += min(Upp[i,j] - Low[i,j], 1.0-Sum)
            Sum += (Random_Chain[i,j] - Low[i,j])
            
    return Random_Chain 



def Index_Update(First_Loop, Reachable_States, Set_Refinement, Index_Stop, m):

    if First_Loop == 1:
        First_Loop = 0
        y = 0
        Count = 0
        Checked_All_Set = 0
        while(y+Count != len(Reachable_States)):
            
            if Checked_All_Set == 0:
                if y+Count == (Set_Refinement[Count])+Count:
                    Reachable_States[y+Count] = []
                    Reachable_States.insert(y+Count,[])
                    Index_Stop[y+Count] = 0
                    Index_Stop[y+Count+1] = 0
                    Count += 1
                    if Count == len(Set_Refinement):
                        Checked_All_Set = 1
                else:
                    End_List = 1
                    for k in range(len(Reachable_States[y+Count])):
                        if(Reachable_States[y+Count][k] == Set_Refinement[0]):
                            del Reachable_States[y+Count][k]
                            Index_Stop[y+Count] = k
                            End_List = 0
                            break
                        elif(Reachable_States[y+Count][k] > Set_Refinement[0]):                                
                            Index_Stop[y+Count] = k
                            End_List = 0
                            break
                                                
                    if End_List == 1:
                        Index_Stop[y+Count] = len(Reachable_States[y+Count]) 
                        
                    
            else:
                End_List = 1
                for k in range(len(Reachable_States[y+Count])):
                    if(Reachable_States[y+Count][k] == Set_Refinement[0]):
                        del Reachable_States[y+Count][k]
                        Index_Stop[y+Count] = k
                        End_List = 0
                        break
                    elif(Reachable_States[y+Count][k] > Set_Refinement[0]):                                
                        Index_Stop[y+Count] = k
                        End_List = 0
                        break

                if End_List == 1:
                    Index_Stop[y+Count] = len(Reachable_States[y+Count])                    
            
            y += 1
                        
        if (len(Set_Refinement)== 1):
            for k in range(len(Reachable_States)):
                for n in range(Index_Stop[k], len(Reachable_States[k])):
                    Reachable_States[k][n] += 1
                     
    else:
        
        for k in range(len(Reachable_States)):
            deleted_state = 0
            for n in range(Index_Stop[k], len(Reachable_States[k])):                                                  
                if (Reachable_States[k][n-deleted_state] == Set_Refinement[m]):
                    if (m < len(Set_Refinement) - 1): 
                        del Reachable_States[k][n]
                        Index_Stop[k] = n
                        deleted_state = 1
                        break
                    else:
                        del Reachable_States[k][n]
                        deleted_state = 1
                                                      
                elif (Reachable_States[k][n-deleted_state]  > Set_Refinement[m]):
                    if (m < len(Set_Refinement) - 1):
                        Index_Stop[k] = n
                        break
                    else:
                            Reachable_States[k][n-deleted_state] += m + 1 
                else:
                    Reachable_States[k][n] += m
                    if n == len(Reachable_States[k]) - 1:
                        Index_Stop[k] = len(Reachable_States[k])    

    return First_Loop, Reachable_States, Set_Refinement, Index_Stop, m


def Index_Update_Product(Leaky_States_P_Accepting, Copy_Leaky_States_P_Accepting, Set_Refinement, Index_Stop3, Len_Automata, m):

    for l in range(len(Leaky_States_P_Accepting)):
        Number_Added = 0
        Check = 0
        for k in range(Index_Stop3[l], len(Leaky_States_P_Accepting[l])):
            if Copy_Leaky_States_P_Accepting[l][k+Number_Added]/Len_Automata > Set_Refinement[m]:
                if Check == 0:
                    Index_Stop3[l] = k
                    Check = 1
                Leaky_States_P_Accepting[l][k+Number_Added] += Len_Automata
            elif Copy_Leaky_States_P_Accepting[l][k+Number_Added]/Len_Automata == Set_Refinement[m]:
                if Check == 0:
                    Index_Stop3[l] = k
                    Check = 1 
                    
                Copy_Leaky_States_P_Accepting[l].insert(k+Number_Added+1,Copy_Leaky_States_P_Accepting[l][k+Number_Added])                           
                Leaky_States_P_Accepting[l].insert(k+Number_Added+1, Leaky_States_P_Accepting[l][k+Number_Added] + Len_Automata)
                Number_Added += 1

        Leaky_States_P_Accepting[l].sort()

    return Leaky_States_P_Accepting, Copy_Leaky_States_P_Accepting, Index_Stop3
