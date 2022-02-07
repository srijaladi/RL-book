# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Mapping
import numpy as np
import itertools
from rl.distribution import (Categorical, Distribution, FiniteDistribution,
                             SampledDistribution, Constant)
from rl.markov_process import MarkovProcess, NonTerminal, State, Terminal
from rl.gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func
from rl.markov_process import FiniteMarkovProcess, MarkovRewardProcess, FiniteMarkovRewardProcess
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        FiniteMarkovRewardProcess)
from rl.policy import FinitePolicy, FiniteDeterministicPolicy
from rl.chapter2.stock_price_simulations import\
    plot_single_trace_all_processes
from rl.chapter2.stock_price_simulations import\
    plot_distribution_at_time_all_processes
import matplotlib         
from matplotlib import pyplot as plt
from typing import (Callable, Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, TypeVar, Set)
import itertools
from itertools import combinations
from rl.dynamic_programming import evaluate_mrp_result
from rl.dynamic_programming import policy_iteration_result
from rl.dynamic_programming import value_iteration_result
from pprint import pprint


@dataclass(frozen=True)
class petalState:
    i: int

PetalActionMapping = Mapping[
    petalState,
    Mapping[int, Categorical[Tuple[petalState, float]]]
]    
    
class frogMoveMDP(FiniteMarkovDecisionProcess[petalState, int]):
    def __init__(self, n: int):
        #n represents the nth (max) petal
        self.n = n
        #numPetals is the total number of petals
        self.numPetals = n + 1
        
        super().__init__(self.get_action_transition_reward_map())
        
    def get_action_transition_reward_map(self) -> PetalActionMapping:
        
        #Mapping
        overall_state_action_map: Dict[
            petalState, Dict[int, Categorical[Tuple[petalState, float]]]] = {}        
        
        for currPetal in range(1, self.n):
            
            #The current state using the current petal's int
            curr_state = petalState(currPetal)
            
            #Maps each of the possible actions to respective categorical 
            #transition/reward probabilities
            curr_state_actions: Dict[int, Categorical[Tuple[petalState, float]]] = {}
            
            #Categorical distribution for Action A
            action_A_dist: Categorical[Tuple[petalState, float]] = {}
            
            if currPetal + 1 == self.n:
                action_A_dist[(petalState(currPetal+1), 1)] = 1 - currPetal/self.n
            else:
                action_A_dist[(petalState(currPetal+1), 0)] = 1 - currPetal/self.n
            
            action_A_dist[(petalState(currPetal-1), 0)] = currPetal/self.n
            
            #Represent Action A with 0
            curr_state_actions[0] = Categorical(action_A_dist)
            
            #Categorical distribution for Action B
            action_B_dist: Categorical[Tuple[petalState, float]] = {}
            
            for i in range(0, self.n):
                if (i == currPetal):
                    continue
                
                action_B_dist[(petalState(i), 0)] = 1/(self.n)
                
            action_B_dist[(petalState(self.n), 1)] = 1/self.n
             
            #Represent Action B with 1
            curr_state_actions[1] = Categorical(action_B_dist)
            
            #Add curr state's action's categorical distribution to entire map
            overall_state_action_map[curr_state] = curr_state_actions
        
        return overall_state_action_map
    
n_vals = [3,10,25]


for n_val in n_vals:
    frogMDP = frogMoveMDP(n = n_val)
    
    print("\nFrog MDP with n: " + str(n_val))
    print("------------------")
    #print(frogMDP)
    
    
    #print("MDP Policy Iteration Optimal Value Function and Optimal Policy: n = " + str(n_val))
    #print("--------------")
    opt_vf_pi, opt_policy_pi = policy_iteration_result(frogMDP, gamma=1)
        
    #pprint(opt_vf_pi)
    #print(opt_policy_pi)
    #print()
        
    fdp_A: FiniteDeterministicPolicy[petalState, int] = \
            FiniteDeterministicPolicy(
                {petalState(curr_petal): 0
                 for curr_petal in range(1,n_val)})
            
    fdp_B: FiniteDeterministicPolicy[petalState, int] = \
            FiniteDeterministicPolicy(
                {petalState(curr_petal): 1
                 for curr_petal in range(1,n_val)})
    
    implied_mrp_A: FiniteMarkovRewardProcess[petalState] =\
            frogMDP.apply_finite_policy(fdp_A)
            
    implied_mrp_B: FiniteMarkovRewardProcess[petalState] =\
            frogMDP.apply_finite_policy(fdp_B)
    
    implied_mrp_opt = frogMDP.apply_finite_policy(opt_policy_pi)
    opt_value_vec = implied_mrp_opt.get_value_function_vec(gamma = 1)
    
    graphValueA = np.array([], float)
    graphValueB = np.array([], float)
    
    for stateI in range(1, n_val):
        A_val = 0
        if (stateI-1 != 0):
            A_val += opt_value_vec[stateI - 2] * (stateI/n_val)
        if (stateI+1 != n_val):
            A_val += opt_value_vec[stateI] * (1 - (stateI/n_val))
        else:
            A_val += (1 - (stateI/n_val))
        
        B_val = 0
        for rand_petal in range(1, n_val):
            B_val += opt_value_vec[rand_petal-1] * (1/n_val)
        
        B_val += 1/n_val
    
        graphValueA = np.append(graphValueA, A_val)
        graphValueB = np.append(graphValueB, B_val)
    
    
    #graphArrX = np.array([], float)
    
    #graphValueA = implied_mrp_A.get_value_function_vec(gamma=1)
    
   # graphValueB = implied_mrp_B.get_value_function_vec(gamma=1)
    
    graphArrX = np.arange(1,n_val)
    
    
    plt.plot(graphArrX, graphValueA, color = "pink", label = "Value function if Action = A")
    plt.plot(graphArrX, graphValueB, color = "blue", label = "Value function if Action = B")
    
    plt.show()
    
    plt.clf()
    


    
    