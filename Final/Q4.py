# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Mapping
import numpy as np
import itertools
import random
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
                    Mapping, TypeVar, Set, List, Iterator)
import itertools
from itertools import combinations
from itertools import product
from rl.dynamic_programming import evaluate_mrp_result
from rl.dynamic_programming import policy_iteration_result
from rl.dynamic_programming import value_iteration_result
from rl.dynamic_programming import value_iteration
from typing import (Callable, Iterable, Iterator, Optional, TypeVar)
from rl.iterate import converged, iterate, last
from rl.markov_process import NonTerminal
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        FiniteMarkovRewardProcess)
from rl.dynamic_programming import value_iteration, almost_equal_vfs, greedy_policy_from_vf
from pprint import pprint
import rl.markov_process as mp
from rl.approximate_dynamic_programming import (ValueFunctionApprox,
                                                QValueFunctionApprox,
                                                NTStateDistribution)
import math
from collections import defaultdict
from typing import Iterator
from rl.distribution import Choose
from rl.function_approx import Tabular, learning_rate_schedule
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap, \
    InventoryState
from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.chapter11.control_utils import get_vf_and_policy_from_qvf
from rl.monte_carlo import epsilon_greedy_policy
from rl.td import q_learning_experience_replay
from rl.dynamic_programming import value_iteration_result
import rl.iterate as iterate
import itertools
from pprint import pprint
from typing import Sequence, Iterable, Callable
from rl.function_approx import AdamGradient
from rl.function_approx import LinearFunctionApprox
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.distribution import Choose
from rl.markov_decision_process import NonTerminal
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.chapter2.simple_inventory_mrp import InventoryState
from rl.chapter10.prediction_utils import (
    mc_prediction_learning_rate,
    td_prediction_learning_rate
)
from rl.td import (epsilon_greedy_action, q_learning)
import numpy as np
from itertools import islice

S = TypeVar('S')
A = TypeVar('A')


@dataclass(frozen=True)
class cropState:
    quality: int
    
    
cropActionMapping = Mapping[cropState, Mapping[int, Categorical[Tuple[cropState, float]]]]


class cropsMDP(FiniteMarkovDecisionProcess[cropState, int]):
    
    def __init__(self, C: int, z: int, u: Callable):
        self.zeroProb = z
        self.maxQ = C
        self.utilityFunc = u
        super().__init__(self.get_action_transition_reward_map()) 
        
    def get_action_transition_reward_map(self) -> cropActionMapping:
        d: Dict[cropState, Dict[int, Categorical[Tuple[cropState, float]]]] = {}
        u = self.utilityFunc
        
        for quality in range(0, self.maxQ + 1):
            #Holds the distribution for each action
            state_dist_map: Dict[int, Categorical[Tuple[cropState, float]]] = {}
            
            #Below is adding action of 0 (don't sell) and its distribution
            action_dist_map: Dict[Tuple[cropState, float], float] = {}
            action = 0
            
            numPossQualities = self.maxQ + 1 - quality
            zeroProb = self.zeroProb
            incProb = 1 - zeroProb
            
            for newQuality in range(quality, self.maxQ + 1):
                state_reward = (cropState(newQuality), 0)
                action_dist_map[state_reward] = (1/numPossQualities) * incProb
            
            zero_state_reward = (cropState(0), u(0))
            
            if zero_state_reward in action_dist_map.keys():
                action_dist_map[zero_state_reward] += zeroProb
            else:
                action_dist_map[zero_state_reward] = zeroProb
                
            state_dist_map[action] = Categorical(action_dist_map)
            
            #Below is adding action of 1 (sell) and its distribution
            action_dist_map: Dict[Tuple[cropState, float], float] = {}
            action = 1
            state_reward_sell = (cropState(0), u(quality))
            action_dist_map[state_reward_sell] = 1
            
            state_dist_map[action] = Categorical(action_dist_map)
            
            d[cropState(quality)] = state_dist_map
        
        #print(d)
        return d            
            

C = 100
z = 0.1
user_gamma = 0.9
def u(x):
    return x
cropMDP = cropsMDP(C = C, z = z, u = u)


print("MDP Transition Map created successfully")
print("------------------\n")

opt_vf_vi, opt_policy_vi = value_iteration_result((cropMDP), gamma=user_gamma)
print("Optimal Policy and value function for MDP computed")
print("--------------\n")

print("Optimal Policy MDP at every state")
print("--------------\n")
print(opt_policy_vi)

print("Optimal Value Function MDP at every state")
print("--------------\n")
print(opt_vf_vi)

def collect_data_vi(opt_policy_vi, opt_vf_vi):
    vi_policy_data = []
    vi_vf_data = []
    
    for i in range(C+1):
        checkState = NonTerminal(cropState(i))
        vi_policy_data.append(opt_policy_vi.act(checkState).sample())
        vi_vf_data.append(opt_vf_vi[checkState])
    
    return [np.array(vi_policy_data), np.array(vi_vf_data)]

def graph_vi(vi_p, vi_vf):
    graphArrX = np.arange(len(vi_p))
    
    plt.plot(graphArrX, vi_p, color = "red")
    plt.show()
    
    plt.clf()
    
    plt.plot(graphArrX, vi_vf, color = "red")
    plt.show()
    
    plt.clf()

npArr = collect_data_vi(opt_policy_vi, opt_vf_vi)
graph_vi(npArr[0], npArr[1])


            
#ffs: Sequence[Callable[Tuple[cropState, int], float]] = \
#    [(lambda x, s=s: float(x.quality == s.quality)) for s in cropMDP.non_terminal_states]


def ftr_func(x, C = 100, z = 0.1, gamma = user_gamma):
    state = x[0].state
    action  = x[1]
    
    returnArr = [0 for s in cropMDP.non_terminal_states]
    
    if action == 1:
        if state.quality > 0:
            returnArr[state.quality] += 0
            returnArr[0] += gamma
        #print("hello")
        #print(returnArr)
        return [state.quality] + returnArr
    
    numPoss = C - state.quality + 1
    prob = (1 - z)/numPoss
    
    for i in range(state.quality, C+1):
        returnArr[i] += (gamma * prob)
    
    returnArr[0] += gamma*z
    
    return [0] + returnArr

all_ftrs = []

for currS in cropMDP.non_terminal_states:
    all_ftrs.append([ftr_func((currS, a)) for a in [0,1]])
    

#print(all_ftrs)


ffs = [(lambda x, s=s: all_ftrs[x[0].state.quality][x[1]][s.state.quality]) for s in cropMDP.non_terminal_states]   

q_ag: AdamGradient = AdamGradient(
   learning_rate=0.05,
    decay1=0.9,
    decay2=0.999
)

q_func_approx: QValueFunctionApprox[NonTerminal[cropState], A]
q_func_approx = LinearFunctionApprox.create(feature_functions=ffs,adam_gradient=q_ag)
    
#tempstate = cropMDP.non_terminal_states[0]
#print(tempstate)
#testList = [(s,a) for s in cropMDP.non_terminal_states for a in [0,1]]
testList = [(NonTerminal(cropState(5)),0), (NonTerminal(cropState(5)),1)]
print(q_func_approx.get_feature_values(testList))
print()

episodeLen = 10
numEpisodes = 10000
epsilon = 0.1

print("Completed episode: ")

for episode in range(numEpisodes):

    print(str(episode+1), end = ", ")
    
    startQuality = random.randint(0,C)
    startState = NonTerminal(cropState(startQuality))
    currState = startState
    
    last_q_func_approx = q_func_approx
        
    for step in range(episodeLen):
        #Epsilon Greedy Policy choice defined by epsilon value
        rand = np.random.uniform(0,1)
        currAction = 1
        if (rand < epsilon) or (q_func_approx((currState,0)) == q_func_approx((currState,1))):
            currAction = random.randint(0,1)
        else:
            if (q_func_approx((currState,0)) > q_func_approx((currState,1))):
                currAction = 0
                    
        #Sample the next step
        transitionStep = cropMDP.step(currState, currAction).sample()
        
        nextState = transitionStep[0]
        reward = transitionStep[1]
        
        #Find best action nextState
        bestNextAction = 1
        if (q_func_approx((nextState,0)) > q_func_approx((nextState,1))):
            bestNextAction = 0
        
        #Calculate and hold the target for updating purposes
        target = reward + user_gamma * q_func_approx((nextState, bestNextAction))
        #print("CurrState: " + str(currState) + " and currAction: " + str(currAction) + ": target: " + str(target))
        #print(target)
        
        #Pass in the curent (s,a) pair and target value for update
        q_func_approx = q_func_approx.update([((currState, currAction), target)])
        currState = nextState #Iterate the currState to the nextState
        
        
q_optimal_policy: Mapping[NonTerminal[cropState], int] = {}
q_optimal_vf: Mapping[NonTerminal[cropState], float] = {}

for s in cropMDP.non_terminal_states:
    maxAction = 1
    if (q_func_approx((s, 0)) > q_func_approx((s, 1))):
        maxAction = 0
    q_optimal_policy[s] = maxAction
    q_optimal_vf[s] = q_func_approx((s, maxAction))
    

print()
print("Optimal Policy Q-Learning at every state")
print("--------------\n")
for s in cropMDP.non_terminal_states:
    print("For State " + str(s.state) + ": Do Action " + str(q_optimal_policy[s]))

print()
print("Optimal Value Function Q-Learning at every state")
print("--------------\n")
for s in cropMDP.non_terminal_states:
    print("For State " + str(s.state) + ": VF " + str(q_optimal_vf[s]))

    
def collect_data(opt_policy_vi, opt_vf_vi, q_optimal_policy, q_optimal_vf):
    vi_policy_data = []
    q_policy_data = []
    vi_vf_data = []
    q_vf_data = []
    
    for i in range(C+1):
        checkState = NonTerminal(cropState(i))
        vi_policy_data.append(opt_policy_vi.act(checkState).sample())
        vi_vf_data.append(opt_vf_vi[checkState])
        q_policy_data.append(q_optimal_policy[checkState])
        q_vf_data.append(q_optimal_vf[checkState]);
    
    return [np.array(vi_policy_data), np.array(q_policy_data), np.array(vi_vf_data), np.array(q_vf_data)]

def graph(vi_p, q_p, vi_vf, q_vf):
    graphArrX = np.arange(len(vi_p))
    
    plt.plot(graphArrX, vi_p, color = "red")
    plt.plot(graphArrX, q_p, color = "blue")
    plt.show()
    
    plt.clf()
    
    plt.plot(graphArrX, vi_vf, color = "red")
    plt.plot(graphArrX, q_vf, color = "blue")
    plt.show()
    
    plt.clf()

npArr = collect_data(opt_policy_vi, opt_vf_vi, q_optimal_policy, q_optimal_vf)
graph(npArr[0], npArr[1], npArr[2], npArr[3])




"""
C = 100
z = 0.1
user_gamma = 0.9
def u(x):
    return math.log(x+1)
cropMDP = cropsMDP(C = C, z = z, u = u)


print("MDP Transition Map created successfully")
print("------------------\n")
#print(diceMDP)

print("Optimal Policy and value function for MDP computed")
print("--------------\n")
opt_vf_vi, opt_policy_vi = value_iteration_result((cropMDP), gamma=user_gamma)

print("Optimal Policy MDP at every state")
print("--------------\n")
print(opt_policy_vi)

print("Optimal Value Function MDP at every state")
print("--------------\n")
print(opt_vf_vi)
"""
