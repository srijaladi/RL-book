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


S = TypeVar('S')
A = TypeVar('A')

from typing import Sequence, Tuple, Mapping

S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]


def get_state_return_samples(
    data: DataType
) -> Sequence[Tuple[S, float]]:
    """
    prepare sequence of (state, return) pairs.
    Note: (state, return) pairs is not same as (state, reward) pairs.
    """
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]


def get_mc_value_function(
    state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    gamma = 1
    
    print(state_return_samples)
    print()
    
    def def_value():
        return 0
    V: Mapping[S, float] = defaultdict(def_value)
    stateUpdates: Mapping[S, int] = defaultdict(def_value)

    updateResult = updateTabularMC(state_return_samples, gamma, V, stateUpdates)
    stateUpdates = updateResult[0]
    V = dict(updateResult[1])
    
    return V
    
    
def updateTabularMC(
        currSim: Iterable[mp.TransitionStep[S]],
        gamma: float,
        currApprox: Mapping[S, float],
        stateUpdates: Mapping[S, int]
        ) -> Tuple[Mapping[S, int], Mapping[S, float]]:
    
    cumulativeReward = 0
    
    for transitionStep in (currSim):
        currState = transitionStep[0]
        cumulativeReward = transitionStep[1]
        
        n = stateUpdates[currState]
        change = (1/(n+1)) * (cumulativeReward - currApprox[currState])
        
        currApprox[currState] += change
        stateUpdates[currState] += 1
            
    
    return (stateUpdates, currApprox)


def get_state_reward_next_state_samples(
    data: DataType
) -> Sequence[Tuple[S, float, S]]:
    """
    prepare sequence of (state, reward, next_state) triples.
    """
    return [(s, r, l[i+1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    
    def def_dict():
        return defaultdict(int)
    transitionCount: Mapping[S, Mapping[S, int]] = defaultdict(def_dict)
    transitionToReward: Mapping[S, Mapping[S, float]] = defaultdict(def_dict)
    stateCount: Mapping[S, int] = defaultdict(int)
    rewardFunc: Mapping[S, float] = defaultdict(int)
    
    for step in srs_samples:
        currState = step[0]
        reward = step[1]
        nextState = step[2]
        
        transitionCount[currState][nextState] += 1
        transitionToReward[nextState][currState] += reward
        
        stateCount[currState] += 1
        n = stateCount[currState]
        
        currReward = rewardFunc[currState]
        rewardFunc[currState] += (1/(n)) * (reward - currReward)
    
    probFunc: Mapping[S, Mapping[S, float]] = defaultdict(def_dict)
    
    for startState in transitionCount.keys():
        n = stateCount[startState]
        for endState in transitionCount[startState].keys():
            transCount = transitionCount[startState][endState]
            probFunc[startState][endState] = transCount/n
            
    return (probFunc, rewardFunc)


def get_mrp_value_function(
    prob_func: ProbFunc,
    reward_func: RewardFunc
) -> ValueFunc:
    
    rewardVec: List[float] = []
    probVec: List[List[float]] = []
    
    stateOrder: List[S] = []
    
    for state in reward_func.keys():
        stateOrder.append(state)
        
    for state in stateOrder:
        rewardVec.append(reward_func[state])
        tempList: List[float] = []
        for nextState in stateOrder:
            prob = prob_func[state][nextState]
            if state == nextState:
                tempList.append(1 - prob)
            else:
                tempList.append(-prob)
        probVec.append(tempList)
    
    A = np.array(probVec)
    B = np.array(rewardVec)
    
    print(A)
    print(B)
    print()
    
    x = np.linalg.solve(A, B)
    
    V: Mapping[S, float] = {}
    
    for index, state in enumerate(stateOrder):
        V[state] = x[index]
    
    return V
    
    """
    Implement code that calculates the MRP Value Function from the probability
    transitions and reward function, compatible with the interface defined above.
    Hint: Use the MRP Bellman Equation and simple linear algebra
    """

def TabularTDPredict(
        simulations: Iterable[mp.TransitionStep[S]],
        gamma: float,
        num_updates: int = 300000,
        learning_rate: float = 0.3,
        learning_rate_decay: int = 30
        ) -> Mapping[S, float]:
    #print(num_updates)
    def def_value():
        return 0
    def alpha_compute(n):
        #n -> Number of updates so far
        return learning_rate * (n / learning_rate_decay + 1) ** -0.5
        return 1/(n+1)
    
    V: Mapping[S, float] = defaultdict(def_value)
    stateUpdates: Mapping[S, int] = defaultdict(def_value)
    
    #print(num_updates)
    for update in range(num_updates):
        alpha = alpha_compute(update)
        
        i = update % len(simulations)
        transitionStep = simulations[i]
        #print(transitionStep)
        updateResult = updateTabularTD(transitionStep, gamma, V, stateUpdates, alpha)
        stateUpdates = updateResult[0]
        V = updateResult[1]

    return V
    
    
def updateTabularTD(
        transitionStep: mp.TransitionStep[S],
        gamma: float,
        currApprox: Mapping[S, float],
        stateUpdates: Mapping[S, int],
        alpha: float
        ) -> Tuple[Mapping[S, int], Mapping[S, float]]:
    
    currState = transitionStep[0]
    currReward = transitionStep[1]
    nextState = transitionStep[2]
            
    if isinstance(nextState, Terminal):
        futureValue = 0
    else:
        futureValue = currApprox[nextState]
        
    currEstimate = currReward + gamma * futureValue
    
    n = stateUpdates[currState]
    #alpha = alpha_compute(n)
    change = alpha * (currEstimate - currApprox[currState])
    currApprox[currState] += change
        
    stateUpdates[currState] = n+1
        
    return (stateUpdates, currApprox)

def get_td_value_function(
    srs_samples: Sequence[Tuple[S, float, S]],
    num_updates: int = 300000,
    learning_rate: float = 0.3,
    learning_rate_decay: int = 30
) -> ValueFunc:
    
    ValueFunc = dict(TabularTDPredict(srs_samples,1))
    
    
    """
    Implement tabular TD(0) (with experience replay) Value Function compatible
    with the interface defined above. Let the step size (alpha) be:
    learning_rate * (updates / learning_rate_decay + 1) ** -0.5
    so that Robbins-Monro condition is satisfied for the sequence of step sizes.
    """
    return ValueFunc

def get_lstd_value_function(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    gamma = 1
    
    listA: List[List[float]] = []
    listB: List[float] = []
    
    stateOrder: List[S] = []
    stateOrderMap: Mapping[S, int] = {}
    
    print(srs_samples)
    
    for step in srs_samples:
        currState = step[0]
        reward = step[1]
        nextState = step[2]
        
        if currState in stateOrderMap.keys():
            stateIndex = stateOrderMap[currState]
        else:
            stateOrder.append(currState)
            stateIndex = len(stateOrder) - 1
            stateOrderMap[currState] = stateIndex
        
        if nextState in stateOrderMap.keys():
            nextStateIndex = stateOrderMap[nextState]
        else:
            if nextState == 'T':
                nextStateIndex = -1
            else:
                stateOrder.append(nextState)
                nextStateIndex = len(stateOrder) - 1
                stateOrderMap[nextState] = nextStateIndex
        
        
        addListA = [0 for _ in range(len(stateOrder))]
        addListA[stateIndex] += 1
        if nextStateIndex >= 0:
            addListA[nextStateIndex] -= gamma
        
        
        if stateIndex >= len(listB):
            listB.append(reward)
            listA.append(addListA)
        else:
            listB[stateIndex] += reward
            
            for index, val in enumerate(listA[stateIndex]):
                addListA[index] += val
            listA[stateIndex] = addListA
      
        
    while len(stateOrder) > len(listA):
        addListA = [0 for _ in range(len(stateOrder))]
        listA.append(addListA)
        listB.append(0)
        
    A = np.array(listA)
    B = np.array(listB)
    #print(stateOrder)
    print(A)
    print(B)
    print()
    x = np.linalg.solve(A, B)
    
    V: Mapping[S, float] = {}
    
    for index, state in enumerate(stateOrder):
        V[state] = x[index]
    
    return V  
        
        
        
    
    """
    Implement LSTD Value Function compatible with the interface defined above.
    Hint: Tabular is a special case of linear function approx where each feature
    is an indicator variables for a corresponding state and each parameter is
    the value function for the corresponding state.
    """


if __name__ == '__main__':
    given_data: DataType = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    sr_samps = get_state_return_samples(given_data)

    print("------------- MONTE CARLO VALUE FUNCTION --------------")
    print(get_mc_value_function(sr_samps))

    srs_samps = get_state_reward_next_state_samples(given_data)

    pfunc, rfunc = get_probability_and_reward_functions(srs_samps)
    print("-------------- MRP VALUE FUNCTION ----------")
    print(get_mrp_value_function(pfunc, rfunc))

    print("------------- TD VALUE FUNCTION --------------")
    print(get_td_value_function(srs_samps))

    print("------------- LSTD VALUE FUNCTION --------------")
    print(get_lstd_value_function(srs_samps))