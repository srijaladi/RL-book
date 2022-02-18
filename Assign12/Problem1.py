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






def TabularTDLambaPredict(
        simulations: Iterable[Iterable[mp.TransitionStep[S]]],
        gamma: float,
        l: float
        ) -> Mapping[S, float]:
    
    def def_value():
        return 0
    
    V: Mapping[S, float] = defaultdict(def_value)
    
    for currSim in simulations:
        updateResult = updateTabularTDLambda(currSim, gamma, l, V)
        V = updateResult
    
    return V
    
    
def updateTabularTDLambda(
        currSim: Iterable[mp.TransitionStep[S]],
        gamma: float,
        l: float,
        currApprox: Mapping[S, float],
        ) -> Mapping[S, float]:
    
    cumulativeReward = 0
    seqRewardArr = []
    
    for transitionStep in reversed(currSim):
        currState = transitionStep.state
        currReward = transitionStep.reward
        
        cumulativeReward *= gamma
        cumulativeReward += currReward
        
        seqRewardArr.append(cumulativeReward)
        
    seqRewardArr.reverse()
    T = len(seqRewardArr)
    
    for t, transitionStep in enumerate(currSim):
        currState = transitionStep.state
        
        u = math.exp(l, T - t - 1)
        G_t = seqRewardArr[t]
        
        MC_Val = u * G_t
        
        TD_Val = 0
        
        for n, nextTransitionStep in enumerate(currSim[t:]):
            
            u_n = (1 - l) * math.exp(l, n)
            
            if (t+n+1 == len(seqRewardArr)):
                subtract_rewards = 0
            else:
                subtract_rewards = seqRewardArr[t+n+1] * gamma
                
            G_tn = G_t - subtract_rewards + currApprox[currSim[t+n].next_state]
            
            TD_Val += u_n * G_tn
        
        
        currApprox[currState] = TD_Val + MC_Val
        
    
    return currApprox
    

    