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
from collections import defaultdict
S = TypeVar('S')
A = TypeVar('A')






def TabularMCPredict(
        simulations: Iterable[Iterable[mp.TransitionStep[S]]],
        gamma: float
        ) -> Mapping[S, float]:
    
    def def_value():
        return 0
    
    V: Mapping[S, float] = defaultdict(def_value)
    stateUpdates: Mapping[S, int] = defaultdict(def_value)
    
    for currSim in simulations:
        updateResult = updateTabularMC(currSim, gamma, V, stateUpdates)
        stateUpdates = updateResult[0]
        V = updateResult[1]
    
    return V
    
    
def updateTabularMC(
        currSim: Iterable[mp.TransitionStep[S]],
        gamma: float,
        currApprox: Mapping[S, float],
        stateUpdates: Mapping[S, int]
        ) -> Tuple[Mapping[S, int], Mapping[S, float]]:
    
    cumulativeReward = 0
    
    for transitionStep in reversed(currSim):
        currState = transitionStep.state
        currReward = transitionStep.reward
        
        cumulativeReward *= gamma
        cumulativeReward += currReward
        
        n = stateUpdates[currState]
        change = (1/(n+1)) * (cumulativeReward - currApprox[currState])
        
        currApprox[currState] += change
        stateUpdates[currState] += 1
            
    
    return (stateUpdates, currApprox)
    
    
    