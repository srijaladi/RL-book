# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Mapping
import numpy as np
import itertools
from rl.distribution import (Categorical, Distribution, FiniteDistribution,
                             SampledDistribution, Constant, Choose)
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
                                        FiniteMarkovRewardProcess, MarkovDecisionProcess,
                                        TransitionStep)
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

def TabularSARSA(
        mdp: MarkovDecisionProcess[S, A],
        startStates: NTStateDistribution[S],
        #simulations: Iterable[Iterable[mp.TransitionStep[S]]],
        gamma: float,
        episodes: int
        ) -> Mapping[S, A]:
    
    def def_value_neg():
        return -1
    def def_value():
        return 0
    def def_map():
        return defaultdict(int)
    def def_blank_map():
        return {}
    
    QAction: Mapping[S, Mapping[A, float]] = defaultdict(def_map)
    ActionStateCount: Mapping[S, Mapping[A, int]] = defaultdict(def_map)
    
    ss = startStates.sample_n(1)[0]
    
    updateResult = updateTabularSARSA(mdp, ss, episodes, gamma,\
                                             ActionStateCount, QAction)
    
    ActionStateCount = updateResult[0]
    QAction = updateResult[1]
    
    #print(QAction)
    Optimal_Policy: Mapping[S, int] = {s : max(QAction[s], key=QAction[s].get)\
                                       for s in QAction.keys()}

    print("Q Counts: ")
    print("--------------")
    for state in ActionStateCount.keys():
        print("STATE: " + str(state.state))
        for act in ActionStateCount[state].keys():
            print("Action: " + str(act) + ", Count: " + str(ActionStateCount[state][act]))
    print("")
    
    print("Q Values")
    print("--------------")
    for state in QAction.keys():
        print("STATE: " + str(state.state))
        for act in QAction[state].keys():
            print("Action: " + str(act) + ", Value: " + str(QAction[state][act]))
    print()
    return Optimal_Policy
    
    
def updateTabularSARSA(
        mdp: MarkovDecisionProcess[S, A],
        startState: S,
        num_episodes: int,
        #currSim: Iterable[mp.TransitionStep[S]],
        gamma: float,
        ASCount: Mapping[S, Mapping[A, int]],
        currQ: Mapping[S, Mapping[A, float]],
        ) -> Tuple[Mapping[S, Mapping[A, int]], Mapping[S, Mapping[A, float]]]:
    
    #greedy_prob = 1 - (1/1)
    
    cumulativeReward = 0
    seqRewardArr = []
    currState = startState
    
    lastReward = -88
    lastAction = -88
    lastState = -88
    
    currReward = -88
    actionChoice = -88
    
    trace: List[TransitionStep[S, A]] = []
    
    def def_false():
        return False
    state_occur: Mapping[S, bool] = defaultdict(def_false)            
    
    for episode_num in range(1, num_episodes + 1):
        
        greedy_prob = 1 - (1/(episode_num / 100))
        
        rand = np.random.uniform(0,1,1)
        
        if (rand >= greedy_prob or len(currQ[currState]) == 0):
            optionsChoose = Choose(list(mdp.actions(currState)))
            actionChoice = optionsChoose.sample_n(1)[0]
        else:
            actionChoice = max(currQ[currState], key=currQ[currState].get)
        
        nextDist = mdp.step(currState, actionChoice)
        
        nextStateReward = nextDist.sample_n(1)[0]
        nextState = nextStateReward[0]
        
        currReward = nextStateReward[1]
        
        if episode_num > 1:
            useCount = ASCount[lastState][lastAction] + 1
            ASCount[lastState][lastAction] = useCount
            
            lastASValue = currQ[lastState][lastAction]
            currASValue = currQ[currState][actionChoice]
            
            EstASValue = lastReward + gamma * currASValue
            alpha_step = 1/useCount
            
            newASValue = lastASValue + alpha_step * (EstASValue - lastASValue)
            currQ[lastState][lastAction] = newASValue

        
        lastReward = currReward
        lastAction = actionChoice
        lastState = currState
        
        currState = nextState
        
    
    return (ASCount, currQ)

    