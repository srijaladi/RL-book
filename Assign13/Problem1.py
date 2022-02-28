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

def TabularMCPolicy(
        mdp: MarkovDecisionProcess[S, A],
        startStates: NTStateDistribution[S],
        #simulations: Iterable[Iterable[mp.TransitionStep[S]]],
        gamma: float,
        episodes: int,
        num_in_ep: int
        ) -> Mapping[S, A]:
    
    def def_value_neg():
        return -1
    def def_value():
        return 0
    def def_map():
        return defaultdict(def_value)
    
    GreedyV: Mapping[S, float] = defaultdict(def_value)
    QAction: Mapping[S, Mapping[A, float]] = defaultdict(def_map)
    ActionStateCount: Mapping[S, Mapping[A, int]] = defaultdict(def_map)
    
    ss = startStates.sample_n(1)[0]
    
    for episode in range(episodes):
        updateResult = updateTabularMCAction(mdp, ss, episode+1, num_in_ep, gamma,\
                                             ActionStateCount, QAction, GreedyV)
        ActionStateCount = updateResult[0]
        QAction = updateResult[1]
        GreedyV = updateResult[2]
    
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
    
    
def updateTabularMCAction(
        mdp: MarkovDecisionProcess[S, A],
        startState: S,
        episode_num: int,
        num_in_ep: int,
        #currSim: Iterable[mp.TransitionStep[S]],
        gamma: float,
        ASCount: Mapping[S, Mapping[A, int]],
        currQ: Mapping[S, Mapping[A, float]],
        GreedyV: Mapping[S, float],
        ) -> Tuple[Mapping[S, Mapping[A, int]], Mapping[S, Mapping[A, float]], Mapping[S, float]]:
    
    greedy_prob = 1 - (1/episode_num)
    
    cumulativeReward = 0
    seqRewardArr = []
    currState = startState
    
    trace: List[TransitionStep[S, A]] = []
            
    
    for _ in range(num_in_ep):
        rand = np.random.uniform(0,1,1)
        if (rand >= greedy_prob):
            optionsChoose = Choose(list(mdp.actions(currState)))
            actionChoice = optionsChoose.sample()
            #print(list(mdp.actions(currState)))
            #print(actionChoice)
        else:
            #print(currQ)
            #print(greedy_prob)
            actionChoice = max(currQ[currState], key=currQ[currState].get)
        
        #print(actionChoice)
        nextDist = mdp.step(currState, actionChoice)
        
        nextStateReward = nextDist.sample_n(1)[0]
        nextState = nextStateReward[0]
        currReward = nextStateReward[1]
        
        trace.append(TransitionStep(state=currState,action=actionChoice,\
                                    next_state=nextStateReward[0],reward=currReward))
        
        currState = nextState
    
    for transStep in reversed(trace):
        addState = transStep.state
        addAction = transStep.action
        useReward = transStep.reward
        
        cumulativeReward *= gamma
        cumulativeReward += useReward
        
        currCount = ASCount[addState][addAction] + 1
        ASCount[addState][addAction] = currCount
        
        currASValue = currQ[addState][addAction]
        newASValue = currASValue + (1/currCount) * (cumulativeReward - currASValue)
        
        currQ[addState][addAction] = newASValue
    
    return (ASCount, currQ, GreedyV)
    

    