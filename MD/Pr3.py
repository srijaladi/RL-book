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

@dataclass(frozen=True)
class handState:
    Ones: int
    Sum: int
    Dice: Tuple[int]
    
handActionMapping = Mapping[handState, Mapping[List[int], Categorical[Tuple[handState, float]]]]

class diceRollsMDP(FiniteMarkovDecisionProcess[handState, int]):
    def __init__(self, N: int, K: int, C: int):
        self.totalRolls = N
        self.numFaces = K
        self.minOnes = C
        self.possRolls = [i for i in range(1, K+1)]
        self.allCombinations: Dict[int, Iterable[List[int]]] = {}
        
        for i in range(0, N+1):
            possibleChoices = [self.possRolls for _ in range(i)]
            
            allPossibilities = list(product(*possibleChoices))
            
            self.allCombinations[i] = allPossibilities
            
        super().__init__(self.get_action_transition_reward_map()) 
    
    def get_action_transition_reward_map(self) -> handActionMapping:
        
        d: Dict[handState, Dict[List[int], Categorical[Tuple[handState, float]]]] = {}

        for currDiceLeft in range(1, self.totalRolls+1):
            print(currDiceLeft)
            for numOnes in range(self.totalRolls - currDiceLeft + 1):
                
                minSum = diceTaken = self.totalRolls - currDiceLeft
                maxSum = ((diceTaken - numOnes) * self.numFaces) + numOnes 
                
                for currSum in range(minSum, maxSum+1):
                    for diceAvail in self.allCombinations[currDiceLeft]:
                        
                        currHand = handState(Ones = numOnes, Sum = currSum, Dice = diceAvail)
                        
                        action_dist_map: Dict[List[int], Categorical[Tuple[handState, float]]]= {}
                        
                        for takeAmount in range(1, currDiceLeft+1):
                            allActions = list(combinations(diceAvail, takeAmount))
                            for action in allActions:
                                rem = currDiceLeft - takeAmount
                                newOnes = numOnes + action.count(1)
                                newSum = currSum + sum(action)
                                
                                new_state_dist: Dict[Tuple[handState, float], float]= {}
                                
                                
                                if rem == 0:
                                    if newOnes < self.minOnes:
                                        new_state_dist[(handState(newOnes, newSum, tuple([])), 0)] = 1
                                    else:
                                        new_state_dist[(handState(newOnes, newSum, tuple([])), newSum)] = 1
                                else:
                                    totalOptions = len(self.allCombinations[rem])
                                    for nextDiceAvail in self.allCombinations[rem]:
                                        new_state_dist[(handState(newOnes, newSum, tuple(nextDiceAvail)), 0)] = 1/totalOptions
                                
                                action_dist_map[action] = Categorical(new_state_dist)
                        
                        d[currHand] = action_dist_map
        #print(d)
              
        return d
    
N = 6
K = 3
C = 1    
diceMDP = diceRollsMDP(N = N, K = K, C = C)


print("MDP Transition Map created successfully")
print("------------------\n")
#print(diceMDP)

print("Optimal Policy and value function for MDP1 computed")
print("--------------\n")
opt_vf_vi, opt_policy_vi = value_iteration_result((diceMDP), gamma=1)

expScore = 0

for possRolls in diceMDP.allCombinations[N]:
    expScore += opt_vf_vi[NonTerminal(handState(0,0,possRolls))]
    
expScore /= len(diceMDP.allCombinations[N])

print("Expected Score of playing game optimally: ")
print(expScore)

startRollToFindOptimal = tuple([1,2,2,2,2,3])
print("Optimal action when rolling: " + str(startRollToFindOptimal) + " on first roll:")
print(opt_policy_vi.action_for[handState(0,0,startRollToFindOptimal)])





    
    
    