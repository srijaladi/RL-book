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
from rl.chapter2.stock_price_simulations import\
    plot_single_trace_all_processes
from rl.chapter2.stock_price_simulations import\
    plot_distribution_at_time_all_processes
import matplotlib         
from matplotlib import pyplot as plt
from typing import (Callable, Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, TypeVar, Set)
"""
Created on Sun Jan  9 06:58:52 2022

@author: SriJaladi
"""

"""
When exmaining the state space and structure of probabilities, we can first
denote the state space, non-terminal space, and terminal space as:
S = {s1, s2, s3, s4, ..., s100} -> State Space
N = {s1, s2, s3, s4, ..., s99} -> Non-terminal states
T = {s100} -> Termainl states
Where each si represents tile number i on the board. This is why s100 is the 
terminal state because reaching tile number 100 means you have won.

In a general sense (ignoring ladders, snakes, and end points for now) the 
structure of transition probabilities would be in the following form:
P(St+1 = s' | St = s) = 1/6. In terms of a transition probability map, this 
would look as follows:
Transition Map for Si = [Si+1: 1/6, Si+2: 1/6, Si+3: 1/6, Si+4: 1/6, Si+5: 1/6, Si+6: 1/6]
Obviously this gets a bit more complex with snakes and ladders and that Si where i>100
results in going backwards but this is the basic idea of the transition map.

We can create the BASIC version of this transition map (ignoring snakes, ladders, and ending
                                                        for now as shown in "create_transition_map":
"""

@dataclass(frozen=True)
class StateMP:
    current_point: int


@dataclass
class SAndLMP(FiniteMarkovProcess[MarkovProcess[StateMP]]):
    
    def create_transition_map(self):
        transition_map= {}
        
        snakes = {16:6,
                  47:26,
                  49:11,
                  56:53,
                  62:19,
                  64:60,
                  87:24,
                  93:73,
                  95:75,
                  98:78}
        ladders = {1:38,
                   4:14,
                   9:31,
                   21:42,
                   28:84,
                   36:46,
                   51:67,
                   71:91,
                   80:100}
        
        for i in range(1,100):
            transition_probabilities = {}
            
            for j in range(i+1,i+7):
                
                if j in snakes:
                    new_state = snakes[j]
                elif j in ladders:
                    new_state = ladders[j]
                else:
                    new_state = j
                
                if (new_state == 100):
                    transition_probabilities[Terminal(StateMP(new_state))] = 1/6
                elif (new_state > 100):
                    transition_probabilities[Terminal(StateMP(new_state))] = 1/6
                else:
                    transition_probabilities[NonTerminal(StateMP(new_state))] = 1/6
            
            transition_map[NonTerminal(StateMP(i))] = transition_probabilities
            
        
        
        return transition_map
    
    
    #Simple transition function to return categorical distribution of next states from current
    def transition(
        self,
        state: NonTerminal[StateMP]
    ) -> Categorical[State[StateMP]]:
        
        transition_map = self.create_transition_map()
        
        return Categorical(transition_map[state])
    
    
    
mp = SAndLMP()

#The starting state distribution is just always starting at current_point = 1
start_state_distribution = Constant(
        NonTerminal(StateMP(1))
        )

#Have 1000 traces or simulations
numTraces = 10
traceNumber = 0

roll_counts = np.array([], int)

for i in mp.traces(start_state_distribution):
    traceNumber += 1
    
    roll_counts = np.append(roll_counts, len([j for j in i]))
    
    if (traceNumber == numTraces): 
        break

plt.hist(roll_counts)
plt.show()
plt.clf()

print("Expected Rolls Needed: " + str(np.average(roll_counts)))


"""
We can also model the same concept of the ogame (trying to identify how many
expected number of dice rolls must be made to end the game) from the concept
of a reward. From each state, the future reward is expressed as the expected
number of moves needed from that position to reach the end.

Essentially, R(s) = E{num_more_moves_needed}

"""

@dataclass
class SAndLMPReward(FiniteMarkovRewardProcess[StateMP]):
    
    def __init__ (self):
        #transition_map = self.create_transition_map()
        super().__init__(self.get_transition_reward_map())
        
    def create_transition_map(self):
        transition_map= {}
        
        snakes = {16:6,
                  47:26,
                  49:11,
                  56:53,
                  62:19,
                  64:60,
                  87:24,
                  93:73,
                  95:75,
                  98:78}
        ladders = {1:38,
                   4:14,
                   9:31,
                   21:42,
                   28:84,
                   36:46,
                   51:67,
                   71:91,
                   80:100}
        
        for i in range(1,100):
            transition_probabilities = {}
            
            for j in range(i+1,i+7):
                
                if j in snakes:
                    new_state = snakes[j]
                elif j in ladders:
                    new_state = ladders[j]
                else:
                    new_state = j
                
                if (new_state == 100):
                    transition_probabilities[Terminal(StateMP(new_state))] = 1/6
                elif (new_state > 100):
                    transition_probabilities[Terminal(StateMP(new_state))] = 1/6
                else:
                    transition_probabilities[NonTerminal(StateMP(new_state))] = 1/6
            
            transition_map[NonTerminal(StateMP(i))] = transition_probabilities
        
        return transition_map    
    
    def get_transition_reward_map(self) -> \
            Mapping[
                StateMP,
                FiniteDistribution[Tuple[StateMP, float]]
            ]:
        temp_transition_map = self.create_transition_map()
        
        tran_reward_map  = {}
        
        for curr_state in temp_transition_map:
            curr_state_reward_map = {(StateMP(new_state), 1) : 1/6 for new_state in temp_transition_map[curr_state]}
            tran_reward_map[curr_state] = Categorical(curr_state_reward_map)
        
        print(tran_reward_map)
        return tran_reward_map
        
        

user_gamma = 1.0

mpr = SAndLMPReward()

#mpr.display_reward_function()


"""
totalSims = 1000
rolls_needed = np.array([], int)



for i in range(totalSims):
    rolls = -1
    
    simulationStates = mp.simulate(start_state_distribution)
    
    for j in simulationStates:
        rolls += 1
    
    rolls_needed = np.append(rolls_needed, rolls)
    
plt.hist(rolls_needed)
print("Expected Rolls Needed: " + str(np.average(rolls_needed)))
"""
