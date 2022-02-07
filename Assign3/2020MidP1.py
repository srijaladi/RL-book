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
class gridState:
    loc: Tuple[int, int]
    
SPACE = 'SPACE'
BLOCK = 'BLOCK'
GOAL = 'GOAL'

maze_grid = {(0, 0): SPACE, (0, 1): BLOCK, (0, 2): SPACE, (0, 3): SPACE, (0, 4): SPACE, 
             (0, 5): SPACE, (0, 6): SPACE, (0, 7): SPACE, (1, 0): SPACE, (1, 1): BLOCK,
             (1, 2): BLOCK, (1, 3): SPACE, (1, 4): BLOCK, (1, 5): BLOCK, (1, 6): BLOCK, 
             (1, 7): BLOCK, (2, 0): SPACE, (2, 1): BLOCK, (2, 2): SPACE, (2, 3): SPACE, 
             (2, 4): SPACE, (2, 5): SPACE, (2, 6): BLOCK, (2, 7): SPACE, (3, 0): SPACE, 
             (3, 1): SPACE, (3, 2): SPACE, (3, 3): BLOCK, (3, 4): BLOCK, (3, 5): SPACE, 
             (3, 6): BLOCK, (3, 7): SPACE, (4, 0): SPACE, (4, 1): BLOCK, (4, 2): SPACE, 
             (4, 3): BLOCK, (4, 4): SPACE, (4, 5): SPACE, (4, 6): SPACE, (4, 7): SPACE, 
             (5, 0): BLOCK, (5, 1): BLOCK, (5, 2): SPACE, (5, 3): BLOCK, (5, 4): SPACE, 
             (5, 5): BLOCK, (5, 6): SPACE, (5, 7): BLOCK, (6, 0): SPACE, (6, 1): BLOCK, 
             (6, 2): BLOCK, (6, 3): BLOCK, (6, 4): SPACE, (6, 5): BLOCK, (6, 6): SPACE, 
             (6, 7): SPACE, (7, 0): SPACE, (7, 1): SPACE, (7, 2): SPACE, (7, 3): SPACE, 
             (7, 4): SPACE, (7, 5): BLOCK, (7, 6): BLOCK, (7, 7): GOAL}

MazeActionMapping = Mapping[gridState, Mapping[int, Constant[Tuple[gridState, float]]]]

class gridMDP1(FiniteMarkovDecisionProcess[gridState, int]):
    def __init__(self, maze_grid: Dict[gridState, SPACE]):
        maze_states: Set[gridState] = set()
        
        for location in maze_grid.keys():
            locType = maze_grid[location]
            if locType != BLOCK:
                maze_states.add(location)
            if locType == GOAL:
                self.goal = location
                
        self.maze = maze_states
        
        super().__init__(self.get_action_transition_reward_map())
    
    def get_action_transition_reward_map(self) -> MazeActionMapping:
        
        d: Dict[gridState, Dict[int, Constant[Tuple[gridState, float]]]] = {}
        
        for currLoc in self.maze:
            if currLoc == self.goal:
                continue
            
            curr_loc_map: Dict[int, Constant[Tuple[gridState, float]]] = {}
            
            #New potential locations based on actions
            upLoc = (currLoc[0]-1, currLoc[1])
            downLoc = (currLoc[0]+1, currLoc[1])
            rightLoc = (currLoc[0], currLoc[1]+1)
            leftLoc = (currLoc[0], currLoc[1]-1)
            
            #UP -> Represented by action int = 0
            if upLoc in self.maze:
                curr_loc_map[0] = Constant((upLoc, -1))
            
            #DOWN -> Represented by action int = 1
            if downLoc in self.maze:
                curr_loc_map[1] = Constant((downLoc, -1))
               
            #RIGHT -> Represented by action int = 2
            if rightLoc in self.maze:
                curr_loc_map[2] = Constant((rightLoc, -1))
                
            #LEFT -> Represented by action int = 3
            if leftLoc in self.maze:
                curr_loc_map[3] = Constant((leftLoc, -1))
            
            d[gridState(currLoc)] = curr_loc_map
            
        return d
    
class gridMDP2(FiniteMarkovDecisionProcess[gridState, int]):
    def __init__(self, maze_grid: Dict[gridState, SPACE]):
        maze_states: Set[gridState] = set()
        
        for location in maze_grid.keys():
            locType = maze_grid[location]
            if locType != BLOCK:
                maze_states.add(location)
            if locType == GOAL:
                self.goal = location
                
        self.maze = maze_states
        
        super().__init__(self.get_action_transition_reward_map())
    
    def get_action_transition_reward_map(self) -> MazeActionMapping:
        
        d: Dict[gridState, Dict[int, Constant[Tuple[gridState, float]]]] = {}
        
        for currLoc in self.maze:
            if currLoc == self.goal:
                continue
            
            curr_loc_map: Dict[int, Constant[Tuple[gridState, float]]] = {}
            
            #New potential locations based on actions
            upLoc = (currLoc[0]-1, currLoc[1])
            downLoc = (currLoc[0]+1, currLoc[1])
            rightLoc = (currLoc[0], currLoc[1]+1)
            leftLoc = (currLoc[0], currLoc[1]-1)
            
            #UP -> Represented by action int = 0
            if upLoc in self.maze:
                if upLoc == self.goal:
                    curr_loc_map[0] = Constant((upLoc, 1))
                else:
                    curr_loc_map[0] = Constant((upLoc, 0))
                
            
            #DOWN -> Represented by action int = 1
            if downLoc in self.maze:
                if downLoc == self.goal:
                    curr_loc_map[1] = Constant((downLoc, 1))
                else:
                    curr_loc_map[1] = Constant((downLoc, 0))
               
            #RIGHT -> Represented by action int = 2
            if rightLoc in self.maze:
                if rightLoc == self.goal:
                    curr_loc_map[2] = Constant((rightLoc, 1))
                else:
                    curr_loc_map[2] = Constant((rightLoc, 0))
                
            #LEFT -> Represented by action int = 3
            if leftLoc in self.maze:
                if leftLoc == self.goal:
                    curr_loc_map[3] = Constant((leftLoc, 1))
                else:
                    curr_loc_map[3] = Constant((leftLoc, 0))
            
            d[gridState(currLoc)] = curr_loc_map
            
        return d
                                                  
 
mazeMDP1 = gridMDP1(maze_grid = maze_grid)
mazeMDP2 = gridMDP2(maze_grid = maze_grid)

#print("\Grid MDP with Reward Process 1: ")
#print("------------------")
#print(mazeMDP1)

#print("\Grid MDP with Reward Process 2: ")
#print("------------------")
#print(mazeMDP2)

print("Optimal Policy for MDP1")
print("--------------")
opt_vf_vi, opt_policy_vi_1 = value_iteration_result((mazeMDP1), gamma=1)
#pprint(opt_vf_vi)
print(opt_policy_vi_1)
print()
    
print("ptimal Policy for MDP2")
print("--------------")
opt_vf_vi, opt_policy_vi_2 = value_iteration_result((mazeMDP2), gamma=0.9)
#pprint(opt_vf_vi)
print(opt_policy_vi_2)
print()

EqualPolicy = True

for state in opt_policy_vi_1.action_for:
    if opt_policy_vi_1.action_for[state] != opt_policy_vi_2.action_for[state]:
        EqualPolicy = False
        print(state)




print("Optimal Policy MDP1 and Optimal Policy MDP2 are equal (BOOL): ")
print(EqualPolicy)

mazeV2MDP1 = gridMDP1(maze_grid = maze_grid)
mazeV2MDP2 = gridMDP2(maze_grid = maze_grid)

X = TypeVar('X')
Y = TypeVar('Y')
A = TypeVar('A')
S = TypeVar('S')

V = Mapping[NonTerminal[S], float]

DEFAULT_TOLERANCE = 1e-8

def tracked_converge(values: Iterator[X], done: Callable[[X, X], bool]) -> Iterator[X]:
    '''Read from an iterator until two consecutive values satisfy the
    given done function or the input iterator ends.
    Raises an error if the input iterator is empty.
    Will loop forever if the input iterator doesn't end *or* converge.
    '''
    a = next(values, None)
    if a is None:
        return

    yield a

    for i,b in enumerate(values):
        if done(a, b):
            print(f'took {i} iterations to converge')  ### This is the only part you needed to change
            return

        a = b
        yield b

def tracked_converged(values: Iterator[X],
              done: Callable[[X, X], bool]) -> X:
    '''Return the final value of the given iterator when its values
    converge according to the done function.
    Raises an error if the iterator is empty.
    Will loop forever if the input iterator doesn't end *or* converge.
    '''
    result = last(tracked_converge(values, done))

    if result is None:
        raise ValueError("converged called on an empty iterator")

    return result

def almost_equal_vfs(
    v1: V[S],
    v2: V[S],
    tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    '''Return whether the two value function tables are within the given
    tolerance of each other.
    '''
    return max(abs(v1[s] - v2[s]) for s in v1) < tolerance

def tracked_value_iteration_result(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> Tuple[V[S], FiniteDeterministicPolicy[S, A]]:
    opt_vf: V[S] = tracked_converged(
        value_iteration(mdp, gamma),
        done=almost_equal_vfs
    )
    opt_policy: FiniteDeterministicPolicy[S, A] = greedy_policy_from_vf(
        mdp,
        opt_vf,
        gamma
    )

    return opt_vf, opt_policy

vf_1, op_1 = tracked_value_iteration_result(mazeV2MDP1, gamma=1)
vf_2, op_2 = tracked_value_iteration_result(mazeV2MDP2, gamma=0.9)

print("\nMDP1 Iterations:")
print()

print("\nMDP2 Iterations:")
print()
