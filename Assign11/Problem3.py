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
                                        FiniteMarkovRewardProcess)
from rl.dynamic_programming import value_iteration, almost_equal_vfs, greedy_policy_from_vf
from pprint import pprint
import rl.markov_process as mp
import rl.markov_process as mp
from rl.approximate_dynamic_programming import (ValueFunctionApprox,
                                                QValueFunctionApprox,
                                                NTStateDistribution)
from collections import defaultdict
from rl.chapter10.prediction_utils import (
    mc_finite_equal_wts_correctness,
    mc_finite_learning_rate_correctness,
    td_finite_learning_rate_correctness,
    td_lambda_finite_learning_rate_correctness,
    compare_td_and_mc
)
from rl.markov_process import TransitionStep
from Assign11.Problem1 import (TabularMCPredict, updateTabularMC)
from Assign11.Problem2 import (TabularTDPredict, updateTabularTD)
import rl.monte_carlo
from rl.monte_carlo import (mc_prediction)
import rl.td
from rl.td import td_prediction

S = TypeVar('S')
A = TypeVar('A')

from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_process import MarkovRewardProcess
from rl.markov_process import FiniteMarkovRewardProcess
from rl.markov_process import State, NonTerminal
from scipy.stats import poisson
from rl.distribution import SampledDistribution, Categorical, \
    FiniteDistribution
import numpy as np


@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


class SimpleInventoryMRP(MarkovRewardProcess[InventoryState]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

    def transition_reward(
        self,
        state: NonTerminal[InventoryState]
    ) -> SampledDistribution[Tuple[State[InventoryState], float]]:

        def sample_next_state_reward(state=state) ->\
                Tuple[State[InventoryState], float]:
            demand_sample: int = np.random.poisson(self.poisson_lambda)
            ip: int = state.state.inventory_position()
            next_state: InventoryState = InventoryState(
                max(ip - demand_sample, 0),
                max(self.capacity - ip, 0)
            )
            reward: float = - self.holding_cost * state.on_hand\
                - self.stockout_cost * max(demand_sample - ip, 0)
            return NonTerminal(next_state), reward

        return SampledDistribution(sample_next_state_reward)


class SimpleInventoryMRPFinite(FiniteMarkovRewardProcess[InventoryState]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_transition_reward_map())

    def get_transition_reward_map(self) -> \
            Mapping[
                InventoryState,
                FiniteDistribution[Tuple[InventoryState, float]]
            ]:
        d: Dict[InventoryState, Categorical[Tuple[InventoryState, float]]] = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state = InventoryState(alpha, beta)
                ip = state.inventory_position()
                beta1 = self.capacity - ip
                base_reward = - self.holding_cost * state.on_hand
                sr_probs_map: Dict[Tuple[InventoryState, float], float] =\
                    {(InventoryState(ip - i, beta1), base_reward):
                     self.poisson_distr.pmf(i) for i in range(ip)}
                probability = 1 - self.poisson_distr.cdf(ip - 1)
                reward = base_reward - self.stockout_cost *\
                    (probability * (self.poisson_lambda - ip) +
                     ip * self.poisson_distr.pmf(ip))
                sr_probs_map[(InventoryState(0, beta1), reward)] = probability
                d[state] = Categorical(sr_probs_map)
        return d

if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.5

    si_mrp = SimpleInventoryMRPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )

    from rl.markov_process import FiniteMarkovProcess
    print("Transition Map")
    print("--------------")
    print(FiniteMarkovProcess(
        {s.state: Categorical({s1.state: p for s1, p in v.table().items()})
         for s, v in si_mrp.transition_map.items()}
    ))

    print("Transition Reward Map")
    print("---------------------")
    print(si_mrp)

    print("Stationary Distribution")
    print("-----------------------")
    si_mrp.display_stationary_distribution()
    print()

    print("Reward Function")
    print("---------------")
    si_mrp.display_reward_function()
    print()

    print("Value Function")
    print("--------------")
    si_mrp.display_value_function(gamma=user_gamma)
    print()
    
    totalAdd = 100000
    perAdd = 10000
    episode_length = perAdd
    
    episodes: Iterable[Iterable[TransitionStep[S]]] = si_mrp.reward_traces(Choose(si_mrp.non_terminal_states))
    curtailed_episodes: Iterable[Iterable[TransitionStep[S]]] = \
        (itertools.islice(episode, episode_length) for episode in episodes)
    
    TDSequence: List[TransitionStep[S]] = []
    MCSequence: List[List[TransitionStep[S]]] = []
    
    count = 0
    
    for potent in curtailed_episodes:
        tempMCSequence: List[TransitionStep[S]] = []
        tempCount = 0
        
        countCheck = False
        
        for step in potent:
            count += 1
            tempCount += 1
            tempMCSequence.append(step)
            TDSequence.append(step)
            
            if count == totalAdd:
                countCheck = True
                break
            if tempCount == perAdd:
                break
        
        MCSequence.append(tempMCSequence)
        
        if countCheck:
            break
        
    print("TDTotal Transition Steps is: " + str(len(TDSequence)))
    print("MCTotal Transition Steps is: " + str(len(MCSequence) * len(MCSequence[0])))
    print()
        
    TD_V: Mapping[S, float] = TabularTDPredict(TDSequence, user_gamma)
    MC_V: Mapping[S, float] = TabularMCPredict(MCSequence, user_gamma)
    blank_NT_mapping: Mapping[NonTerminal[S], float] = {s:0 for s in si_mrp.non_terminal_states}
    
    FA_TD_V: Iterable[ValueFunctionApprox[S]] = td_prediction(TDSequence, blank_NT_mapping, user_gamma)
    FA_MC_V: Iterable[ValueFunctionApprox[S]] = mc_prediction(MCSequence, blank_NT_mapping, user_gamma)
    
    print("Tabular TD Value Function Results")
    print("--------------")
    for state in TD_V.keys():
        value = TD_V[state]
        print("State: " + str(state) + " -> Value: " + str(value))
    print()
        
    
    print("Tabular MC Value Function Results")
    print("--------------")
    for state in MC_V.keys():
        value = MC_V[state]
        print("State: " + str(state) + " -> Value: " + str(value))
        
    #for i in FA_TD_V.keys():
    #    print(i)
    
    """
    print("Function Approx MC Value Function Results")
    print("--------------")
    for state in MC_V.keys():
        value = MC_V[state]
        print("State: " + str(state) + " -> Value: " + str(value))
        
    print("Function Approx MC Value Function Results")
    print("--------------")
    for state in MC_V.keys():
        value = MC_V[state]
        print("State: " + str(state) + " -> Value: " + str(value))
    """
            
        
            
            



    
    
    