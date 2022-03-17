from dataclasses import dataclass
from typing import Callable, Tuple, Iterator, Sequence, List
import numpy as np
from rl.dynamic_programming import V
from scipy.stats import norm
from rl.markov_decision_process import Terminal, NonTerminal
from rl.policy import FiniteDeterministicPolicy
from rl.distribution import Constant, Categorical
from rl.finite_horizon import optimal_vf_and_policy

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
from rl.function_approx import LinearFunctionApprox, DNNApprox
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
class OptimalExerciseBinTree:

    spot_price: float
    payoff: Callable[[float, float], float]
    expiry: float
    rate: float
    vol: float
    num_steps: int

    def european_price(self, is_call: bool, strike: float) -> float:
        sigma_sqrt: float = self.vol * np.sqrt(self.expiry)
        d1: float = (np.log(self.spot_price / strike) +
                     (self.rate + self.vol ** 2 / 2.) * self.expiry) \
            / sigma_sqrt
        d2: float = d1 - sigma_sqrt
        if is_call:
            ret = self.spot_price * norm.cdf(d1) - \
                strike * np.exp(-self.rate * self.expiry) * norm.cdf(d2)
        else:
            ret = strike * np.exp(-self.rate * self.expiry) * norm.cdf(-d2) - \
                self.spot_price * norm.cdf(-d1)
        return ret

    def dt(self) -> float:
        return self.expiry / self.num_steps

    def state_price(self, i: int, j: int) -> float:
        return self.spot_price * np.exp((2 * j - i) * self.vol *
                                        np.sqrt(self.dt()))

    def get_opt_vf_and_policy(self) -> \
            Iterator[Tuple[V[int], FiniteDeterministicPolicy[int, bool]]]:
        dt: float = self.dt()
        up_factor: float = np.exp(self.vol * np.sqrt(dt))
        up_prob: float = (np.exp(self.rate * dt) * up_factor - 1) / \
            (up_factor * up_factor - 1)
        return optimal_vf_and_policy(
            steps=[
                {NonTerminal(j): {
                    True: Constant(
                        (
                            Terminal(-1),
                            self.payoff(i * dt, self.state_price(i, j))
                        )
                    ),
                    False: Categorical(
                        {
                            (NonTerminal(j + 1), 0.): up_prob,
                            (NonTerminal(j), 0.): 1 - up_prob
                        }
                    )
                } for j in range(i + 1)}
                for i in range(self.num_steps + 1)
            ],
            gamma=np.exp(-self.rate * dt)
        )

    def option_exercise_boundary(
        self,
        policy_seq: Sequence[FiniteDeterministicPolicy[int, bool]],
        is_call: bool
    ) -> Sequence[Tuple[float, float]]:
        dt: float = self.dt()
        ex_boundary: List[Tuple[float, float]] = []
        for i in range(self.num_steps + 1):
            ex_points = [j for j in range(i + 1)
                         if policy_seq[i].action_for[j] and
                         self.payoff(i * dt, self.state_price(i, j)) > 0]
            if len(ex_points) > 0:
                boundary_pt = min(ex_points) if is_call else max(ex_points)
                ex_boundary.append(
                    (i * dt, opt_ex_bin_tree.state_price(i, boundary_pt))
                )
        return ex_boundary


if __name__ == '__main__':
    from rl.gen_utils.plot_funcs import plot_list_of_curves
    spot_price_val: float = 100.0
    strike: float = 100.0
    is_call: bool = False
    expiry_val: float = 1.0
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_steps_val: int = 300

    if is_call:
        opt_payoff = lambda _, x: max(x - strike, 0)
    else:
        opt_payoff = lambda _, x: max(strike - x, 0)

    opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree(
        spot_price=spot_price_val,
        payoff=opt_payoff,
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        num_steps=num_steps_val
    )

    vf_seq, policy_seq = zip(*opt_ex_bin_tree.get_opt_vf_and_policy())
    ex_boundary: Sequence[Tuple[float, float]] = \
        opt_ex_bin_tree.option_exercise_boundary(policy_seq, is_call)
    time_pts, ex_bound_pts = zip(*ex_boundary)
    label = ("Call" if is_call else "Put") + " Option Exercise Boundary"
    plot_list_of_curves(
        list_of_x_vals=[time_pts],
        list_of_y_vals=[ex_bound_pts],
        list_of_colors=["b"],
        list_of_curve_labels=[label],
        x_label="Time",
        y_label="Underlying Price",
        title=label
    )

    european: float = opt_ex_bin_tree.european_price(is_call, strike)
    print(f"European Price = {european:.3f}")

    am_price: float = vf_seq[0][NonTerminal(0)]
    print(f"American Price = {am_price:.3f}")
 
    
def ftr_func(x, strike = 100):
    return [i  for i in range(1,strike)]

all_ftrs = []
for currS in OptimalExerciseBinTree.state_price(10, 5, 2):
    all_ftrs.append([ftr_func((currS, a)) for a in [0,1]])
    

#print(all_ftrs)


ffs = [(lambda x, s=s: all_ftrs[x[0].state.quality][x[1]][s]) for s in OptimalExerciseBinTree.non_terminal_states]   

q_ag: AdamGradient = AdamGradient(
   learning_rate=0.05,
    decay1=0.9,
    decay2=0.999
)

q_func_approx: QValueFunctionApprox[NonTerminal[OptimalExerciseBinTree], A]
q_func_approx = DNNApprox.create(feature_functions=ffs,adam_gradient=q_ag)


episodeLen = 10
numEpisodes = 10000
epsilon = 0.1

for episode in range(numEpisodes):    
    
    startState = NonTerminal(OptimalExerciseBinTree())
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
        transitionStep = OptimalExerciseBinTree.step(currState, currAction).sample()
        
        nextState = transitionStep[0]
        reward = transitionStep[1]
        
        #Find best action nextState
        bestNextAction = 1
        if (q_func_approx((nextState,0)) > q_func_approx((nextState,1))):
            bestNextAction = 0
        
        #Calculate and hold the target for updating purposes
        target = reward + q_func_approx((nextState, bestNextAction))
        
        #Pass in the curent (s,a) pair and target value for update
        q_func_approx = q_func_approx.update([((currState, currAction), target)])
        currState = nextState #Iterate the currState to the nextState
        
        
q_optimal_policy: Mapping[NonTerminal[OptimalExerciseBinTree], int] = {}
q_optimal_vf: Mapping[NonTerminal[OptimalExerciseBinTree], float] = {}

for s in OptimalExerciseBinTree.non_terminal_states:
    maxAction = 1
    if (q_func_approx((s, 0)) > q_func_approx((s, 1))):
        maxAction = 0
    q_optimal_policy[s] = maxAction
    q_optimal_vf[s] = q_func_approx((s, maxAction))
    

print()
print("Optimal Policy Q-Learning at every state")
print("--------------\n")
for s in OptimalExerciseBinTree.non_terminal_states:
    print("For State " + str(s.state) + ": Do Action " + str(q_optimal_policy[s]))

print()
print("Optimal Value Function Q-Learning at every state")
print("--------------\n")
for s in OptimalExerciseBinTree.non_terminal_states:
    print("For State " + str(s.state) + ": VF " + str(q_optimal_vf[s]))
    
    
