#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 05:29:17 2022

@author: SriJaladi
"""

"""
In the most straightforward sense, if we are seeking an answer, the frog game
can be thought of as a simple dynamic programming problem working backwards.

In other words, to solve this problem (ignoring states/probabilities at this moment),
we can simply work backwards from the last leaf and go to the first leaf.

We can start by creating an array with 10 values, one for each leaf that the 
frog can land on (for simplicity assume the frog starts on a leaf). Each of these 
values (at the end) will represent the expected number of jumps the frog needs 
to reach the end from that specific leaf. We start this entire array at all 0s 
such that it looks like [0,0,0,0,0,0,0,0,0,0]

When examining the very LAST (10th) leaf, there is only one option for the frog
which is to jump directly to the end. This means the expected number of jumps for
the frog from the 10th leaf is 1. We can modify our array to match this so our array
would look like:
[0,0,0,0,0,0,0,0,0,1]

Now we work backwards. When looking at the 9th leaf, there are 2 choices for the frog, each
with equal probability. Either directly jump to the end (1 jump) or jump to the next
leaf (resulting in 1 jump + expected jumps from this leaf). Each of these occur
with probability 0.5 so the expected number of jumps from this (9th) leaf can be expressed 
as 1*0.5 + (1+E{Leaf10})*0.5 = 1.5. Thus our array becomes
[0,0,0,0,0,0,0,0,1.5,1]

Working backwards again, we look at the 8th leaf now. When examining the 8th leaf, there
are 3 choices, each with equal probability of 1/3. Thus, the expected number of jumps
from the 8th leaf can be expressed as 1*(1/3) + (1+E{Leaf9})*(1/3) + (1+E{Leaf10})*(1/3) =
(11/6). The array becomes:
[0,0,0,0,0,0,0,(11/6),1.5,1]

This process is continued until the first leaf is completed and since the frog starts on
the first leaf, the value at position 0 of the array is the answer to the expected
number of jumps the frog needs to reach the end.

The code implementation of this dynamic programming method is below

"""
import numpy as np

expected_arr = [0 for i in range(10)]

for i in range(9,-1,-1):
    trans_probability = 1/(10 - i)
    expected_sum = 1 #Start at one to represent case where frog jumps over everything
    
    for j in range(i+1,10):
        expected_sum += (1+(expected_arr[j]))
        
    expected_jumps = expected_sum * trans_probability #Just distributing out transition probabiltiy
    expected_arr[i] = expected_jumps
    
print(expected_arr)
print("Expected Jumps from original position: " + str(expected_arr[0]))

"""
We can also think of each Leaf as a Non-terminal state and the ground at the end as a
terminal state. For simplicity sake lets assume the frog is trying to get to Leaf11.
In this case, each state would simply be defined as a single leaf because this is
all that is needed to determine the probability distribution for the next state.

Lets label each state as Si where i is the number leaf that the state is represented
by. In this case, S0 is the original leaf the frog starts on and S10 is the end goal
the frog reaches.

S = {S0, S1, S2, S3...S10}
T = {S10} <- Terminal States, only S10
N = {S0, S1, S2, S3...S9} <- Non-Terminal States, everything but S9.

The transition map or transition probabilities can be created as shown below:
Where Xt = St
Each state probability can be expressed as P[X(t+1) | Xt] = 1/(n - Xt) where n = 10
"""

transition_map = {}

for state_i in range(10):
    state_prob_distr = {}
    max_future_transitions = 10 - state_i
    for i in range(state_i+1, 11):
        state_prob_distr[i] = (1/max_future_transitions)
    transition_map[state_i] = state_prob_distr

print("\nFull Transition Probability Map: ")
print(transition_map)

"""
We can create a simple simulation that tracks the number of transitions/jumps as follows
(10000 iterations and averaged):
"""

expectedJumps = 0
simulations = 10000

for i in range(0,simulations):
    totalJumps = 0
    curr_state = 0
    
    while (curr_state != 10):
        state_distr = transition_map[curr_state]
        keys = list(state_distr.keys())
        probs = list(state_distr.values())
        curr_state = np.random.choice(keys, 1, replace=True, p=probs)[0]
        totalJumps += 1
    
    expectedJumps += totalJumps
    
expectedJumps = expectedJumps/simulations
    
print("\nExpected Jumps: " + str(expectedJumps))

    


        
