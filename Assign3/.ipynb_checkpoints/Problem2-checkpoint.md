
The conditions we know are:

$P(s+1 | s,a) = a$

$P(s | s,a) = 1-a$

$R_{T}(s,a,s+1) = 1-a$

$R_{T}(s,a,s) = 1+a$


Using the formula for $V^{*}(s)$ and using $a$ as our action at each state gives:

$V^{*}(s) = max_{a\in [0,1]}(V^{a}(s))$ for all $s\in N$

Assuming a is any action to be taken, we can create a general value function as:

$V^{a}(s) = P(s,a,s+1) \cdot R_{T}(s,a,s+1) + P(s,a,s) \cdot R_{T}(s,a,s) + \gamma \cdot (P(s,a,s) \cdot V^{a}(s) + P(s,a,s+1) \cdot V^{a}(s+1))$

At this stage there are two things we are going to do: 
First, Replace all the P,R functions with their values from above.
Second, Notice that $V^{a}(s) = V^{a}(s+n)$ for any n, and replace $V^{a}(s+n)$ with $V^{a}(s)$. This gives a resulting function of:

$V^{a}(s) = a \cdot (1-a) + (1-a) \cdot (1+a) + \gamma \cdot ((1-a) \cdot V^{a}(s) + (a) \cdot V^{a}(s))$

$V^{a}(s) = a \cdot (1-a) + (1-a) \cdot (1+a) + \frac{1}{2} \cdot V^{a}(s)$

$\frac{1}{2} \cdot V^{a}(s) = 1 + a - 2a^{2}$

$V^{a}(s) = 2 + 2a - 4a^{2}$


Thus, to maximize the value function, we simply need to maximize the function:

$2 + 2a - 4a^{2}$

After taking the derivative, the maximum value of this function occurs when $a = 0.25$ and the value of the function at $a = 0.25$ is $0.5$. Thus, we now have the optimal value function and optimal policy at every state to be:

$V^{\pi(s) = 0.25}(s) = 0.5$ and $\pi^{*}(s) = 0.25$ at all s