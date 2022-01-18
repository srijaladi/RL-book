Since we are focused on when $\gamma = 0$, we know that we are only focused on the next step of the process or the corresponding cost associated with moving to $s'$. We know that the corresponding cost of moving to $s'$ can be expressed as $e^{as'}$ where $a$ is a number that we select. Our goal here is to minimize the expected vlaue of $e^{as'}$ for all $s'$ given a choice of a.

First notice that $s' ~ N(s, \sigma^{2})$. Thus, $as'$ is distributed as $N(as, a^2 \cdot \sigma^{2})$. Essentially we are trying to minimize the expected value of $e^X$ where $X ~ N(as, a^2 \cdot \sigma^{2})$. This is a log-normal distribution, meaning that $E\{e^x\}$ = $e^{(\mu + 0.5 \cdot \sigma^{2})}$

We know that the mean of $X$ is $as$ and the standard deviation is $a^2 \cdot \sigma^{2}$, so the expected value can be expressed as:

$e^{(as + 0.5 \cdot (a^2 \cdot \sigma^{2}))}$. 

Given that we want to minimize this expected value, we are going to take the derivative of this with respect to a to get:

$(s + a \cdot \sigma^{2}) \cdot e^{(as + 0.5 \cdot (a^2 \cdot \sigma^{2}))}$. This equals 0 when $a = -\frac{s}{\sigma^2}$. Through plugging this in, we can verify that this does indeed create a minimum, and thus, to minimize the cost, the optimal action at any state is:

$a = -\frac{s}{\sigma^2}$. Further, the expected corresponding cost is:

$e^{(-\frac{s^2}{\sigma^2} + 0.5 \cdot \frac{s^2}{\sigma^2})} = e^{(0.5 \cdot \frac{s^2}{\sigma^2})}$