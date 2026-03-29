## Deep Hedging Implementation: derived from "Deep Hedging" - Oxford, JPMorgan Chase (2018)

A framework for hedging a derivatives portfolio under market frictions (mainly transaction costs) using deep reinforcement learning methods.
Authors: Akshu Enaganti, Leandro Sanchez-Betancourt

### Step 1: Heston Model Simulation

The Heston model is a continuous time stochastic volatility model dependent on 2 stochastic differential equations (one for stock price, and one for variance). 

Stock Price Process: $$dS^1_t = \sqrt{V_t} S^1_t dB_t$$

Variance Process (CIR): $$dV_t = \kappa(\theta - V_t)dt + \sigma\sqrt{V_t}dW_t$$

where $B_t$, $W_t$ are Brownian motions with correlation $\rho$, i.e. $d\langle B, W \rangle_t = \rho \, dt$.

The variance is sampled from its true conditional distribution, a scaled noncentral chi-squared, using the Broadie-Kaya sampling scheme (referenced in paper footnote 5).

Variance Process (Broadie-Kaya sampling):
$$V_{k+1} | V_k \sim c \cdot \chi^2(df, \lambda)$$
$$c = \frac{\sigma^2(1-e^{-\kappa \Delta t})}{4\kappa}, \quad df = \frac{4\kappa\theta}{\sigma^2}, \quad \lambda = \frac{V_k e^{-\kappa \Delta t}}{c}$$

The stock price is sampled using a simplified Broadie-Kaya scheme, replacing $V$ with a piecewise, constant process between timesteps.
$$\log\frac{S^1_{k+1}}{S^1_k} = \frac{\rho}{\sigma}\left(V_{k+1} - V_k - \kappa\theta\Delta t + \kappa V_k \Delta t\right) - \frac{1-\rho^2}{2} V_k \Delta t + \sqrt{(1-\rho^2) V_k \Delta t} \cdot Z$$

where $Z \sim \mathcal{N}(0,1)$.

Aside from the stock itself, the other tradeable instrument is a variance swap, or a contract whose price reflects total expected realized variance over [0, T], given a current variance $V_t$. The equation for the variance swap price is shown below.

Variance Swap Price (equation 5.4): $$S^2_t = \int_0^t V_s \, ds + L(t, V_t),$$ $$L(t, v) = \frac{v - \theta}{\kappa}\left(1 - e^{-\kappa(T-t)}\right) + \theta(T-t)$$

where the integral $\int_0^t V_s\, ds$ is approximated as a running sum $\sum_{j=0}^{k-1} V_j \Delta t$.

Simulating the Heston model gives a market model to exercise and validate trading strategies within.

### Step 2: Recurrent Neural Network 

In markets with friction, the optimal hedging strategy $\delta^*$ is an unknown function from market state to optimal holdings, at each timestep. The goal of the neural networks is to approximate $\delta^*$, as analytically solving for it is implausible. From Hornik's 1991 work, any continuous function can be approximated by a sufficiently large neural network, arbitrarily well enough.

Each network, at each timestep $k$, takes in current market information and the previous position ($\delta_{k-1}$), outputting the new position. 
$$\delta_k = F^{\theta_k}(I_k, \delta_{k-1}),$$
$$I_k = (\log S^1_k, V_k)$$

Transaction costs are dependent upon the change in position, necessitating the inclusion of $\delta_{k-1}$ in the networks.
$$C_T(\delta) = \sum_{k=0}^{n} \varepsilon |\delta_k - \delta_{k-1}| \cdot S^i_k$$

$n = 30$ distinct networks are used, one per trading day; each network independently learns its parameters $\theta_k$, modeling the optimal strategy on day $k$ (which, naturally, differs from the optimal strategy on day $k+1$). These networks are chained together; the output $\delta_k$ of network $k$ feeds as input into network $k+1$, allowing gradients to flow back through time through training.

### Step 3: Convex Risk Measures

Consider the terminal profit and loss (P&L) under market frictions (sum of gains/losses from trades, premium charged, call option payoff cost, and transaction costs).
$$PL_T(Z, p_0, \delta) = -Z + p_0 + (\delta \cdot S)_T - C_T(\delta)$$

Regardless of what strategy is used, the terminal P&L will be random (due to the derivative prices in trades), and this demands a measure of how desirable a distribution of outcomes is. A convex risk measure $\rho(X)$ is defined as the minimum amount of cash that must be injected into position $X$ to make it "acceptable" ($\rho(X) \leq 0$). The convex risk measure satisfies three properties:
- Monotone: if X is always a better position than Y, X must require less cash injection
- Convex: a portfolio that is half A, half B is not riskier than the average individual risk of A and B 
- Cash-invariant: adding $\$c$ in cash reduces risk by exactly $\$c$ 

In turn, this creates the optimization problem (and loss function) for the training procedure.
$$\pi(X) = \inf_{\delta \in H} \rho\left(X + (\delta \cdot S)_T - C_T(\delta)\right)$$

$\pi(X)$ then represents the minimum achievable risk given an asset position $X$. This acts as the loss function; the networks attempt to find a trading strategy minimizing risk. The indifference price, defined as $p(Z) = \pi(-Z) - \pi(0)$, is the extra risk incurred from selling Z (difference of min. risk faced when selling derivative/hedging and min. risk faced when trading optimally distinctly). This indifference price, as shown later, is computed using two separately trained models: one that minimizes risk under liability $Z$ for $\pi(-Z)$, and one that minimizes risk with no liability for $\pi(0)$. $p(Z)$ is typically charged to the client as a premium to compensate for the risk. To verify this, consider the case where $C_T = 0$. Then, $PL = 0$ with the optimal trading strategy, giving $\pi(-Z) = p_0 + \pi(0)$ or $p(Z) = p_0$ (the standard frictionless market result).

Two different risk measures are used in this implementation, as detailed below.

- CVaR: represents the expected shortfall or tail risk; average loss over worst $\alpha\%$ of outcomes. A higher $\alpha$ means a more risk averse position, where $\alpha = 0.5$ recovers behaviour close to the model delta hedge.
$$\rho(X) = \frac{1}{1-\alpha} \int_0^{1-\alpha} \text{VaR}_\gamma(X) \, d\gamma$$

- Entropic: similar to exponential utility. A higher $\lambda$ means a more risk averse position; a nice result is that as $\lambda \rightarrow 0$, $\rho(X) \rightarrow -E[X]$, which is the pure expected value of the asset without considering risk.
$$\rho(X) = \frac{1}{\lambda} \log \mathbb{E}\left[e^{-\lambda X}\right]$$

### Step 4: Model Training

The training loop is where the model "learns" the art of approximating an optimal strategy, after which its results can be evaluated in the testing stage. The full training loop is described below.

1. Sample a batch of market paths from Heston model simulator
2. Run each path through the 30 networks, outputting holdings $\delta_k$ at each timestep
3. Compute terminal P&L for batch
4. Evaluate risk measure $\rho$ on distribution of P&L outcomes
5. Backpropagate to update network weights based on loss function (Adam optimizer)

As mentioned earlier, terminal P&L is defined as follows. 
$$PL_T(Z, p_0, \delta) = -Z + p_0 + (\delta \cdot S)_T - C_T(\delta)$$

The term $(\delta \cdot S)_T$ simply represents cumulative trading gains, broken down below.
$$(\delta \cdot S)_T = \sum_{k=0}^{n-1} \delta_k \cdot (S_{k+1} - S_k)$$

To model transaction costs, a proportional model is used, where transaction costs are a function of the change in position and asset price. 
$$C_T(\delta) = \sum_{k=0}^{n} \varepsilon \sum_{i=1}^{d} |\delta^i_k - \delta^i_{k-1}| \cdot S^i_k$$



The loss function is a chosen risk measure, evaluated on the terminal P&L distribution across batch. A learning rate of 0.005 and batch size of 256 are used as standard hyperparameters for neural network training. The parameters are updated via the Adam optimizer, which allows for an adaptive learning rate across different parameters and tracks a momentum that simplifies gradient descent with magnitude.

### Step 5: Results

After the deep learning model is trained on the Heston model paths, its performance is evaluated using 3 experiments, to verify the results against those achieved by the researchers.

1. **No Transaction Costs**: the deep hedging model is evaluated against the classical model delta hedge, computed via the QuantLib library. A market without transaction costs (frictionless market) implies the classical model delta hedge is the "benchmark" hedge that the network should aim to learn. At $\alpha=0.5$ CVaR the deep hedge should reproduce the model hedge almost perfectly. The equation is shown below, where $u(t, s, v)$ is the Heston model option price and $L(t, v)$ is the variance swap formula (equation 5.4).

$$\delta^1_t = \partial_s u(t, S^1_t, V_t), \quad \delta^2_t = \frac{\partial_v u(t, S^1_t, V_t)}{\partial_v L(t, V_t)}$$

2. **Risk Aversion**: two distinct CVaR levels are compared ($\alpha = 0.5$ vs $\alpha = 0.99$) to gather insight into how the model reacts with greater risk aversion. With a higher risk aversion factor, the model should adapt a more conservative strategy, leading to a higher indifference price and smaller tail losses (but a larger mean loss).

3. **Transaction Costs**: after training at different $\varepsilon$ values, $\log(p_\varepsilon - p_0)$ is plotted against $\log(\varepsilon)$, and should produce a line of slope $\approx 2/3$. This number is quite prevalent in problems involving proportional transaction costs, and serves as a strong validator of the model's predictions.
