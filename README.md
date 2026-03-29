# Deep Hedging - Financial Derivatives, Oxford

A framework for hedging a derivatives portfolio under market frictions (transaction costs) using deep reinforcement learning methods.

## Overview
1. **Market Model**: Heston stochastic volatility model with Broadie-Kaya sampling.
2. **Hedging Instruments**: stock, variance swap; processes modeled by Heston SDEs.
3. **Network Architecture**: 30 semi-recurrent neural networks, one per trading day, each with two hidden layers.
4. **Risk Measures**: CVaR (expected shortfall) and entropic (exponential utility); loss modeled by P&L.
5. **Benchmark**: model delta hedge via QuantLib's Heston engine.

Reference paper attached (https://arxiv.org/pdf/1802.03042). 
