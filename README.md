[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/morozkovda/Simple_Portfolio_Optimization_Backtest">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>


<h3 align="center">Simple Portfolio Backtest</h3>

  <p align="center">
   <p align="center">
This is a simple script to test your portfolio on real data using preset or custom portfolio optimization strategies

</div>

## Getting started
1. Clone the repo
   ```sh
   git clone https://github.com/morozkovda/Simple_Portfolio_Optimization_Backtest.git
   ```
2. Install requirements.txt. Note that you may have some problems with plotting on backtrader to avoid this try: 
```sh
   pip uninstall backtrader
```
```sh
   pip install git+https://github.com/mementum/backtrader.git@0fa63ef4a35dc53cc7320813f8b15480c8f85517#egg=backtrader
  ```

## Workflow
To run backtest right away jump into main.py, set params that you need for backtest (including list of models, training days, rebelance days, printlog e.t.c.)
and you are good to go. As the output you will get stats, return and cumulative return summary as separate CSV's for each model and one that contains data on all the models, the results are located in folders as follows.

To implement custom rebalancing models jump into backtest.py. And add your model to get_allocations func (as template use already implemented strategies).

To graph all the model jump into graphs.py.

More features will be added soon : )

## Supported model list
['MV','CLA', 'HRP', 'risk_parity', 'cvar','cdar','random','equal']

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/daniil-morozkov/





