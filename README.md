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
   git clone https://github.com/github_username/repo_name.git
   ```
2. Install requirements.txt. Note that you may have some problems with plotting on backtrader to avoid this try: 
```sh
   pip uninstall backtrader
```
```sh
   pip install git+https://github.com/mementum/backtrader.git@0fa63ef4a35dc53cc7320813f8b15480c8f85517#egg=backtrader
  ```

## Workflow
To run backtest right away jump into main.py, set params that you need for backtest (including model, training days, rebelance days, printlog e.t.c.)
and you are good to go. As the return you will get final value of your portfolio and buy/sell orders plotted on the graph as well as change in your funds. 

To implement custom rebalancing models jump into backtest.py. And add your model to get_allocations func (as template use already implemented strategies).

More features will be added soon : )

## Supported model list
(**in development**)

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/daniil-morozkov/





