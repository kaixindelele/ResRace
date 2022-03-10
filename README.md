# Residual Policy Learning Facilitates Efficient Model-Free Autonomous Racing
The official code for model-free autonomous racing algorithm ResRace. 

<img src="demos/montreal.gif" width="300"> <img src="demos/plechaty.gif" width="440"> 

<img src="demos/barcelona.gif" width="745"> 

## Dependencies
Our code has been tested on `Ubuntu 18.04` with `Python 3.6` and `Tensorflow 2.3.0`. 

The simulator used in this repository is Axel Brunnbauer's [racecar_gym](github.com/axelbr/racecar_gym). 

PPO and TRPO are based on [OpenAI SpinningUp](spinningup.openai.com/en/latest/) 

```
git clone https://github.com/openai/spinningup.git 
cd spinningup 
pip install -e . 
pip install tensorflow==2.3.0 
```
