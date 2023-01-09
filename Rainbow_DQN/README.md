# Training and Evaluation of Rainbow DQN

The main script is `rainbow.py`.

Before running it, make sure you register the CryptoEnv-v0 and run `pip install -r requirements.txt` to install the necessary packages.

You can run `rainbow.py` using the following command:

```shell
python rainbow.py --ExpID BTC_Exp_1 --frames 30000 --name 0 --wnd 30 --cycle 9 --memory_size 10000 --batch_size 128 --target_update 100 --gamma 0.95 --v_min 0 --v_max 20 --atom_size 51 --n_step 3 --data_-path 'path/to/price/data' --mode 0
```

, where `mode 0` for training and `mode 1` for evaluation.


After training, you can find the log file in `./logs`, and you can use the following command to visualize the evolution of the score. You might consider to change some of the source code of `rl_plooter`.

```shell
rl_plotter --show --save --avg_group --shaded_std --style default --title "Episode score v.s timesteps" --legend_outside --no_legend_group_num --resample 4096
```

# Reference:

[Deep Reinforcement Learning Course (simoninithomas.github.io)](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)

[Curt-Park/rainbow-is-all-you-need: Rainbow is all you need! A step-by-step tutorial from DQN to Rainbow (github.com)](https://github.com/Curt-Park/rainbow-is-all-you-need)

[深度强化学习调参技巧：以D3QN、TD3、PPO、SAC算法为例 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/345353294)

[gxywy/rl-plotter: A plotter for reinforcement learning (RL) (github.com)](https://github.com/gxywy/rl-plotter)

