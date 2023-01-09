# Training and Evaluation of Rainbow DQN

The main script is `rainbow.py`.

Before running it, make sure you register the CryptoEnv-v0 and run `pip install -r requirements.txt` to install the necessary packages.

You can run `rainbow.py` using the following command:

```shell
python rainbow.py --ExpID BTC_Exp_1 --frames 30000 --name 0 --wnd 30 --cycle 9 --memory_size 10000 --batch_size 128 --target_update 100 --gamma 0.95 --v_min 0 --v_max 20 --atom_size 51 --n_step 3 --data_-path 'path/to/price/data' --mode 0
```

, where `mode 0` for training and `mode 1` for evaluation.