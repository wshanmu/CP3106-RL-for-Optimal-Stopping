# How to register the `CryptoEnv`:

1. install gym

   ```shell
   pip install gym
   ```

2. find the location of gym

   1. if you are no sure, use the command `pip install gym` again

3. enter `./gym/envs/`

4. creater new folde `user`

5. move CrytoEnv.py to the `user` folder

6. set up `__init__.py`:

   ```python
   from gym.envs.user.CryptoEnv import CryptoEnv
   ```

7. back to the upper folder: `./gym/envs/`

8. open  `__init__.py`, add:

   ```python
   register(
   	id='CryptoEnv-v0',
   	entry_point='gym.envs.user:CryptoEnv',
   )
   ```

- Directory structure:

```
|--  path to gym
|    |-- envs
|    |   |-- user
|    |   |   |-- CryptoEnv.py
|    |   |   |-- __init__.py
|    |   |-- __init__.py
```