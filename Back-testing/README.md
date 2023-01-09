# Back-testing experimentation:
This folder includes scripts about back-testing with the Dollar Cost Averaging method.

Run the following command 

````shell
python Back_Testing_with_DCA.py
````

to obtain the back-testing results.  Note that Index strategies (Index-10 & Max-Ratio) might take ~15 min. Before running that, make sure the `Data.pkl` file is in the `./Data` folder.



This scripts includes three main functions:

| Function          | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| ShowEffectiveness | Generate the starting date list and call the strategy function |
| AtomStrategy      | The strategy of buying a single cryptocurrency               |
| IndexStrategy     | The strategy of buying multiple cryptocurrencies based on market capitalization |

More details about the above functions can be found in the code comments.
