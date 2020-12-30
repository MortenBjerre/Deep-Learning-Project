# Deep Learning Project
## Generalization in video games
By Morten Bjerre (s174397) and Eric Jensen (s174379).

In this project we have used reinforcement learning and PPO to train an AI to play Starpilot. It is possible to try the game out yourself with the following commands:
```
$ pip install procgen # install
$ python -m procgen.interactive --env-name starpilot
```

We have mainly used google colab to create the code used for reinforcement learning, which can be found in Starpilot-PPO.ipynb. This notebook contains the entire code. For a more structural overview one can check out the AI python folder. The Starpilot-PPO script should run in colab without issues.

The Data folder contains all the experemental results where a small note has been added inside each folder to describe its content. All the results has been carried out using the DTU HPC. 

data-visualization.ipynb contains the script for producing the graphs included in the paper. One will have to download the data folder and change the paths so the location fits in order to recreate the plots.

A video of the AI playing the game Starpilot can be found here: https://www.youtube.com/watch?v=kpkCoP1g8O4
