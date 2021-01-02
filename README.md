# Deep Learning Project
## Generalization in video games
### Project in 02456 Deep Learning at DTU
By Morten Bjerre (s174397) and Eric Jensen (s174379).

In this project we have used reinforcement learning and PPO to train an AI to play Starpilot. It is possible to try the game out yourself with the following commands:
```
$ pip install procgen # install
$ python -m procgen.interactive --env-name starpilot
```

We have used google colab to create the code used for reinforcement learning, which can be found in Starpilot-PPO.ipynb in the AI notebooks folder or [here](https://colab.research.google.com/drive/1no8neo9IY6Uq3eBny_dWUHTEhJJ48OwF?usp=sharing). This notebook contains the entire code. For a more structural overview one can check out the AI python folder. The Starpilot-PPO script should run in colab without issues.

The Data folder found [here](https://drive.google.com/drive/folders/1lfRfz9HO6znKIrBKwqXkFoJPtO9PJJK9?usp=sharing) contains all the experemental results. All the results has been carried out using the DTU HPC. For each test the policy, training data, hyperparameters and a video of the AI playing has been saved.

In the AI notebooks folder, data-visualization.ipynb contains the script for producing the graphs included in the paper. It can also be viewed in colab [here](https://colab.research.google.com/drive/1cfZtBI2A3bHGLxOaZtDDSZNVcxY8qDPU?usp=sharing). One will have to download the data folder and change the paths so the location fits in order to recreate the plots.

A video of the AI playing the game Starpilot can be found here: https://www.youtube.com/watch?v=kpkCoP1g8O4
