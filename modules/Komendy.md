usage: gym_env.py [-h] [-eo] [-ne] [-s] [-p] [-np] [-mt MODELTYPE]
                  [-mn MODELNUM] [-t TRAIN] [-r REWARD] [-l] [-pi PATHINPUT]
                  [-po PATHOUTPUT]

Helper do skryptu

optional arguments:
  -h, --help            show this help message and exit
  -eo, --evalonly       Eval without train
  -ne, --noeval         Don't plot eval when training
  -s, --skip            Skip first plot
  -p, --plot            Start script to plot all gains from training
  -np, --noplot         Do not plot during training.
  -mt MODELTYPE, --modeltype MODELTYPE
                        Typ: PPO,DQN, default: PPO
  -mn MODELNUM, --modelnum MODELNUM
                        Model number
  -t TRAIN, --train TRAIN
                        Train amount
  -r REWARD, --reward REWARD
                        Wybierz reward function. Default: 1
  -l, --live            Start live. Add -pi & -po arguments.
  -pi PATHINPUT, --pathinput PATHINPUT
                        Ścieżka wejściowa do pliku Live
  -po PATHOUTPUT, --pathoutput PATHOUTPUT
                        Ścieżka wyjściowa do Live


# Wybór modelu
python gym_env.py -mt ppo
python gym_env.py -mt dqn

# Wybór mdelu i reward
python gym_env.py -mt dqn -r 2

# Rysowanie gainów
python gym_env.py -p

# Startowanie Live
python gym_env.py -l -pi P:\LocalPrograms\stock\friendly_solution_23-07\dane\test_updating.txt -po P:\LocalPrograms\stock\friendly_solution_23-07\dane\baseline_live.txt
python gym_env.py -l -pi P:\LocalPrograms\stock\friendly_solution_23-07\dane\test_updating.txt -po P:\LocalPrograms\stock\friendly_solution_23-07\dane\baseline_live.txt
