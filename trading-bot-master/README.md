## Getting Started

In order to use this project, you'll need to install the required python packages:

```bash
pip install -r requirements.txt
```

Now you can open up a terminal and start training the agent:

```bash
python train.py data/SS.csv data/SS_2018.csv data/economy_leading_2005.csv --strategy dqn
```

Once you're done training, run the evaluation script and let the agent make trading decisions:

```bash
python eval.py data/SS_2019.csv --model-name model_debug_50 --debug
```

Now you are all set up!

## Acknowledgements

- [@keon](https://github.com/keon) for [deep-q-learning](https://github.com/keon/deep-q-learning)
- [@edwardhdlu](https://github.com/edwardhdlu) for [q-trader](https://github.com/edwardhdlu/q-trader)

## References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Human Level Control Through Deep Reinforcement Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/)
- [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)