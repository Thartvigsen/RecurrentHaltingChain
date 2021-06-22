# Recurrent Halting Chain (RHC)

This repository contains the source code for our paper *Recurrent Halting Chain for Early Multi-label Classification*, published at **KDD, 2020**.
Please contact Tom Hartvigsen (twhartvigsen@wpi.edu) with any questions.

The key idea of our method is to step through the timesteps one by one, and when
the time is right for a given class (as determined by a Halting Policy), add it
to a growing set of class labels. We prefer both accurate label sets and early
predictions for each class. There are many ways to encode this key idea into a
classification architecture and in this work we pair an Order-Free Recurrent
Classifier Chain (a state-of-the-art multi-label model) with a
Reinforcement Learning-based Halting Policy (a state-of-the-art early
classifier).

Examples of code use will be coming soon. For now, this model can effectively be used in
lieu of an RNN in pytorch:
```
from model import RHC

RHC = RHC(ninp, nhid, nclasses) # ninp is the number of variables, nhid is the hidden state size of the rnn, nclasses is the number of classes you want RHC to predict
logits, mean_halting_point = RHC(X) # where X is a tensor of time series of shape (timesteps, instances, variables)
```

If you find this code helpful, feel free to cite our paper:
```
@inproceedings{hartvigsen2020recurrent,
  title={Recurrent Halting Chain for Early Multi-label Classification},
  author={Hartvigsen, Thomas and Sen, Cansu and Kong, Xiangnan and Rundensteiner, Elke},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1382--1392},
  year={2020}
}
```

**Tom Hartvigsen will be adding train/test scripts soon -- for now, please check out some analogous code for our [EARLIEST paper](https://github.com/Thartvigsen/EARLIEST)**
