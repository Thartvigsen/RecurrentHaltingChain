# Recurrent Halting Chain (RHC)

This repository contains the source code for our paper *Recurrent Halting Chain for Early Multi-label Classification*, published at **KDD, 2020**.
Please contact Tom Hartvigsen (twhartvigsen@wpi.edu) with any questions.

Examples of code use will be coming soon. For now, this model can effectively be used in
lieu of an RNN in pytorch:
```
from model import RHC

RHC = RHC(ninp, nhid, nclasses) # ninp is the number of variables, nhid is the hidden state size of the rnn, nclasses is the number of classes you want RHC to predict
logits = RHC(X) # where X is a tensor of time series of shape (timesteps, instances, variables)
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
