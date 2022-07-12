# Bellman Complete Representation Learning (BCRL)

Implementation of BCRL, an algorithm for learning linear Bellman Complete representations with coverage, for offline policy evaluation (OPE). 

## Quick Start
First install [DeepMind Control Suite (dm_env)](https://github.com/deepmind/dm_env),
[PyTorch](https://pytorch.org/),
[Hydra](https://github.com/facebookresearch/hydra). Then, run
```
pip install -e .
```
For example, to run on environment cheetah run, 
```
python run.py task=cheetah_run
```

## Environments Supported
This repository supports 4 DeepMind Control Suite environments:
1. Finger Turn Hard
2. Cheetah Run
3. Quadruped Walk
4. Humanoid Stand


## Paper and Citation
To cite this work, please use the following citation. 
<br>
Chang, J., Wang, K., Kallus, N. &amp; Sun, W.. (2022). 
Learning Bellman Complete Representations for Offline Policy Evaluation. 
<i>Proceedings of the 39th International Conference on Machine Learning</i>,
in <i>Proceedings of Machine Learning Research</i> 162:2938-2971.
<br>Available from https://proceedings.mlr.press/v162/chang22b.html.

BibTex:
```
@InProceedings{pmlr-v162-chang22b,
  title = 	 {Learning {B}ellman Complete Representations for Offline Policy Evaluation},
  author =       {Chang, Jonathan and Wang, Kaiwen and Kallus, Nathan and Sun, Wen},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {2938--2971},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/chang22b/chang22b.pdf},
  url = 	 {https://proceedings.mlr.press/v162/chang22b.html},
  abstract = 	 {We study representation learning for Offline Reinforcement Learning (RL), focusing on the important task of Offline Policy Evaluation (OPE). Recent work shows that, in contrast to supervised learning, realizability of the Q-function is not enough for learning it. Two sufficient conditions for sample-efficient OPE are Bellman completeness and coverage. Prior work often assumes that representations satisfying these conditions are given, with results being mostly theoretical in nature. In this work, we propose BCRL, which directly learns from data an approximately linear Bellman complete representation with good coverage. With this learned representation, we perform OPE using Least Square Policy Evaluation (LSPE) with linear functions in our learned representation. We present an end-to-end theoretical analysis, showing that our two-stage algorithm enjoys polynomial sample complexity provided some representation in the rich class considered is linear Bellman complete. Empirically, we extensively evaluate our algorithm on challenging, image-based continuous control tasks from the Deepmind Control Suite. We show our representation enables better OPE compared to previous representation learning methods developed for off-policy RL (e.g., CURL, SPR). BCRL achieve competitive OPE error with the state-of-the-art method Fitted Q-Evaluation (FQE), and beats FQE when evaluating beyond the initial state distribution. Our ablations show that both linear Bellman complete and coverage components of our method are crucial.}
}
```
