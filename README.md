# A deep generative model for probabilistic energy forecasting in power systems: normalizing flows
Official implementation of generative models to compute scenario of renewable generation and consumption on the GEFcom2014 open dataset presented in the paper:
> Dumas, Jonathan, et al. "A deep generative model for probabilistic energy forecasting in power systems: normalizing flows." arXiv preprint arXiv:2106.09370 (2021).
> [[arxiv]](https://arxiv.org/abs/2106.09370)

Note: this paper is under review for Applied Energy.

## Cite

If you make use of this code, please cite our arXiv paper:

```
@article{dumas2021deep,
  title={A deep generative model for probabilistic energy forecasting in power systems: normalizing flows},
  author={Dumas, Jonathan and Lanaspeze, Antoine Wehenkel Damien and Corn{\'e}lusse, Bertrand and Sutera, Antonio},
  journal={arXiv preprint arXiv:2106.09370},
  year={2021}
}
```

Note: the reference will be changed if the paper is accepted for publication in Applied Energy

# Framework of the study
![strategy](https://github.com/jonathandumas/generative-models/blob/9549e0c301b448a749660ce716742ff928dc2778/figures/applied-energy-framework.png)

# Numerical experiments of the study
![numerical-experiments](https://github.com/jonathandumas/generative-models/blob/918ba080d82b04f541e2196a803165708f64fb73/figures/numerical-experiments-methodology.png)

# Dependencies

To implement the normalizing flows two libraries are required:
* https://github.com/AWehenkel/Normalizing-Flows
* Unconstrained Monotonic Neural Networks: https://github.com/AWehenkel/UMNN

If you make use of the Unconstrained Monotonic Neural Networks code, please cite the paper:

```
@inproceedings{wehenkel2019unconstrained,
  title={Unconstrained monotonic neural networks},
  author={Wehenkel, Antoine and Louppe, Gilles},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1543--1553},
  year={2019}
}
```

Concerning the forecast value assessment: the Python Gurobi library is used to implement the algorithms in Python 3.7, and [Gurobi](https://www.gurobi.com/) 9.0.2 to solve all the optimization problems.
