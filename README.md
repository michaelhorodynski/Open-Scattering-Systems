# Open-Scattering-Systems

Open scattering systems can be used solve the 2D Helmholtz equation numerically
for scattering problems where waves can enter and exit from all sides. It also 
allows for the computation of the system's unitary and transposition symmetric 
scattering matrix. The repository includes a jupyter notebook which introduces 
all features by simple examples.

Installation and usage
----------------------

Open scattering systems relies on the open source finite-element software [NGSolve](https://ngsolve.org/),
which needs to be installed prior to using open scattering systems. Then simply 
import the functions into your notebook using

```python
from OpenScatteringSystems2d_funcs import *
```

Dependencies
------------

Open scattering systems runs on recent versions of NGSolve and additionaly
requires numpy, scipy and matplotlib.

Licence
-------

For full licensing details see LICENSE.md.

Open scattering systems can be referenced by citing the following paper

> Michael Horodynski, Tobias Reiter, Matthias Kühmayer, and Stefan Rotter
> "Tractor beams with optimal pulling force using structured waves",
> [arXiv](http://arxiv.org/XXX)

or by directly citing

> Michael Horodynski, Tobias Reiter, Matthias Kühmayer, and Stefan Rotter
> "Open scattering systems", https://github.com/michaelhorodynski/Open-Scattering-Systems

and the respective Bibtex entry

```latex
@misc{horodynski_open_2023,
  author = {Horodynski, Michael and Reiter, Tobias and Kühmayer, Mattthias and Rotter, Stefan},
  title = {Open Scattering Systems},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/michaelhorodynski/Open-Scattering-Systems}}}
}
```

Contact us
----------

The best person to contact for inquiries about open scattering systems
is [Michael Horodynski](mailto:michael.horodynski@gmail.com).

File listing
------------

```
README.md                           - Overview (this file)
LICENSE.md                          - License information
OpenScatteringSystems2d_funcs.py    - Functions for running open scattering systems
OpenScatteringSystems_example.ipynb - Example jupyter notebook showing different features
```
