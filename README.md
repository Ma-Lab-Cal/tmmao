# tmmao
---
Physics-agnostic inverse design, based on transfer matrix method

## Description
---
This module (soon to be package!) is an implementation of adjoint-gradient-based inverse design using the transfer matrix method. If 
If you find any part of the `tmmao` package helpful and you happen to use it a publication of your own, please cite our work introducing the algorithm:

> N. Morrison, S. Pan, and E. Ma, "Physics-Agnostic Inverse Design Using Transfer Matrices", under review, 2023.

Thanks!

### `angostic_linear_adjoint.py`
The central piece of code is `angostic_linear_adjoint.py`. You hand it your transfer matrices, cost funciton, and their partial derivatives, and it will hand you the total derivative of your cost function with respect to the optimization parameters - exactly the quantity a gradient-descent or quasi-Newton optimizer needs to figure out how to update the optimization paramters to minimize your cost function. `angostic_linear_adjoint.py`'s algorithm is, as the name implies, agnostic. It does not care what kind of physics you're working with - optics, acoustics, quantum mechanics, electronics, etc. It doesn't even care why you need the gradient. Maybe you're not doing an optimization and you just like Jacobians. `angostic_linear_adjoint.py` doesn't care. It just finds the gradient. That's what it does. `angostic_linear_adjoint.py` has a fully documented API, and you are encouraged to integrate it into your own optimziation routines. 

### `agnostic_invDes.py`
If, however, you happen to be a fan of BFGS quasi-Newton optimization, we've got some extra stuff for you. Also provided for you convenience (and because we need them to make the interactive examples run) is `agnostic_invDes.py`, a file/class that interfaces `angostic_linear_adjoint.py` with `scipy.minimize`'s BFGS (if you want to do unconstrained optimization) or L-BFGS-B (if you want to do constrained optimization) routine. It additionally enables you to do dynamic scheduling and extenal parameter manipulation. This means you can decide, at every step of the optimization, to change your cost function weights, add more constraints, eliminate optimziation parameters that have gotten too small, etc. - something not natively supported by `scipy.minimize`. As with `angostic_linear_adjoint.py`, `agnostic_invDes.py` is physics-agnostic: you provide the transfer matrices, cost function, partial derivatives, and initial optimization parameters, and it hands you a cost function which is as small as possible. It is slightly more constraining than `angostic_linear_adjoint.py`, however, in that it is built around the BFGS algorithm in particular. `agnostic_invDes.py` also has a fully documented API, and you are encouraged to integrate it into your inverse design routines. 

### `agnostic_director.py`
If you want to work with some even higher-level code, we've provided `agnostic_director.py`, a file/class designed to serve as an interface between `agnostic_invDes.py` and your (or one of our) physics packages - the modules that contain the physics-specific transfer matrix calculations. `agnostic_invDes.py` requires the matrices/partial derivatives to be arranged in particular formats, so it knows which corresponds to wich layer/variable. These formats can be found in `agnostic_invDes.py`'s docs, but `agnostic_director.py` will handle that formating for you. `agnostic_director.py` currently supports two types of inverse design problem:

1. The "independent" problem: the system being optimized is a series of layers of particular thickness and composition. Currently, only binary layer stacks - with the optimization region composed of at most two different materials - are supported in the stable build. The thickness $l_j$ of each layer $j$ may be used as optimization parameters. The "ratio" $r_j$ of each layer may also be used. If $r_j=0$, then the layer is 100% mateial 1; if $r_j=1$, then the layer is 100% material 2. Any other value will mean the properties of layer $j$ are a weighted average of the properties of material 1 and material 2. This allows for a "continuous phase" of optimization, with a variety of options to gradually (or abruptly) discretize the layers to either $r_j=0$ or $r_j=1$ by the end of optimization if necesary. You can choose whether to use $r_j$, $l_j$, or both during optimization, and can even choose to alternate which set of parameters are being optimized. You can also add as many non-optimizable substrates or superstrates to your stack as you wish, of any material (e.g. not just material 1 or material 2)
2. The "fixedPoint" problem: the system is a sequence of "fixed points". The possible optimization parameters are the spacing between each pair of neighboring fixed points, and the "height" of each fixed point, whatever "height" means in your optimization. For example, in the provied quantum example, the fixed points are electrodes held at particular electric potentials. A mesh of transfer matrices is then interpolated between each fixed point.

`agnostic_director.py` will allow you to load your own (or one of the three provided) physics packages, and will make sure that the transfer matrices and partial derivatives get where they need to go. This module has gotten a bit large, and full documentation is currently in progress.

### `agnostic_analysis.py`
So, you've inverse designed some fancy device using agnostic_director.py, and you want to see some pretty picture of how it works. `agnostic_analysis.py` has you covered. `agnostic_analysis.py` is inherited by `agnostic_director.py`, so you can use its methods through the same object that you use to do your optimizing. This module allows you to build a variety of the most common analysis plots with a high degree of customization (and more on the way!), using both fresh and old data. The current menu of options includes:
1. Cost function plots: Plot your training curve to see what the computer was doing while it took its sweet time to converge.
2. Structure plots: Plot a visualization of the structure of your device, be it the final design, an intermediate step during the optimization, or an entirely different, custom device you just came up with.
3. Field plots: Plot a colormesh or contour plot of what's happening inside your device during operation: where energy is being absorbed, the magnitude of the current electric field, the local acoustic pressure, etc.
4. Indicator plots: Plot anything that describes how well your device performs. From the transmission of a filter to the echo reduction of an antisonar coating, your only limit is what you can code in your physics package.

### `materials_library.py`
Can't do an optimization without material models. We've provided a small selection of materials for acoustics, optics, and quantum optimizations. From silicon to germanium and polyurethane to steel, there should be enough to get you started. All the materials functions are formatted for use with `agnostic_director.py`; to add your own, simply follow the same style. More instructions on how to do so will be provided in the upcoming `materials_library.py` documentation.

### `misc_utils.py`
Miscellaneous utilities used by the other modules, including pickling and plotting. Will eventually get documented, but not necessary to understand the rest of the code.

### `optics_physics_package.py` / `acoustics_physics_package.py` / `quantum_physics_package.py`
Some example physics packages implemented by us, configured for use with `agnostic_director.py`. These house the explicit formulas for the transfer matrices, as well as the cost function. If you wish to add your own (which we encourage you to do!) to use with agnostic_director.py, simply follow the same format. Full documentation for each is forthcomming; in the meantime, look at the provided example notebooks.

### `optics_example.ipynb` / `acoustics_example.ipynb` / `quantum_example.ipynb`
Interactive notebooks showcasing how to use `agnostic_director.py`, applied to some inverse design problems. We've done our best to explain what every option we use means. More examples - including how to use just `agnostic_invDes.py` and `angostic_linear_adjoint.py` with your own optimization routines/inverse design directors - are forthcomming. There are two copies of these notebooks. The copies in the main directory are Google Colab notebooks; click the link at the top of the page to play with them online, no install required! The copies in the `tmmao` subfolder are Jupyter notebooks intended to be run locally.

## Installation
---
Simply clone the repository! To use this module on Google Colab, simply type
```
!git clone https://github.com/Ma-Lab-Cal/tmmao.git
```
This will create a folder called `tmmao/` on the Google machine you're using, housing everything in the repository. The modules can then be found in the `tmmao/tmmao/` folder. A quick (though a bit ham-fisted) way to gain access to these modules is to simpyl change the working directory on the Google machine to this folder via 
```Python
import os
os.chdir('tmmao/tmmao/')
```
Then import as demonstrated in the examples!

To use this module locally, clone the repository and make sure to import the requisite modules from the `tmmao/tmmao/` folder.

## Dependencies
---
Actually, nothing too exotic! The following packages are required for `agnostic_director.py`'s full functionality. The other modules may not require all of them
1. `numpy`
2. `scipy`
3. `matplotlib`
4. `cmath`

## Testing/Demos
---
To test this module (and see how to use its API), try out `optics_example.ipynb` / `acoustics_example.ipynb` / `quantum_example.ipynb`. These are very heavily commented interactive notebooks that walk you through each step in using `agnostic_director.py` to do some inverse designs. You can run them locally using the notebooks in the `tmmao` subfolder, but the easiest way to play with them is clicking the Open In Colab link at the top of each of the example files in the main folder, next to this document. No installation required!

