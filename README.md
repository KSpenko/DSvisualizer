# DSvisualizer
Python script for visualizing chaos in dynamical systems.

## Description
While analyzing different dynamical systems, it is always useful, practical, and often necessary to search for and observe chaotic dynamics.
The main visualization tool that helps us grasp chaotic behavior in a (usually) vast phase space is the so-called Poincare surface of section. 
It is then very practical to study how a Poincare map changes by varying some global parameters and to be able to sample initial conditions from different parts off the Poincare map.

Precisely for this purpose, I created this tool as a very convenient tool for quickly demonstrating chaotic motion. 
I composed a general class/solver which can be applied to many different physical problems in different ways.
The main part of the tool draws Poincare maps, while the second part samples individual trajectories, for which it then calculates a FFT (Fast-Fourier transform), which can be particularly useful for estimating periodic motion, 
and also the maximal Lyapunov exponent, which is THE objective measure to determine chaos.

## Getting Started

### Dependencies

* Python3
* numpy
* matplotlib
* scipy

### Executing program

* compose a script that describes a physics problem (i.e. the example files)
* call ```DSvisualizer()```

## Help

By mindfull of the computational burden of many tracks,
because solving NDE and visualizing the Poincare scatter plot can take quite some time.

## Authors

ex. Krištof Špenko [@KSpenko](https://twitter.com/kspenko)

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [CardicaEA_SiroosNazari](https://www.researchgate.net/publication/274122953_Modified_Modeling_of_the_Heart_by_Applying_Nonlinear_Oscillators_and_Designing_Proper_Control_Signal)
* [ColHe_Wolfram](https://demonstrations.wolfram.com/CollinearClassicalHeliumAtom/)
