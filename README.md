# PyZMAT

Author: Yixuan Huang

ASE wrapper for workflows in internal coordinates/Z-Matrix.

Built as a personal tool as a part of my research in ML-CSP under the HAIEFF framework [citations]. 

## Main capabilities

* Building molecules in Cartesian coordinates (ASE Atoms object) from Z-matrices.
* Calculating internal coordinates and forming Z-matrices from Cartesian coordinates.
* Analytical first and second derivatives of Cartesian coordinates with respect to internal coordinates.
* Analytical transformation of force vectors and Hessian matrices from Cartesian to internal coordinates based on the above derivatives.
* Executing a selection of ASE minimisation routines using ML force fields.

## Installation

(placeholder)

PyZMAT should work independently if you only wish to use its coordinate transform capabilities. Please install this package in a working MACE or AIMNet2 conda environment if you wish to use ML-FFs. 

## Acknowledgements

Benjamin Tan for calculating the analytical first derivatives [citation].
