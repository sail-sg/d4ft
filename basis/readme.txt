''''''''''''''''''''''''''
Basis set example format
The format for the basis sets is the input format used by the Gaussian computer program.
An example is the carbon atom in the STO-3G basis set.

A STO-3G orbital is built from 3 Gaussian functions (GTOs):-
 

[STO-3G expansion]
 
There are 5 key components in the data:-
Standard basis: STO-3G (5D, 7F)

\psi = c_1 \phi_1 + c_2 \phi_2 + \c_3 \phi_3

where 


Basis set in the form of general basis input:
1   0
S   3   1.00
  exponent         s coefficient     p coefficient

.7161683735D+02 .1543289673D+00
.1304509632D+02 .5353281423D+00
.3530512160D+01 .4446345422D+00
SP 3 1.00
.2941249355D+01 -.9996722919D-01 .1559162750D+00
.6834830964D+00 .3995128261D+00 .6076837186D+00
.2222899159D+00 .7001154689D+00 .3919573931D+00
****
The SP line and the 3 following lines are items 3 and 4 repeated for the second basis function.
The explanation of these 5 components is:-

STO-3G is the name of the basis set. (5D, 7F) or (6D, 10F) or combinations thereof indicates 
5 d functions (spherical harmonics) or 6 d functions (cartesians) and similarly for the f functions.

Basis set in the form of general basis input: - is just a header.
Atom number (1 in this particular example, since carbon is the first atom in the molecule) or atom symbol, 
followed by a zero. Zero just ends a list, in this case of 1 element. Type of function (S, P, D, F etc or SP). 
SP indicates that same exponent for S and P will be used. This is followed by the number of individual Gaussians 
that make up the basis function (in this case 3) and a scale factor that is normally 1.00 as here. The scale 
factor can be altered to scale the STO that the GTOs are fitting.
Where there are pairs of numbers the first is the exponent (b1, b2 or b3) and the second is the coefficient 
(c1, c2 or c3) of the GTO in the basis function. Where there are three numbers for SP, the first is the exponent, 
the second is the coefficient in the S function and the third is the coefficient in the P function. Taking the 
same exponents for S and P speeds up the calculation, but we have to take different coefficents.

**** to end the data for one atom.
Items 3 and 4 are repeated for each basis function on a given atom. Items 2 - 5 are repeated for each atom.
This example is STO-3G for carbon. There is one S type function, and one SP set (i.e. one S and a set of 
three P: px, py and pz), making a total of 5 basis functions on this atom. You will see that this output 
also tells you how the d and f functions are being handled, usually by default for a named basis set.

''''''''''''''''''''''''''
 
Gaussian basis sets are identified by abbreviations such as N-MPG*. N is the number of Gaussian primitives used 
for each inner-shell orbital. The hyphen indicates a split-basis set where the valence orbitals are double zeta. 
The M indicates the number of primitives that form the large zeta function (for the inner valence region), and P 
indicates the number that form the small zeta function (for the outer valence region). G identifies the set a 
being Gaussian. The addition of an asterisk to this notation means that a single set of Gaussian 3d polarization 
functions (discussed elswhere) is included. A double asterisk means that a single set of Gaussian 2p functions is 
included for each hydrogen atom.

For example, 3G means each STO is represented by a linear combination of three primitive Gaussian functions. 
6-31G means each inner shell (1s orbital) STO is a linear combination of 6 primitives and each valence shell 
STO is split into an inner and outer part (double zeta) using 3 and 1 primitive Gaussians, respectively

For Json format:
  'elements':
    '1'
      'angular_momentum' （l): 
        0: s
        1: p 
        2: d



