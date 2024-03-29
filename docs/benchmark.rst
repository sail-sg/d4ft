Benchmark
=========


Speed 
*****

**Running time comparison on carbon Fullerene molecules**

+---------+----------+--------+----------+---------+----------+
| Package | C20      | C60    | C80      | C100    | C180     |
+=========+==========+========+==========+=========+==========+
| *D4FT*  | 11.46    | 23.14  | 62.98    | 195.13  | 1103.19  |
+---------+----------+--------+----------+---------+----------+
| *PySCF* | 20.75    | 186.67 | >3228.20 | 672.98  | 1925.25  |
+---------+----------+--------+----------+---------+----------+
| *GPAW*  | >2783.08 | \-\-   | \-\-     | \-\-    | \-\-     |
+---------+----------+--------+----------+---------+----------+
| *Psi4*  | >46.12   | 510.35 | >2144.49 | 2555.16 | 14321.64 |
+---------+----------+--------+----------+---------+----------+


Accuracy
********

**Ground state energy calculation on small molecules** (LDA, 6-31g, Ha)


+-----------+---------+---------------------------+--------------------------+-----------------+------------+---------------+---+---+---+
| Molecule  | Method  | Nuclear Repulsion Energy  | Kinetic+ Externel Energy | Hartree Energy  | XC Energy  | Total Energy  |   |   |   |
+===========+=========+===========================+==========================+=================+============+===============+===+===+===+
| Hydrogen  | PySCF   | 0.71375                   | -2.48017                 | 1.28032         | -0.55256   | -1.03864      |   |   |   |
|           | D4FT    | 0.71375                   | -2.48343                 | 1.28357         | -0.55371   | -1.03982      |   |   |   |
|           | JaxSCF  | 0.71375                   | -2.48448                 | 1.28527         | -0.5539    | -1.03919      |   |   |   |
+-----------+---------+---------------------------+--------------------------+-----------------+------------+---------------+---+---+---+
| Water     | PySCF   | 13.47203                  | -79.79071                | 32.68441        | -5.86273   | -39.497       |   |   |   |
|           | D4FT    | 13.47203                  | -79.77446                | 32.68037        | -5.85949   | -39.48155     |   |   |   |
|           | JaxSCF  | 13.47203                  | -79.80889                | 32.71502        | -5.86561   | -39.48745     |   |   |   |
+-----------+---------+---------------------------+--------------------------+-----------------+------------+---------------+---+---+---+
| Oxygen    | PySCF   | 28.04748                  | -261.19971               | 99.92152        | -14.79109  | -148.0218     |   |   |   |
|           | D4FT    | 28.04748                  | -261.13046               | 99.82705        | -14.77807  | -148.03399    |   |   |   |
|           | JaxSCF  | 28.04748                  | -261.16314               | 99.87551        | -14.7829   | -148.02304    |   |   |   |
+-----------+---------+---------------------------+--------------------------+-----------------+------------+---------------+---+---+---+
| Ethanol   | PySCF   | 82.01074                  | -371.73603               | 156.36891       | -18.65741  | -152.01379    |   |   |   |
|           | D4FT    | 82.01074                  | -371.73431               | 156.34982       | -18.65517  | -152.02893    |   |   |   |
|           | JaxSCF  | 82.01074                  | -371.64258               | 156.27433       | -18.65711  | -152.01460    |   |   |   |
+-----------+---------+---------------------------+--------------------------+-----------------+------------+---------------+---+---+---+
| Benzene   | PySCF   | 203.22654                 | -713.15807               | 312.42026       | -29.84784  | -227.35910    |   |   |   |
|           | D4FT    | 203.22654                 | -712.71081               | 311.94665       | -29.81097  | -227.34860    |   |   |   |
|           | JaxSCF  | 203.22654                 | -712.91053               | 312.15690       | -29.84042  | -227.36755    |   |   |   |
+-----------+---------+---------------------------+--------------------------+-----------------+------------+---------------+---+---+---+
