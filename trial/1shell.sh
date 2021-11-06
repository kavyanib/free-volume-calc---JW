#!/bin/bash

python3 fourier_poly.py -n 1000

#again, the input -n is the number of coarse-grained unit per chain.
python3 3body.py -n 1000 -t1 450

#calculate the 3 body contribution term in the equation of state as stated in branch_II.pdf
python3 gvdw_2body.py -n 1000

#calculate the amount of free volume and the corresponding probability. The 3 body term was taken into account also. 
# the file pressure_vol.txt gives result in Table 2 of temp_dep.pdf.
# the file table_result.txt gives the probability F. 
