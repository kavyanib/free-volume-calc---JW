#!/bin/bash

for t in 490 
do 
    for n_bead in 600
    do

        python fourier_poly.py -n $n_bead -t1 $t

        #again, the input -n is the number of coarse-grained unit per chain.
        python 3body.py -n $n_bead -t1 $t

        #calculate the 3 body contribution term in the equation of state as stated in branch_II.pdf
        python gvdw_2body.py -n $n_bead -t1 $t

        #calculate the amount of free volume and the corresponding probability. The 3 body term was taken into account also. 
        # the file pressure_vol.txt gives result in Table 2 of temp_dep.pdf.
        # the file table_result.txt gives the probability F. 
    done
done
