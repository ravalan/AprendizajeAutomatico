#!/bin/bash


#data_file="grid-1.out"
data_file="grid-2.out"
cat ${data_file} | awk '{ print $1, $2, $5 }' | sed 's/%//g' | sort >grid.txt


cat >grid.plot <<-EOF
set grid
set hidden3d
set dgrid3d
set contour
set surface
splot "grid.txt" using 1:2:3 with linespoints t "Accuracy"

pause -1 "Press ENTER to continue ... "
EOF


gnuplot grid.plot

