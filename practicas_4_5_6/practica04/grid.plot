set grid
set hidden3d
set dgrid3d
set contour
set surface
splot "grid.txt" using 1:2:3 with linespoints t "Accuracy"

pause -1 "Press ENTER to continue ... "
