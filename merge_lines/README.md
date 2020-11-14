# Example usage
merge_lines_bin.exe - in C://Users//yulia//Research//NPRRendering//DataSets//3d_sketches//3d_sketches// -model_name e8288b5f567569f19fb4103277a6b93_opt_quad_network_20_sketch_0.1.obj 

## Requirements
It requires downloading libigl https://libigl.github.io/.

## Compilation
Compliation can be done using cmake on CMakeLists.txt.
In Windows it was compiled for x64.

## Algoritm

* First, reads all the chains, and breaks them in subchain if the angle is smaller than 135 degrees.
* Remove all lines that are shorter than 10% of the smallest dimension D of the shape bounding box.
* Resamples all the lines with RDP algorithm with epsilon set to 2% of D.
* Goes over all lines iteratively, computes for each line the closest, tangentially alaigned line. If the distance is smaller than 5% of the largets lenght of two considered lines, the lines are merged.

