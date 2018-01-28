# Speedometer-Detection
A computer Vision project that involves an initial training of minimmum(0) and maximmum point(max speed).


In this case the training is done through circles detected on the speedometer, color chosen was green. 

From here We dtect the speedometers lines, choosing the longest line, from a centralized point. THis allows us to remove extraneous lines that are not going through the center of a speedometer, intersecting it like a radius. 

We then compute intersection along a normalzed circle( calibrated points are often a few pixels off where a circle would be located. Relative to the Radius)

From here we compute the speed based on a percentage of the arc length traversed. 

