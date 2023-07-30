In my testing, a simple setup of two Conv2D layers (18 nodes 3x3 w/ relu activation, 16 nodes 3x3 w/ sigmoid activation) and one dense output layer (43 nodes) produced the best accuracy of ~98%.

Too many or too few nodes in any of the Conv2D layers reduced accuracy.
Additional Conv2D layers reduced accuracy.
A single Conv2D setup only performed slightly worse but almost as well as two, with accuracy decreasing with too many or too few nodes in that single layer.
A mix of relu and sigmoid activation functions for the 2 Conv2D layers gave the best accuracy over the same function for both.
Little difference between kernel matrix sizes of 2x2, 3x3, 4x4.

Adding Dense layers after flattening did not increase accuracy.

Any type of pooling seemed to decrease accuracy.
Adding dropout did not increase accuracy - with the relatively simple setup and large dataset, overfitting doesn't seem to be a problem.
Generally, many setups failed to breach 6% accuracy, but if they did, would end up at >60% accuracy.