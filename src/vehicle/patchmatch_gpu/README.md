# GPU Implementation of the Patchmatch Algorithm

![Patchmatch GPU implementation example](/resources/patchmatch_example.png)

## Overview

This folder contains a fast CUDA implementation of the [Patchmatch](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/) stereo algorithm. I wanted to compare various non-learning stereo matching algorithms to compare their performance in low-visibility, low-texture underwater environments. The Patchmatch algorithm is interesting because you can optionally initialize it with "seed" pixels. If you have a source of sparse groundtruth, or sparse tracked features (which our VIO system provides), you can seed the algorithm at these image locations for better convergence. I didn't end up integrating Patchmatch with tracked features, but that was part of my motivation for implementing this algorithm.

## Parallelizing the Algorithm

In a CPU-based implementation of Patchmatch, you propagate information in a row-major scanning pattern (the way you would read a page of text), and then propagate in reverse. Iterating over all of the pixels in this way is very slow.

To parallelize the process, I split up the algorithm into row propagation and column propagation steps. For each iteration of Patchmatch we:
- propagate each row to the right
- propagate each column downwards
- propagate each row to the left
- propagate each column upwards

Furthermore, to utilize all of the GPU threads, we subdivide rows and columns into smaller chunks (e.g 8 strips per row).

This is an approximation of the original algorithm, since information is only propagated in one direction at a time. However, I found that in practice it converges to a similar result, since every pixel has the opportunity to propagate to a significant portion of the image over several iterations.
