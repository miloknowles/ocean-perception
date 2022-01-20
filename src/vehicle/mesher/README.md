# Stereo Object Meshing

This system estimates a mesh representation of objects in the current camera view using stereo feature tracking and delaunay triangulation.

## Algorithm Overview:
- We track keypoints from frame to frame using Lucas-Kanade optical flow
- We segment the image into "foreground" and "background" (empty space) regions using a gradient filter (not visualized here). Basically, highly textured areas are assumed to be foreground.
- We build a **feature graph** from keypoints in the scene. Features are connected by edges in the graph if (1) they are close together, and (2) the edge between them crosses mostly foreground pixels.
- We find connected components in the feature graph, and assume each is an object. This process can be brittle, and objects can merge or split.
- For each object (connected component of features), do Delaunay triangulation to build a mesh. The depth of each point is estimated from stereo matching.

The runtime is about 10-20 ms per frame, with feature tracking taking over 90% of that time.
