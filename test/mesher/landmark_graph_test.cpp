#include <gtest/gtest.h>

#include "core/uid.hpp"
#include "mesher/landmark_graph.hpp"

using namespace bm;
using namespace core;
using namespace mesher;


TEST(LandmarkGraph, All)
{
  LandmarkGraph g;

  // Make sure adding and removing a landmark works.
  g.AddLandmark(123);
  EXPECT_EQ(1ul, g.GraphSize());
  EXPECT_EQ(0ul, g.SubgraphSize());
  g.RemoveLandmark(123);

  EXPECT_EQ(0ul, g.GraphSize());
  EXPECT_EQ(0ul, g.SubgraphSize());

  // Add two landmarks and connect with an edge.
  g.AddLandmark(123);
  g.AddLandmark(456);
  // Weight is 1.0 and min weight is 2.0, so not added to subgraph.
  g.UpdateEdge(123, 456, 1.0f, -5.0f, 5.0f, 2.0);
  EXPECT_EQ(2ul, g.GraphSize());
  EXPECT_EQ(0ul, g.SubgraphSize());

  // Edge v1/v2 should be symmetric.
  g.UpdateEdge(456, 123, 1.0, -5.0, 5.0, 2.0);  // Now added to subgraph.
  EXPECT_EQ(2ul, g.GraphSize());
  EXPECT_EQ(2ul, g.SubgraphSize());

  LmkClusters clusters = g.GetClusters();
  EXPECT_EQ(1ul, clusters.size());
  EXPECT_EQ(1ul, clusters.at(0).count(123));
  EXPECT_EQ(1ul, clusters.at(0).count(456));

  // Make sure edges are removed from the graph when their weight goes below threshold.
  g.UpdateEdge(123, 456, -6.0, -2.0, 2.0, 2.0);
  EXPECT_EQ(2ul, g.GraphSize());
  EXPECT_EQ(2ul, g.SubgraphSize()); // Vertices still exist, but edge is gone.

  LmkClusters clusters2 = g.GetClusters();
  EXPECT_EQ(2ul, clusters2.size());   // Each vertex in own cluster.

  // Now if we update the edge again with +4, it should be added to subgraph, since it was clamped at -2.
  g.UpdateEdge(123, 456, 4.0, -2.0, 2.0, 2.0);

  LmkClusters clusters3 = g.GetClusters();
  EXPECT_EQ(1ul, clusters3.size());
  EXPECT_EQ(1ul, clusters3.at(0).count(123));
  EXPECT_EQ(1ul, clusters3.at(0).count(456));
}
