#include <gtest/gtest.h>
#include <iostream>

#include "rrt/rrt.hpp"
#include "rrt/nanoflann_adaptor.hpp"

using namespace bm;
using namespace core;
using namespace rrt;


static const int kDimension = 3;


TEST(RRTTest, KDTreeStatic)
{
  VecVector3d points;

  for (int i = 0; i < 100; ++i) {
    points.emplace_back(i, i, i);
  }

	typedef KDTreeVectorOfVectorsAdaptor<VecVector3d, double> kdtree_t;

	kdtree_t kdtree(kDimension, points, 10);
	kdtree.index->buildIndex();

	// do a knn search
	const size_t num_results = 3;
	std::vector<size_t> ret_indexes(num_results);
	std::vector<double> out_dists_sqr(num_results);

	nanoflann::KNNResultSet<double> resultSet(num_results);

  const Vector3d query_pt(25, 25, 25);

	resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
	kdtree.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

	std::cout << "knnSearch (nn="<<num_results<<"): \n";
	for (size_t i = 0; i < num_results; i++)
		std::cout << "ret_index["<<i<<"]=" << ret_indexes[i] << " out_dist_sqr=" << out_dists_sqr[i] << std::endl;
}


TEST(TreeTest, Nearest)
{
  Tree tree;
  const Vector3d p0 = Vector3d(1, 2, 3);
  tree.AddNode(Node(p0, -1, 100));

  const kdtree_t kd = tree.BuildKdTree();

  Tree::index_t out = tree.Nearest(kd, Vector3d(0, 0, 0));

  ASSERT_EQ(0ul, out);

  const Vector3d p1(-2, 4, 8);
  tree.AddNode(Node(p1, 0, 101));

  const kdtree_t kd2 = tree.BuildKdTree();

  out = tree.Nearest(kd2, Vector3d(-2, 4, 7));
  ASSERT_EQ(1ul, out);
}


TEST(TreeTest, Nearby)
{
  Tree tree;

  const Vector3d p0 = Vector3d(1, 2, 3);
  const Vector3d p1 = Vector3d(-1, -2, -3);
  const Vector3d p2 = Vector3d(0, 0, 0);

  tree.AddNode(Node(p0, -1, 100));
  tree.AddNode(Node(p1, -1, 100));
  tree.AddNode(Node(p2, -1, 100));

  const kdtree_t kd = tree.BuildKdTree();

  // Search radius is small, no points nearby.
  std::vector<Tree::index_t> out;
  tree.Nearby(kd, Vector3d(10, 10, 10), 0.1, out);
  EXPECT_EQ(0ul, out.size());

  // Query at origin, 1 point nearby.
  tree.Nearby(kd, Vector3d(0, 0, 0), 0.1, out);
  EXPECT_EQ(1ul, out.size());
  EXPECT_EQ(2ul, out[0]);

  // Large search radius, all points nearby.
  const double r = Vector3d(1, 2, 3).norm();
  tree.Nearby(kd, Vector3d(0, 0, 0), r + 0.01, out);
  EXPECT_EQ(3ul, out.size());
}
