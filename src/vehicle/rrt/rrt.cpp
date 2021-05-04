#include <glog/logging.h>

#include "rrt/rrt.hpp"
#include "core/random.hpp"

namespace bm {
namespace rrt {

static const int kDimension = 3;


kdtree_t Tree::BuildKdTree() const
{
  kdtree_t kdtree(kDimension, points_, 10);
	kdtree.index->buildIndex();
  return kdtree;
}


int Tree::Nearby(const kdtree_t& kdtree,
                 const Vector3d& query_point,
                 double radius,
                 std::vector<IndexAndDist>& index_and_dist)
{
  index_and_dist.clear();

  // NOTE(milo): Not documented, but this seems to expect radius^2.
  return kdtree.index->radiusSearch(&query_point[0], radius*radius, index_and_dist, nf::SearchParams(10, 0.0f, true));
}


void Tree::Nearest(const kdtree_t& kdtree, const Vector3d& query_point, IndexAndDist& index_and_dist)
{
  CHECK_GE(kdtree.m_data.size(), 1) << "Must have at least 1 point in the KDTree" << std::endl;

	nf::KNNResultSet<double> result_set(1);
  std::vector<size_t> indices(1);
  std::vector<double> dists_sqr(1);
	result_set.init(&indices[0], &dists_sqr[0]);
	kdtree.index->findNeighbors(result_set, &query_point[0], nanoflann::SearchParams(10));

  index_and_dist.first = indices[0];
  index_and_dist.second = std::sqrt(dists_sqr[0]);
}


void Tree::AddNode(const Node& node)
{
  points_.emplace_back(node.point);
  nodes_.emplace_back(node);
}


Vector3d SampleBoxPoint(const Vector3d& pmin, const Vector3d& pmax)
{
  const double x = RandomUniformd(pmin.x(), pmax.x());
  const double y = RandomUniformd(pmin.y(), pmax.y());
  const double z = RandomUniformd(pmin.z(), pmax.z());
  return Vector3d(x, y, z);
}


bool ClipCollisionFree(const Vector3d& x0,
                       const Vector3d& x1,
                       double min_obstacle_dist,
                       double max_line_dist,
                       Vector3d& x_new)
{
  const Vector3d v = x1 - x0;
  const double t = std::fmin(max_line_dist, v.norm());
  x_new = x0 + t*v.normalized();

  return true;
}


// void BuildTree(Tree& tree, const Vector3d& start, const Vector3d& goal, const PointSampler& sampler, int maxiters)
// {
//   tree.AddNode(Node(start, -1, 0));

//   for (int iter = 0; iter < maxiters; ++iter) {
//     const Vector3d x_cand = sampler();

//     IndexAndDist index_and_dist;
//     tree.Nearby(x_cand, search_radius, index_and_dist);
//   }
// }


}
}
