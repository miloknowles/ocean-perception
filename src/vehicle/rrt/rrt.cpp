#include <utility>
#include <glog/logging.h>

#include "rrt/rrt.hpp"
#include "core/random.hpp"

namespace bm {
namespace rrt {

static const int kDimension = 3;
static const double kMinObstacleDist = 0.5;
static const double kMaxLineDist = 5.0;


kdtree_t Tree::BuildKdTree() const
{
  kdtree_t kdtree(kDimension, points_, 10);
	kdtree.index->buildIndex();
  return kdtree;
}


size_t Tree::Nearby(const kdtree_t& kdtree,
                    const Vector3d& query_point,
                    double radius,
                    std::vector<index_t>& indices) const
{
  std::vector<IndexAndDist> out;

  // NOTE(milo): Not documented, but this seems to expect radius^2.
  const size_t N = kdtree.index->radiusSearch(&query_point[0], radius*radius, out, nf::SearchParams(10, 0.0f, true));

  indices.resize(out.size());
  for (size_t i = 0; i < out.size(); ++i) {
    indices.at(i) = out.at(i).first;
  }

  return N;
}


Tree::index_t Tree::Nearest(const kdtree_t& kdtree,
                            const Vector3d& query_point) const
{
  CHECK_GE(kdtree.m_data.size(), 1) << "Must have at least 1 point in the KDTree" << std::endl;

	nf::KNNResultSet<double> result_set(1);
  std::vector<size_t> indices(1);
  std::vector<double> dists_sqr(1);
	result_set.init(&indices[0], &dists_sqr[0]);
	kdtree.index->findNeighbors(result_set, &query_point[0], nanoflann::SearchParams(10));

  return indices.at(0);
}


size_t Tree::AddNode(const Node& node)
{
  points_.emplace_back(node.point);
  nodes_.emplace_back(node);
  return (points_.size() - 1);
}


void Tree::Rewire(size_t index, int new_parent, double new_cost_so_far)
{
  nodes_.at(index).parent = new_parent;
  nodes_.at(index).cost_so_far = new_cost_so_far;
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


static std::pair<size_t, double> ChooseParent(const Tree& tree,
                                              const CollisionChecker& collision_checker,
                                              const std::vector<Tree::index_t>& Z_near,
                                              const Tree::index_t& z_nearest,
                                              const Vector3d& x_new)
{
  const Node n_nearest = tree.GetNode(z_nearest);

  Tree::index_t z_min = z_nearest;
  double c_min = n_nearest.cost_so_far + (n_nearest.point - x_new).norm();

  for (const Tree::index_t& z_near : Z_near) {
    const Node n_near = tree.GetNode(z_near);
    const double c = n_near.cost_so_far + (n_near.point - x_new).norm();
    const bool is_collision_free = collision_checker(n_near.point, x_new);

    if (is_collision_free && c < c_min) {
      c_min = c;
      z_min = z_near;
    }
  }

  return std::pair<size_t, double>(z_min, c_min);
}


static void Rewire(Tree& tree,
                   const CollisionChecker& collision_checker,
                   const std::vector<Tree::index_t>& Z_near,
                   Tree::index_t z_min,
                   const Node& n_new,
                   Tree::index_t z_new)
{
  for (const Tree::index_t& z_near : Z_near) {
    // The parent of n_new can't be rewired through it.
    if (z_near == z_min) {
      continue;
    }

    const Node& n_near = tree.GetNode(z_near);
    const bool is_collision_free = collision_checker(n_new.point, n_near.point);

    if (!is_collision_free) {
      continue;
    }

    const double cost_if_rewired = n_new.cost_so_far + (n_new.point - n_near.point).norm();

    // If rewiring reduces the cost to reach n_near, update its parent.
    if (cost_if_rewired < n_near.cost_so_far) {
      tree.Rewire(z_near, z_new, cost_if_rewired);
    }
  }
}


void BuildTree(Tree& tree,
               const Vector3d& start,
               const Vector3d& goal,
               const PointSampler& sampler,
               const CollisionChecker& collision_checker,
               double search_radius,
               int maxiters)
{
  tree.AddNode(Node(start, -1, 0));

  for (int iter = 0; iter < maxiters; ++iter) {
    const kdtree_t kdtree = tree.BuildKdTree();

    // Get the node that is nearest to x_sample.
    const Vector3d x_sample = sampler();
    Tree::index_t z_nearest = tree.Nearest(kdtree, x_sample);

    // Try to find an x_new such that travelling from NN to x_new is collision-free.
    Vector3d x_new;
    const bool valid = ClipCollisionFree(
        tree.GetPoint(z_nearest), x_sample, kMinObstacleDist, kMaxLineDist, x_new);

    if (!valid) {
      continue;
    }

    // Get nodes that are nearby x_new.
    std::vector<Tree::index_t> Z_near;
    tree.Nearby(kdtree, x_new, search_radius, Z_near);

    const std::pair<size_t, double>& z_min = ChooseParent(tree, collision_checker, Z_near, z_nearest, x_new);

    // Insert the new node.
    const Node n_new(x_new, z_min.first, z_min.second);
    const size_t z_new = tree.AddNode(n_new);

    Rewire(tree, collision_checker, Z_near, z_min.first, n_new, z_new);
  }
}


}
}
