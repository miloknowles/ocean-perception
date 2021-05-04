#pragma once

#include <vector>
#include <nanoflann.hpp>

#include "core/eigen_types.hpp"
#include "rrt/nanoflann_adaptor.hpp"

namespace bm {
namespace rrt {

namespace nf = nanoflann;
using namespace core;

typedef std::vector<core::Vector3d> VecVector3d;
typedef KDTreeVectorOfVectorsAdaptor<VecVector3d, double> kdtree_t;
typedef nf::KDTreeSingleIndexAdaptorParams kdtree_params_t;
typedef std::pair<size_t, double> IndexAndDist;
typedef std::function<Vector3d()> PointSampler;
typedef std::function<bool(const Vector3d&, const Vector3d&)> CollisionChecker;

struct Node
{
	Node() = default;

  explicit Node(const Vector3d& point, int parent, double cost_so_far)
			: point(point), parent(parent), cost_so_far(cost_so_far) {}

  Vector3d point;   // Location in 3D space.
	int parent;       // Index of the parent node.
  double cost_so_far;
};


class Tree {
 public:
	typedef size_t index_t;

	// Returns nearby neighbors within a spherical search radius. Note that returned
	// neighbors are sorted by *increasing* distance, so the nearest neighbor is first.
	size_t Nearby(const kdtree_t& kdtree,
								const Vector3d& query_point,
								double radius,
								std::vector<index_t>& indices) const;

	// Find the nearest node to query_point.
	index_t Nearest(const kdtree_t& kdtree,
							 		const Vector3d& query_point) const;

	// Add a node to the tree and return its index.
	index_t AddNode(const Node& node);

	Node GetNode(index_t index) const { return nodes_.at(index); }
	void Rewire(index_t index, int new_parent, double new_cost_so_far);

	Vector3d GetPoint(index_t index) const { return points_.at(index); }

	// Rebuild a kd-tree data structure using the current points_. Note that this has to be
	// recomputed every time we add or remove a node.
	kdtree_t BuildKdTree() const;

 private:
	VecVector3d points_;
  std::vector<Node> nodes_;
};


// Sample a point uniformly from within a box that has corners at pmin and pmax.
Vector3d SampleBoxPoint(const Vector3d& pmin, const Vector3d& pmax);


// Limit the extension of p0 -> p1 to length max_line_dist.
bool ClipCollisionFree(const Vector3d& x0,
											 const Vector3d& x1,
											 double min_obstacle_dist,
											 double max_line_dist,
											 Vector3d& x_new);


// Runs the RRT* algorithm.
void BuildTree(Tree& tree,
							 const Vector3d& start,
							 const Vector3d& goal,
							 const PointSampler& sampler,
							 const CollisionChecker& collision_checker,
							 double search_radius,
							 int maxiters);


}
}
