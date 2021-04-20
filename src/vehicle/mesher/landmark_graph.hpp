#pragma once

#include <unordered_map>
#include <vector>

#include <boost/graph/adjacency_list.hpp>

#include "core/uid.hpp"

namespace bm {
namespace mesher {

using namespace core;

typedef boost::edge_weight_t edge_weight_t;
typedef boost::vertex_index_t vertex_index_t;
typedef boost::vertex_name_t vertex_name_t;
typedef boost::property<vertex_index_t, uid_t> VertexIndex;
typedef boost::property<vertex_name_t, uid_t> VertexName;
typedef boost::property<edge_weight_t, float> EdgeWeight;

// https://cs.brown.edu/~jwicks/boost/libs/graph/doc/adjacency_list.html
// Calling boost::remove_vertex() invalidates vertex ids! Have to use setS or listS to keep them stable.
typedef boost::adjacency_list<boost::setS,
                              boost::setS,
                              boost::undirectedS,
                              VertexIndex,
                              EdgeWeight> Graph;

typedef boost::adjacency_list<boost::vecS,
                              boost::vecS,
                              boost::undirectedS,
                              VertexIndex> Subgraph;

typedef boost::property_map<Graph, vertex_index_t>::type VertexIndexMap;
typedef boost::property_map<Graph, edge_weight_t>::type EdgeWeightMap;

typedef std::unordered_set<uid_t> LmkSet;
typedef std::vector<LmkSet> LmkClusters;


class LandmarkGraph final {
 public:
  LandmarkGraph();

  // Add a new landmark to the graph.
  void AddLandmark(uid_t lmk_id);

  // Delete a landmark from the graph, along with any edges that connect to it.
  void RemoveLandmark(uid_t lmk_id);

  // Update the edge weight between lmk1 and lmk2 with some scalar amount of evidence for existence.
  // If the edge does not exist, it is created with edge weight equal to increment.
  void UpdateEdge(uid_t lmk1,
                  uid_t lmk2,
                  float increment,
                  float clamp_min,
                  float clamp_max,
                  float subgraph_min_weight);

  // Finds connected components in the landmark graph and returns them.
  LmkClusters GetClusters(float subgraph_min_weight);

  // Returns a set of ids for all the landmarks current in the graph.
  LmkSet GetLandmarkIds() const;

  size_t GraphSize() const;
  size_t SubgraphSize() const;

 private:
  void MaybeAddSubgraphEdge(uid_t lmk1, uid_t lmk2);
  void MaybeRemoveSubgraphEdge(uid_t lmk1, uid_t lmk2);

 private:
  std::unordered_map<uid_t, Graph::vertex_descriptor> lmk_to_vtx_;
  std::unordered_map<uid_t, Graph::vertex_descriptor> subgraph_lmk_to_vtx_;

  Graph g_, subgraph_;
};


}
}
