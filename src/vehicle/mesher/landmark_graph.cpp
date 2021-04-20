#include <glog/logging.h>
#include <boost/graph/connected_components.hpp>

#include "mesher/landmark_graph.hpp"

namespace bm {
namespace mesher {


LandmarkGraph::LandmarkGraph() {}


void LandmarkGraph::AddLandmark(uid_t lmk_id)
{
  if (lmk_to_vtx_.count(lmk_id) == 0) {
    const uid_t vtx = boost::add_vertex(VertexName(lmk_id), g_);
    lmk_to_vtx_.emplace(lmk_id, vtx);
  }
}


// https://stackoverflow.com/questions/29931231/boost-how-to-remove-all-the-out-edges-for-a-vertex
void LandmarkGraph::RemoveLandmark(uid_t lmk_id)
{
  CHECK_GT(lmk_to_vtx_.count(lmk_id), 0)
      << "Trying to RemoveLandmark() for lmk_id not in g_" << std::endl;

  const uid_t vtx = lmk_to_vtx_.at(lmk_id);
  boost::clear_vertex(vtx, g_);   // Clears all edges, but not the vertex.
  boost::remove_vertex(vtx, g_);  // Then remove the vertex.
  lmk_to_vtx_.erase(lmk_id);

  // If the landmark is in the subgraph, remove there also.
  if (subgraph_lmk_to_vtx_.count(lmk_id) > 0) {
    const uid_t subgraph_vtx = subgraph_lmk_to_vtx_.at(lmk_id);
    boost::clear_vertex(subgraph_vtx, subgraph_);
    boost::remove_vertex(subgraph_vtx, subgraph_);
    subgraph_lmk_to_vtx_.erase(lmk_id);
  }
}


void LandmarkGraph::UpdateEdge(uid_t lmk1,
                               uid_t lmk2,
                               float increment,
                               float clamp_min,
                               float clamp_max,
                               float subgraph_min_weight)
{
  if (lmk_to_vtx_.count(lmk1) == 0) {
    AddLandmark(lmk1);
  }
  if (lmk_to_vtx_.count(lmk2) == 0) {
    AddLandmark(lmk2);
  }

  const uid_t v1 = lmk_to_vtx_.at(lmk1);
  const uid_t v2 = lmk_to_vtx_.at(lmk2);

  std::pair<Graph::edge_descriptor, bool> edge = boost::edge(v1, v2, g_);
  const bool edge_exists = edge.second;

  if (!edge_exists) {
    edge = boost::add_edge(v1, v2, EdgeWeight(0), g_);
  }

  // Otherwise add the increment to the edge weight.
  // https://stackoverflow.com/questions/24366642/how-do-i-change-the-edge-weight-in-a-graph-using-the-boost-graph-library/24372985
  const float weight = boost::get(edge_weight_t(), g_, edge.first);
  const float new_weight = std::min(clamp_max, std::max(clamp_min, weight + increment));
  boost::put(edge_weight_t(), g_, edge.first, new_weight);

  // Check if edge needs to be added/removed from the subgraph.
  if (new_weight >= subgraph_min_weight) {
    MaybeAddSubgraphEdge(lmk1, lmk2);
  } else {
    MaybeRemoveSubgraphEdge(lmk1, lmk2);
  }
}


void LandmarkGraph::MaybeAddSubgraphEdge(uid_t lmk1, uid_t lmk2)
{
  // Make sure vertices exist in subgraph.
  if (subgraph_lmk_to_vtx_.count(lmk1) == 0) {
    subgraph_lmk_to_vtx_.emplace(lmk1, boost::add_vertex(VertexName(lmk1), subgraph_));
  }
  if (subgraph_lmk_to_vtx_.count(lmk2) == 0) {
    subgraph_lmk_to_vtx_.emplace(lmk2, boost::add_vertex(VertexName(lmk2), subgraph_));
  }

  const uid_t v1 = subgraph_lmk_to_vtx_.at(lmk1);
  const uid_t v2 = subgraph_lmk_to_vtx_.at(lmk2);

  std::pair<Graph::edge_descriptor, bool> edge = boost::edge(v1, v2, subgraph_);
  const bool edge_exists = edge.second;

  if (!edge_exists) {
    edge = boost::add_edge(v1, v2, EdgeWeight(1), subgraph_);
  }
}


void LandmarkGraph::MaybeRemoveSubgraphEdge(uid_t lmk1, uid_t lmk2)
{
  // If either vertex doesn't exist, don't need to remove edge.
  if (subgraph_lmk_to_vtx_.count(lmk1) == 0) {
    return;
  }
  if (subgraph_lmk_to_vtx_.count(lmk2) == 0) {
    return;
  }

  const uid_t v1 = subgraph_lmk_to_vtx_.at(lmk1);
  const uid_t v2 = subgraph_lmk_to_vtx_.at(lmk2);

  std::pair<Graph::edge_descriptor, bool> edge = boost::edge(v1, v2, subgraph_);

  if (edge.second) {
    boost::remove_edge(edge.first, subgraph_);
  }
}


LmkClusters LandmarkGraph::GetClusters()
{
  std::vector<int> cluster_ids(boost::num_vertices(subgraph_));
  const int C = boost::connected_components(subgraph_, &cluster_ids[0]);

  LmkClusters out(C);

  for (uid_t v = 0; v < cluster_ids.size(); ++v) {
    const int cluster_id = cluster_ids[v];
    const uid_t lmk_id = boost::get(vertex_name_t(), subgraph_, v);
    out[cluster_id].insert(lmk_id);
  }

  return out;
}


}
}
