#include <glog/logging.h>
#include <boost/graph/connected_components.hpp>

#include "mesher/landmark_graph.hpp"

namespace bm {
namespace mesher {


LandmarkGraph::LandmarkGraph() {}


void LandmarkGraph::AddLandmark(uid_t lmk_id)
{
  if (lmk_to_vtx_.count(lmk_id) == 0) {
    const Graph::vertex_descriptor v = boost::add_vertex(VertexIndex(lmk_id), g_);
    lmk_to_vtx_.emplace(lmk_id, v);
  }
}


size_t LandmarkGraph::GraphSize() const
{
  return boost::num_vertices(g_);
}


size_t LandmarkGraph::SubgraphSize() const {
  return boost::num_vertices(subgraph_);
}


// https://stackoverflow.com/questions/29931231/boost-how-to-remove-all-the-out-edges-for-a-vertex
void LandmarkGraph::RemoveLandmark(uid_t lmk_id)
{
  CHECK_GT(lmk_to_vtx_.count(lmk_id), 0)
      << "Trying to RemoveLandmark() for lmk_id not in g_" << std::endl;

  LOG(INFO) << "removing lmk_id " << lmk_id << " from graph" << std::endl;
  const Graph::vertex_descriptor v = lmk_to_vtx_.at(lmk_id);
  boost::clear_vertex(v, g_);   // Clears all edges, but not the vertex.
  boost::remove_vertex(v, g_);  // Then remove the vertex.
  lmk_to_vtx_.erase(lmk_id);

  // If the landmark is in the subgraph, remove there also.
  if (subgraph_lmk_to_vtx_.count(lmk_id) > 0) {
    LOG(INFO) << "removing from subgraph" << std::endl;
    const Graph::vertex_descriptor subv = subgraph_lmk_to_vtx_.at(lmk_id);
    boost::clear_vertex(subv, subgraph_);
    boost::remove_vertex(subv, subgraph_);
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

  const Graph::vertex_descriptor v1 = lmk_to_vtx_.at(lmk1);
  const Graph::vertex_descriptor v2 = lmk_to_vtx_.at(lmk2);

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
    subgraph_lmk_to_vtx_.emplace(lmk1, boost::add_vertex(VertexIndex(lmk1), subgraph_));
  }
  if (subgraph_lmk_to_vtx_.count(lmk2) == 0) {
    subgraph_lmk_to_vtx_.emplace(lmk2, boost::add_vertex(VertexIndex(lmk2), subgraph_));
  }

  const Graph::vertex_descriptor v1 = subgraph_lmk_to_vtx_.at(lmk1);
  const Graph::vertex_descriptor v2 = subgraph_lmk_to_vtx_.at(lmk2);

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

  const Graph::vertex_descriptor v1 = subgraph_lmk_to_vtx_.at(lmk1);
  const Graph::vertex_descriptor v2 = subgraph_lmk_to_vtx_.at(lmk2);

  std::pair<Graph::edge_descriptor, bool> edge = boost::edge(v1, v2, subgraph_);

  if (edge.second) {
    boost::remove_edge(edge.first, subgraph_);
  }
}

// https://stackoverflow.com/questions/47909707/find-connected-components-using-boost-graph-library-with-the-vertex-and-edge-ty/47911251#47911251
LmkClusters LandmarkGraph::GetClusters(float subgraph_min_weight)
{
  Subgraph subgraph;
  std::unordered_map<uid_t, Subgraph::vertex_descriptor> lmk_id_to_vtx;

  const auto edge_bounds = boost::edges(g_);

  VertexIndexMap lmk_id_map;

  for (auto it = edge_bounds.first; it != edge_bounds.second; ++it) {
    Graph::vertex_descriptor v1 = boost::source(*it, g_);
    Graph::vertex_descriptor v2 = boost::target(*it, g_);
    const uid_t lmk1 = boost::get(vertex_index_t(), g_, v1);
    const uid_t lmk2 = boost::get(vertex_index_t(), g_, v2);

    std::pair<Graph::edge_descriptor, bool> edge = boost::edge(v1, v2, g_);
    const float weight = boost::get(edge_weight_t(), g_, edge.first);

    if (weight >= subgraph_min_weight) {
      if (lmk_id_to_vtx.count(lmk1) == 0) {
        lmk_id_to_vtx.emplace(lmk1, boost::add_vertex(subgraph));
      }
      if (lmk_id_to_vtx.count(lmk2) == 0) {
        lmk_id_to_vtx.emplace(lmk2, boost::add_vertex(subgraph));
      }

      const Subgraph::vertex_descriptor sv1 = lmk_id_to_vtx.at(lmk1);
      const Subgraph::vertex_descriptor sv2 = lmk_id_to_vtx.at(lmk2);
      boost::add_edge(sv1, sv2, subgraph);
    }
  }

  // std::map<Graph::vertex_descriptor, int> cluster_ids;
  // std::map<uid_t, int> cluster_ids;
  std::vector<int> cluster_ids(boost::num_vertices(subgraph));
  // std::vector<int> cluster_ids(boost::num_vertices(subgraph));

  // TODO(milo): incremental_components() is faster.
  // NOTE(milo): The compiler errors from Boost Graph are just about impossible to understand!
  // For some reason, boost::connected_components() will only work if the graph has a vertex index
  // property. If not, the compiler complains about a reference to void.
  // auto property_map = boost::make_assoc_property_map(cluster_ids);
  // auto comp_map = boost::make_iterator_property_map(cluster_ids.begin(), boost::get(boost::vertex_index, subgraph_));
  const int C = boost::connected_components(subgraph, &cluster_ids[0]);

  LmkClusters out(0);

  // for (auto it = cluster_ids.begin(); it != cluster_ids.end(); ++it) {
  //   const int cluster_id = it->second;
  //   const uid_t lmk_id = boost::get(vertex_index_t(), subgraph_, it->first);
  //   out.at(cluster_id).insert(lmk_id);
  // }

  return out;
}


LmkSet LandmarkGraph::GetLandmarkIds() const
{
  LmkSet out;
  for (auto it = lmk_to_vtx_.begin(); it != lmk_to_vtx_.end(); ++it) {
    out.insert(it->first);
  }

  return out;
}


}
}
