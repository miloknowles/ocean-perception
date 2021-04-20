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


// https://stackoverflow.com/questions/29931231/boost-how-to-remove-all-the-out-edges-for-a-vertex
void LandmarkGraph::RemoveLandmark(uid_t lmk_id)
{
  CHECK_GT(lmk_to_vtx_.count(lmk_id), 0)
      << "Trying to RemoveLandmark() for lmk_id not in g_" << std::endl;

  const Graph::vertex_descriptor v = lmk_to_vtx_.at(lmk_id);
  boost::clear_vertex(v, g_);   // Clears all edges, but not the vertex.
  boost::remove_vertex(v, g_);  // Then remove the vertex.
  lmk_to_vtx_.erase(lmk_id);
}


void LandmarkGraph::UpdateEdge(uid_t lmk1,
                               uid_t lmk2,
                               float increment,
                               float clamp_min,
                               float clamp_max)
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
}

// https://stackoverflow.com/questions/47909707/find-connected-components-using-boost-graph-library-with-the-vertex-and-edge-ty/47911251#47911251
LmkClusters LandmarkGraph::GetClusters(float subgraph_min_weight)
{
  Subgraph subgraph;
  std::unordered_map<uid_t, Subgraph::vertex_descriptor> lmk_id_to_vtx;
  std::unordered_map<Subgraph::vertex_descriptor, uid_t> vtx_to_lmk_id;

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
        const Subgraph::vertex_descriptor v = boost::add_vertex(subgraph);
        lmk_id_to_vtx.emplace(lmk1, v);
        vtx_to_lmk_id.emplace(v, lmk1);
      }
      if (lmk_id_to_vtx.count(lmk2) == 0) {
        const Subgraph::vertex_descriptor v = boost::add_vertex(subgraph);
        lmk_id_to_vtx.emplace(lmk2, v);
        vtx_to_lmk_id.emplace(v, lmk2);
      }

      const Subgraph::vertex_descriptor sv1 = lmk_id_to_vtx.at(lmk1);
      const Subgraph::vertex_descriptor sv2 = lmk_id_to_vtx.at(lmk2);
      boost::add_edge(sv1, sv2, subgraph);
    }
  }

  std::vector<int> cluster_ids(boost::num_vertices(subgraph));

  // TODO(milo): incremental_components() is faster.
  // NOTE(milo): The compiler errors from Boost Graph are just about impossible to understand!
  // For some reason, boost::connected_components() will only work if the graph has a vertex index
  // property. If not, the compiler complains about a reference to void.
  // Next I realized that running boost::connected_components with a void* vertex_descriptor causes
  // some weird hidden memory corrupting. Easier to just build a graph here from scratch.
  const int num_comp = boost::connected_components(subgraph, &cluster_ids[0]);

  LmkClusters out(num_comp);

  for (uid_t v = 0; v < cluster_ids.size(); ++v) {
    const uid_t lmk_id = vtx_to_lmk_id.at(v);
    out.at(cluster_ids.at(v)).insert(lmk_id);
  }

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
