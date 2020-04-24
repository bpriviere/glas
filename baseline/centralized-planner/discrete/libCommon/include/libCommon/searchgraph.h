#pragma once

#include <set>
#include <unordered_map>

#include <boost/graph/adjacency_list.hpp>

#include "libCommon/common.h"

typedef boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::undirectedS > searchGraphTraits_t;
typedef searchGraphTraits_t::vertex_descriptor vertex_t;
typedef searchGraphTraits_t::edge_descriptor edge_t;

struct Vertex
{
  std::string name;
  position_t pos;
  std::set<vertex_t> generalizedVertexConflicts;
  std::set<edge_t> generalizedVertexEdgeConflicts;
};

struct Edge
{
  std::string name;
  std::set<edge_t> generalizedEdgeConflicts;
  std::set<vertex_t> generalizedEdgeVertexConflicts;
  float length;
  bool isHighway;
};

typedef boost::adjacency_list<
        boost::vecS, boost::vecS, boost::undirectedS,
        Vertex, Edge>
        searchGraph_t;

void loadSearchGraph(
  searchGraph_t& searchGraph,
  std::unordered_map<std::string, vertex_t>& vNameToV,
  std::unordered_map<std::string, edge_t>& eNameToE,
  const std::string& fileName);

void saveSearchGraph(
  const searchGraph_t& searchGraph,
  const std::string& fileName);

// boost graph helpers
// http://stackoverflow.com/questions/13453350/replace-bgl-iterate-over-vertexes-with-pure-c11-alternative

#include <boost/range/iterator_range.hpp>

template<class It>
boost::iterator_range<It> pair_range(std::pair<It, It> const& p){
  return boost::make_iterator_range(p.first, p.second);
}

// #include <boost/range/iterator_range.hpp>
// #include <type_traits>

// template<class T> using Invoke = typename T::type;
// template<class T> using RemoveRef = Invoke< std::remove_reference<T> >;
// template<class G> using VerticesIterator = typename boost::graph_traits<G>::vertices_iterator;
// template<class G> using OutEdgeIterator = typename boost::graph_traits<G>::out_edge_iterator;

// template<class G>
// auto vertices_range(G&& g)
//   -> boost::iterator_range<VerticesIterator<RemoveRef<G>>>
// {
//   auto vertex_pair = vertices(std::forward<G>(g));
//   return boost::make_iterator_range(vertex_pair.first, vertex_pair.second);
// }

// template<class V, class G>
// auto out_edges_range(V&& v, G&& g)
//   -> boost::iterator_range<OutEdgeIterator<RemoveRef<G>>>
// {
//   auto edge_pair = out_edges(std::forward<V>(v), std::forward<G>(g));
//   return boost::make_iterator_range(edge_pair.first, edge_pair.second);
// }
