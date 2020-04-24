#include <boost/heap/fibonacci_heap.hpp>
#include "compute_heuristic.h"
#include <cstring>
#include <limits>
#include <google/dense_hash_map>
#include <unordered_set>
#include "node.h"

using google::dense_hash_map;      // namespace where class lives by default
// using std::cout;
// using std::endl;
using boost::heap::fibonacci_heap;

ComputeHeuristic::ComputeHeuristic(
  const searchGraph_t* searchGraph,
  uint32_t agent,
  uint32_t goal_location,
  double highwayWeight)
  : m_hvalues()
{
  // generate a heap that can save nodes (and a open_handle)
  boost::heap::fibonacci_heap< Node* , boost::heap::compare<Node::compare_node> > heap;
  boost::heap::fibonacci_heap< Node* , boost::heap::compare<Node::compare_node> >::handle_type open_handle;
  // generate hash_map (key is a node pointer, data is a node handler,
  //                    NodeHasher is the hash function to be used,
  //                    eqnode is used to break ties when hash values are equal)
  dense_hash_map<Node*, fibonacci_heap<Node* , boost::heap::compare<Node::compare_node> >::handle_type, Node::NodeHasher, Node::eqnode> nodes;
  nodes.set_empty_key(NULL);
  dense_hash_map<Node*, fibonacci_heap<Node* , boost::heap::compare<Node::compare_node> >::handle_type, Node::NodeHasher, Node::eqnode>::iterator it; // will be used for find()

  Node* goal = new Node (goal_location, 0, 0, NULL, 0, false);
  goal->open_handle = heap.push(goal);  // add goal to heap
  nodes[goal] = goal->open_handle;       // add goal to hash_table (nodes)

  std::vector<uint32_t> neighbors;
  neighbors.reserve(8);

  while ( !heap.empty() ) {
    Node* curr = heap.top();
    heap.pop();

    for (const auto& e : pair_range(boost::in_edges(curr->loc, *searchGraph))) {
      // compute cost to prev_loc via curr node
      double cost = 1.0;//(*searchGraph)[e].length;
      if (!(*searchGraph)[e].isHighway) {
        cost = cost * highwayWeight;
      }
      double next_g_val = curr->g_val + cost;
      Node* next = new Node (boost::source(e, *searchGraph), next_g_val, 0, NULL, 0, false);
      it = nodes.find(next);
      if ( it == nodes.end() ) {  // add the newly generated node to heap and hash table
        next->open_handle = heap.push(next);
        nodes[next] = next->open_handle;
      } else {  // update existing node's g_val if needed (only in the heap)
        delete(next);  // not needed anymore -- we already generated it before
        Node* existing_next = it->first;
        open_handle = it->second;
        if (existing_next->g_val > next_g_val) {
          existing_next->g_val = next_g_val;
          heap.update(open_handle);
        }
      }
    }
  }
  // iterate over all nodes and populate the h_vals
  for (it=nodes.begin(); it != nodes.end(); it++) {
    Node* s = it->first;
    m_hvalues[s->loc] = s->g_val;
    delete s;
  }
  nodes.clear();
  heap.clear();
}

ComputeHeuristic::~ComputeHeuristic()
{

}

double ComputeHeuristic::getHValue(
  uint32_t location) const
{
  auto iter = m_hvalues.find(location);
  if (iter == m_hvalues.end()) {
    throw std::runtime_error("Could not find specified location!");
  }
  return iter->second;
}
