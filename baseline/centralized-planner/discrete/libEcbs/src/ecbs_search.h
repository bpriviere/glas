// ECBS Search (High-level)
#ifndef ECBSSEARCH_H
#define ECBSSEARCH_H

#include <boost/heap/fibonacci_heap.hpp>
#include <google/dense_hash_map>
#include <cstring>
#include <climits>
#include <tuple>
#include <string>
#include <vector>
#include <list>
#include "libCommon/map.h"
#include "compute_heuristic.h"
#include "single_agent_ecbs.h"
#include "ecbs_node.h"
#include "ecbs.h"

using boost::heap::fibonacci_heap;
using boost::heap::compare;
// using std::cout;
// using std::endl;
using google::dense_hash_map;

class ECBSSearch {
public:
  ECBSSearch(
    const searchGraph_t* searchGraph,
    const std::vector<Agent>& agents,
    double e_w,
    double e_f,
    const std::vector< std::vector<uint32_t> >& initialPaths,
    bool tweak_g_val = false);

  ~ECBSSearch();

  bool runECBSSearch();

    const vector < const vector<uint32_t>* >& paths() const {
      return m_paths;
    }

private:

  enum CollisionType
  {
    CollisionTypeVertex,
    CollisionTypeEdge,
    CollisionTypeTwoVertices,
    CollisionTypeGeneralEdge,
    CollisionTypeGeneralEdgeVertex,
  };

  inline double compute_g_val();
  inline double compute_hl_lower_bound();
  inline void updatePaths(
    ECBSNode* curr,
    ECBSNode* root_node);

  inline bool updateECBSNode(
    ECBSNode* leaf_node,
    ECBSNode* root_node);

  inline bool switchedLocations(
    int agent1_id,
    int agent2_id,
    size_t timestep);

  inline bool generalConflict(
    int agent1_id,
    int agent2_id,
    size_t timestep);

  bool generalEdgeConflict(
    int ag1,
    int ag2,
    size_t timestep);

  bool generalEdgeVertexConflict(
    int ag1,
    int ag2,
    size_t timestep);

  inline uint32_t getAgentLocation(
    int agent_id,
    size_t timestep);

  vector< tuple<int, int, int, int, int, CollisionType, int, int> >* extractCollisions();

  size_t getPathsMaxLength();

  void updateReservationTable(
    std::vector< std::unordered_set<vertex_t> >& resTable,
    std::vector< std::set<edge_t> >& resTableEdges,
    size_t max_plan_len,
    int exclude_agent);

  void updateFocalList(
    double old_lower_bound,
    double new_lower_bound,
    double f_weight);

  int computeNumOfCollidingAgents();

  inline void releaseClosedListNodes();

private:

  double m_focal_w = 1.0;
  double m_focal_list_threshold;
  double m_min_sum_f_vals;

  typedef boost::heap::fibonacci_heap< ECBSNode* , boost::heap::compare<ECBSNode::compare_node> > heap_open_t;
  typedef boost::heap::fibonacci_heap< ECBSNode* , boost::heap::compare<ECBSNode::secondary_compare_node> > heap_focal_t;
  typedef dense_hash_map<ECBSNode*, ECBSNode*, ECBSNode::ECBSNodeHasher, ECBSNode::ecbs_eqnode> hashtable_t;

  vector < const vector<uint32_t>* > m_paths;  // agents paths
  vector < const vector<uint32_t>* > m_paths_found_initially;  // contain initial paths found

  bool m_solution_found;
  double m_solution_cost;

  ECBSNode* m_dummy_start;
  vector <int> m_start_locations;
  vector <int> m_goal_locations;

  const searchGraph_t* m_searchGraph;
  int m_num_of_agents;

  uint64_t m_HL_num_expanded;
  uint64_t m_HL_num_generated;
  uint64_t m_LL_num_expanded;
  uint64_t m_LL_num_generated;

  heap_open_t m_open_list;
  heap_focal_t m_focal_list;
  hashtable_t m_allNodes_table;

  // used in hash table and would be deleted from the d'tor
  ECBSNode* m_empty_node;
  ECBSNode* m_deleted_node;

  vector < ComputeHeuristic* > m_heuristics;
  vector < SingleAgentECBS* > m_search_engines;  // used to find (single) agents' paths
  vector <double> m_ll_min_f_vals_found_initially;  // contains initial ll_min_f_vals found
  vector <double> m_ll_min_f_vals;  // each entry [i] represent the lower bound found for agent[i]
  vector <double> m_paths_costs_found_initially;
  vector <double> m_paths_costs;

  tuple<int, int, int, int, int, CollisionType, int, int> m_earliest_conflict;  // saves the earliest conflict (updated in every call to extractCollisions()).

};

#endif
