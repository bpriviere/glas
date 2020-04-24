#ifndef SINGLEAGENTECBS_H
#define SINGLEAGENTECBS_H

#include <stdlib.h>

#include <vector>
#include <list>
#include <unordered_set>
#include <utility>

#include <google/dense_hash_map>

#include "node.h"
#include "compute_heuristic.h"

// using std::cout;

class SingleAgentECBS {
public:
  /* ctor
   */
  // note if tweak_g_val is true, the costs are also inflated by e_weight
  SingleAgentECBS(
    uint32_t agent,
    uint32_t start_location,
    uint32_t goal_location,
    const ComputeHeuristic* my_heuristic,
    const searchGraph_t* searchGraph,
    double e_weight,
    bool tweak_g_val = false);

  ~SingleAgentECBS();

  /* Returns true if a collision free path found (with cost up to f_weight * f-min) while
     minimizing the number of internal conflicts (that is conflicts with known_paths for other agents found so far).
  */
  bool findPath(
    double f_weight,
    const vector < list< pair<uint32_t, uint32_t> > >* constraints,
    const std::vector< std::unordered_set<vertex_t> >& resTable, // per timestamp: set of occupied locations
    const std::vector< std::set<edge_t> >& resTableEdges, // per timestamp: set of occupied edges
    size_t max_plan_len);

  /* return a pointer to the path found.
   */
  const vector<uint32_t>* getPath() const
  {
    return &m_path;
  }

  uint64_t numExpanded() const
  {
    return m_num_expanded;
  }

  uint64_t numGenerated() const
  {
    return m_num_generated;
  }

  double minFval() const
  {
    return m_min_f_val;
  }

  double pathCost() const
  {
    return m_path_cost;
  }


private:
  // define typedefs (will also be used in ecbs_search)
  typedef boost::heap::fibonacci_heap< Node* , boost::heap::compare<Node::compare_node> > heap_open_t;
  typedef boost::heap::fibonacci_heap< Node* , boost::heap::compare<Node::secondary_compare_node> > heap_focal_t;
  //typedef boost::heap::fibonacci_heap< Node* , boost::heap::compare<Node::secondary_hwy_compare_node> > heap_focal_t;

  typedef google::dense_hash_map<Node*, Node*, Node::NodeHasher, Node::eqnode> hashtable_t;
  // note -- hash_map (key is a node pointer, data is a node handler,
  //                   NodeHasher is the hash function to be used,
  //                   eqnode is used to break ties when hash values are equal)

  /* returns the minimal plan length for the agent (that is, extract the latest timestep which
     has a constraint invloving this agent's goal location).
  */
  int extractLastGoalTimestep(
    uint32_t goal_location,
    const vector< list< pair<uint32_t, uint32_t> > >* cons);

  inline void releaseClosedListNodes(
    hashtable_t* allNodes_table);

  /* Checks if a vaild path found (wrt my_map and constraints)
     Note -- constraint[timestep] is a list of pairs. Each pair is a disallowed <loc1,loc2> (loc2=-1 for vertex constraint).
     Returns true/false.
  */
  inline bool isConstrained(
    uint32_t curr_id,
    uint32_t next_id,
    int next_timestep,
    const vector< list< pair<uint32_t, uint32_t> > >* cons);

  /* Updates the path datamember (vector<int>).
     After update it will contain the sequence of locations found from the goal to the start.
  */
  void updatePath(Node* goal);  // $$$ make inline?

  /* Return the number of conflicts between the known_paths' (by looking at the reservation table) for the move [curr_id,next_id].
     Returns 0 if no conflict, 1 for vertex or edge conflict, 2 for both.
   */
  int numOfConflictsForStep(
    uint32_t curr_id,
    uint32_t next_id,
    int next_timestep,
    // bool* res_table,
    const std::vector< std::unordered_set<vertex_t> >& resTable,
    const std::vector< std::set<edge_t> >& resTableEdges,
    int max_plan_len);

  /* Iterate over OPEN and adds to FOCAL all nodes with: 1) f-val > old_min_f_val ; and 2) f-val * f_weight < new_lower_bound.
   */
  void updateFocalList(
    double old_lower_bound,
    double new_lower_bound,
    double f_weight);


private:
  uint32_t m_agent;


  vector<uint32_t> m_path;  // a path that takes the agent from initial to goal location satisying all constraints
  // consider changing path from vector to deque (efficient front insertion)
  double m_path_cost;
  int m_start_location;
  int m_goal_location;
  const ComputeHeuristic* m_heuristic;  // this is the precomputed heuristic for this agent
  const searchGraph_t* m_searchGraph;
  uint64_t m_num_expanded;
  uint64_t m_num_generated;
  bool m_tweak_g_val;
  double m_e_weight;  // EGRAPH's inflation factor
  double m_lower_bound;  // FOCAL's lower bound ( = e_weight * min_f_val)
  double m_min_f_val;  // min f-val seen so far
  int m_num_non_hwy_edges;


  // note -- handle typedefs is defined inside the class (hence, include node.h is not enough).
  //  Node::open_handle_t open_handle;
  heap_open_t m_open_list;

  //  Node::focal_handle_t focal_handle;
  heap_focal_t m_focal_list;

  hashtable_t m_allNodes_table;

  // used in hash table and would be deleted from the d'tor
  Node* m_empty_node;
  Node* m_deleted_node;

};

#endif
