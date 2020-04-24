#include "ecbs_search.h"
#include <exception>
#include <iostream>
#include <utility>
#include <list>
#include <vector>
#include <tuple>
#include <ctime>
#include <climits>
#include <set>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
namespace pt = boost::property_tree;


inline void ECBSSearch::releaseClosedListNodes() {
  hashtable_t::iterator it;
  for (it=m_allNodes_table.begin(); it != m_allNodes_table.end(); it++) {
    delete ( (*it).first );  // should it be .second?
  }
}


// computes High-Level lower-bound based
inline double ECBSSearch::compute_hl_lower_bound() {
  double retVal = 0;
  for (int i = 0; i < m_num_of_agents; i++) {
    retVal += m_ll_min_f_vals[i];
  }
  return retVal;
}


// adding new nodes to FOCAL (those with min-f-val*f_weight between the old and new LB)
void ECBSSearch::updateFocalList(double old_lower_bound, double new_lower_bound, double f_weight) {
  for (ECBSNode* n : m_open_list) {
    if ( n->sum_min_f_vals > old_lower_bound &&
         n->sum_min_f_vals <= new_lower_bound )
      n->focal_handle = m_focal_list.push(n);
  }
}


// takes the paths_found_initially and UPDATE all (constrained) paths found for agents from curr to start
// also, do the same for ll_min_f_vals and paths_costs (since its already "on the way").
inline void ECBSSearch::updatePaths(ECBSNode* cur, ECBSNode* root_node) {
  const ECBSNode* curr = cur;
  m_paths = m_paths_found_initially;
  m_ll_min_f_vals = m_ll_min_f_vals_found_initially;
  m_paths_costs = m_paths_costs_found_initially;
  vector<bool> updated(m_num_of_agents, false);  // initialized for false
  /* used for backtracking -- only update paths[i] if it wasn't updated before (that is, by a younger node)
   * because younger nodes take into account ancesstors' nodes constraints. */
  while ( curr != root_node ) {
    if (updated[curr->agent_id] == false) {
      m_paths[curr->agent_id] = &(curr->path);
      m_ll_min_f_vals[curr->agent_id] = curr->ll_min_f_val;
      m_paths_costs[curr->agent_id] = curr->path_cost;
      updated[curr->agent_id] = true;
    }
    curr = curr->parent;
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// find all constraints on this agent (recursing to the root) and compute (and store) a path satisfying them.
// returns true only if such a path exists (otherwise false and path remain empty).
inline bool ECBSSearch::updateECBSNode(ECBSNode* leaf_node, ECBSNode* root_node) {
  // extract all constraints on leaf_node->agent_id
  list < tuple<int, int, int> > constraints;  // each constraint is <L1,L2,T>
  int agent_id = leaf_node->agent_id;
  //  cout << " update ECBS node for agent:" << agent_id << endl;
  const ECBSNode* curr = leaf_node;
  //  cout << "  Find all constraints on him:" << endl;
  while (curr != root_node) {
    if (curr->agent_id == agent_id) {
      for (const auto& c : curr->constraints) {
        constraints.push_front(c);
      }
      //      cout << "   L1:" << get<0>(curr->constraint) << " ; L2:" << get<1>(curr->constraint) << " ; T:" << get<2>(curr->constraint) << endl;
    }
    curr = curr->parent;
  }
  // cout << "  OVERALL #CONS:" << constraints.size() << endl;

  // calc constraints' max_timestep
  int max_timestep = -1;
  for (list< tuple<int, int, int> >::iterator it = constraints.begin(); it != constraints.end(); it++)
    if ( get<2>(*it) > max_timestep ) {
      max_timestep = get<2>(*it);
    }
  // cout << "  Latest constraint's timestep:" << max_timestep << endl;

  // initialize a constraint vector of length max_timestep+1. Each entry is an empty list< pair<int,int> > (loc1,loc2)
  //  cout << "  Creating a list of constraints (per timestep):" << endl;
  vector < list< pair<uint32_t, uint32_t> > >* cons_vec = new vector < list< pair<uint32_t, uint32_t> > > ( max_timestep+1, list< pair<uint32_t, uint32_t> > () );
  for (list< tuple<int, int, int> >::iterator it = constraints.begin(); it != constraints.end(); it++) {
    //    cout << "   PUSHING a constraint for time:" << get<2>(*it) << " ; (constraint is [" << get<0>(*it) << "," << get<1>(*it) << "])" << endl;
    cons_vec->at(get<2>(*it)).push_back(make_pair(get<0>(*it), get<1>(*it)));
  }

  // build reservation table
  size_t max_plan_len = getPathsMaxLength();
  // bool* res_table = new bool[map_size * max_plan_len]();  // initialized to false
  std::vector< std::unordered_set<vertex_t> > resTable(max_plan_len);
  std::vector< std::set<edge_t> > resTableEdges(max_plan_len);
  updateReservationTable(resTable, resTableEdges, max_plan_len, agent_id);

  //  printResTable(res_table, max_plan_len);

  // find a path w.r.t cons_vec (and prioretize by res_table).
  bool foundSol = m_search_engines[agent_id]->findPath(m_focal_w, cons_vec, resTable, resTableEdges, max_plan_len);
  m_LL_num_expanded += m_search_engines[agent_id]->numExpanded();
  m_LL_num_generated += m_search_engines[agent_id]->numGenerated();

#ifndef NDEBUG
  cout << "Run search for AG" << agent_id << " ; found solution? " << std::boolalpha << foundSol;
#endif
  // update leaf's path to the one found and its low-level search's min f-val
  if (foundSol) {
    leaf_node->path = vector<uint32_t>(*(m_search_engines[agent_id]->getPath()));
    leaf_node->ll_min_f_val = m_search_engines[agent_id]->minFval();
    leaf_node->path_cost = m_search_engines[agent_id]->pathCost();
#ifndef NDEBUG
    cout << " ; path-cost=" << (leaf_node->path_cost) << " ; for which min-f-val=" << leaf_node->ll_min_f_val << " ; The path is:";
    for (auto it = leaf_node->path.begin(); it != leaf_node->path.end(); it++) {
      cout << *it << " ";
    }
#endif
  }
  //  cout << endl;

  // release memory allocated here and return
  delete (cons_vec);
  return foundSol;
}
////////////////////////////////////////////////////////////////////////////////

/*
  return agent_id's location for the given timestep
  Note -- if timestep is longer than its plan length,
          then the location remains the same as its last cell)
 */
inline uint32_t ECBSSearch::getAgentLocation(
  int agent_id,
  size_t timestep)
{
  // if last timestep > plan length, agent remains in its last location
  if (timestep >= m_paths[agent_id]->size()) {
    return m_paths[agent_id]->at(m_paths[agent_id]->size()-1);
  }
  // otherwise, return its location for that timestep
  return m_paths[agent_id]->at(timestep);
}

/*
  return true iff agent1 and agent2 switched locations at timestep [t,t+1]
 */
inline bool ECBSSearch::switchedLocations(int agent1_id, int agent2_id, size_t timestep) {
  // if both agents at their goal, they are done moving (cannot switch places)
  if (   timestep >= m_paths[agent1_id]->size()
      && timestep >= m_paths[agent2_id]->size() ) {
    return false;
  }
  if ( getAgentLocation(agent1_id, timestep) == getAgentLocation(agent2_id, timestep+1) &&
       getAgentLocation(agent1_id, timestep+1) == getAgentLocation(agent2_id, timestep) ) {
    return true;
  }
  return false;
}

bool ECBSSearch::generalConflict(int ag1, int ag2, size_t timestep)
{
  uint32_t loc1 = getAgentLocation(ag1, timestep);
  const auto& conflicts1 = (*m_searchGraph)[loc1].generalizedVertexConflicts;

  uint32_t loc2 = getAgentLocation(ag2, timestep);
  const auto& conflicts2 = (*m_searchGraph)[loc2].generalizedVertexConflicts;

  return  conflicts1.find(loc2) != conflicts1.end()
       || conflicts2.find(loc1) != conflicts2.end();
}

bool ECBSSearch::generalEdgeConflict(int ag1, int ag2, size_t timestep)
{
  // if (   timestep >= m_paths[ag1]->size()
  //     && timestep >= m_paths[ag2]->size() ) {
  //   return false;
  // }

  uint32_t loc1a = getAgentLocation(ag1, timestep);
  uint32_t loc1b = getAgentLocation(ag1, timestep+1);
  uint32_t loc2a = getAgentLocation(ag2, timestep);
  uint32_t loc2b = getAgentLocation(ag2, timestep+1);

  auto e1 = boost::edge(loc1a, loc1b, *m_searchGraph);
  auto e2 = boost::edge(loc2a, loc2b, *m_searchGraph);

  if (e1.second && e2.second) {
    const auto& conflicts1 = (*m_searchGraph)[e1.first].generalizedEdgeConflicts;
    const auto& conflicts2 = (*m_searchGraph)[e2.first].generalizedEdgeConflicts;

    return  conflicts1.find(e2.first) != conflicts1.end()
         || conflicts2.find(e1.first) != conflicts2.end();
  // } else if (loc1a == loc1b && e2.second) {
  //   const auto& conflicts2 = (*m_searchGraph)[e2.first].generalizedEdgeVertexConflicts;
  //   return conflicts2.find(loc1a) != conflicts2.end();
  // } else if (loc2a == loc2b && e1.second) {
  //   const auto& conflicts1 = (*m_searchGraph)[e1.first].generalizedEdgeVertexConflicts;
  //   return conflicts1.find(loc2a) != conflicts1.end();
  } else {
    return false;
  }
}

bool ECBSSearch::generalEdgeVertexConflict(int ag1, int ag2, size_t timestep)
{
  // if (   timestep >= m_paths[ag1]->size()
  //     && timestep >= m_paths[ag2]->size() ) {
  //   return false;
  // }

  uint32_t loc1a = getAgentLocation(ag1, timestep);
  uint32_t loc1b = getAgentLocation(ag1, timestep+1);
  uint32_t loc2a = getAgentLocation(ag2, timestep);
  uint32_t loc2b = getAgentLocation(ag2, timestep+1);

  auto e1 = boost::edge(loc1a, loc1b, *m_searchGraph);
  auto e2 = boost::edge(loc2a, loc2b, *m_searchGraph);

  if (loc1a == loc1b && e2.second) {
    const auto& conflicts2 = (*m_searchGraph)[e2.first].generalizedEdgeVertexConflicts;
    return conflicts2.find(loc1a) != conflicts2.end();
  } else if (loc2a == loc2b && e1.second) {
    const auto& conflicts1 = (*m_searchGraph)[e1.first].generalizedEdgeVertexConflicts;
    return conflicts1.find(loc2a) != conflicts1.end();
  } else {
    return false;
  }
}

#if 0
bool ECBSSearch::quadcopterDownwash(int ag1, int ag2, size_t timestep)
{
  return false;
  // two agents cannot be on top of each other
  float x1,y1,z1,x2,y2,z2;
  m_searchGraph->location(getAgentLocation(ag1, timestep), &x1, &y1, &z1);
  m_searchGraph->location(getAgentLocation(ag2, timestep), &x2, &y2, &z2);

  if (   fabs(x1 - x2) < 0.5
      && fabs(y1 - y2) < 0.5
      && fabs(z1 - z2) < 2.5) {
    std::cout << "downwash collision!" << getAgentLocation(ag1, timestep) << "," << getAgentLocation(ag2, timestep) << "," << timestep << std::endl;
    return true;
  }
  return false;
}
#endif

/*
  Emulate agents' paths and returns a vector of collisions
  Note - a collision is a tuple of <int agent1_id, agent2_id, int location1, int location2, int timestep>).
  Note - the tuple's location_2=-1 for vertex collision.
 */

vector< tuple<int, int, int, int, int, ECBSSearch::CollisionType, int, int> >* ECBSSearch::extractCollisions() {
  vector< tuple<int, int, int, int, int, CollisionType, int, int> >* cons_found = new vector< tuple<int, int, int, int, int, CollisionType, int, int> >();
  m_earliest_conflict = make_tuple(-1, -1, -1, -1, INT_MAX, CollisionTypeVertex, -1, -1);
  for (int a1 = 0; a1 < m_num_of_agents; a1++) {
    for (int a2 = a1+1; a2 < m_num_of_agents; a2++) {
      size_t max_path_length = m_paths[a1]->size() > m_paths[a2]->size() ? m_paths[a1]->size() : m_paths[a2]->size();
      for (size_t timestep = 0; timestep < max_path_length; timestep++) {
        if ( getAgentLocation(a1, timestep) == getAgentLocation(a2, timestep) ) {
        // if (m_searchGraph->dist(getAgentLocation(a1, timestep), getAgentLocation(a2, timestep)) < 0.35) {
          cons_found->push_back(make_tuple(a1,
                                           a2,
                                           getAgentLocation(a1, timestep),
                                           -1,  // vertex collision (hence loc2=-1)
                                           timestep,
                                           CollisionTypeVertex, -1, -1) );
          if ((int)timestep < std::get<4>(m_earliest_conflict)) {
            m_earliest_conflict = cons_found->back();
          }
        }
        if ( switchedLocations(a1, a2, timestep) ) {
          cons_found->push_back(make_tuple(a1,
                                           a2,
                                           getAgentLocation(a1, timestep),
                                           getAgentLocation(a2, timestep),
                                           timestep,
                                           CollisionTypeEdge, -1, -1) );
          if ((int)timestep < std::get<4>(m_earliest_conflict)) {
            m_earliest_conflict = cons_found->back();
          }
        }
        if (generalConflict(a1, a2, timestep)) {
          cons_found->push_back(make_tuple(a1,
                                           a2,
                                           getAgentLocation(a1, timestep),
                                           getAgentLocation(a2, timestep),
                                           timestep,
                                           CollisionTypeTwoVertices, -1, -1) );
          if ((int)timestep < std::get<4>(m_earliest_conflict)) {
            m_earliest_conflict = cons_found->back();
          }

        }
        if ( generalEdgeConflict(a1, a2, timestep) ) {
          cons_found->push_back(make_tuple(a1,
                                           a2,
                                           getAgentLocation(a1, timestep),
                                           getAgentLocation(a1, timestep+1),
                                           timestep,
                                           CollisionTypeGeneralEdge,
                                           getAgentLocation(a2, timestep),
                                           getAgentLocation(a2, timestep+1)) );
          if ((int)timestep < std::get<4>(m_earliest_conflict)) {
            m_earliest_conflict = cons_found->back();
          }
        }
        if ( generalEdgeVertexConflict(a1, a2, timestep) ) {
          cons_found->push_back(make_tuple(a1,
                                           a2,
                                           getAgentLocation(a1, timestep),
                                           getAgentLocation(a1, timestep+1),
                                           timestep,
                                           CollisionTypeGeneralEdgeVertex,
                                           getAgentLocation(a2, timestep),
                                           getAgentLocation(a2, timestep+1)) );
          if ((int)timestep < std::get<4>(m_earliest_conflict)) {
            m_earliest_conflict = cons_found->back();
          }
        }
      }
    }
  }
  return cons_found;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Returns the maximal path length (among all agent)
size_t ECBSSearch::getPathsMaxLength() {
  size_t retVal = 0;
  for (int ag = 0; ag < m_num_of_agents; ag++)
    if ( m_paths[ag] != NULL && m_paths[ag]->size() > retVal )
      retVal = m_paths[ag]->size();
  return retVal;
}

// Generates a boolean reservation table for paths (cube of map_size*max_timestep).
// This is used by the low-level ECBS to count possible collisions efficiently
// Note -- we do not include the agent for which we are about to plan for
void ECBSSearch::updateReservationTable(
  std::vector< std::unordered_set<vertex_t> >& resTable,
  std::vector< std::set<edge_t> >& resTableEdges,
  size_t max_path_len,
  int exclude_agent)
{
  for (int ag = 0; ag < m_num_of_agents; ag++) {
    if (ag != exclude_agent && m_paths[ag] != NULL) {
      for (size_t timestep = 0; timestep < max_path_len; timestep++) {
        int id = getAgentLocation(ag, timestep);
        resTable[timestep].insert(id);

        const auto& vertexConflicts = (*m_searchGraph)[id].generalizedVertexConflicts;
        resTable[timestep].insert(vertexConflicts.begin(), vertexConflicts.end());

        if (timestep > 0) {
          vertex_t last = getAgentLocation(ag, timestep - 1);
          if (id == last) {// agent executed wait action
            const auto& vertexEdgeConflicts = (*m_searchGraph)[id].generalizedVertexEdgeConflicts;
            resTableEdges[timestep].insert(vertexEdgeConflicts.begin(), vertexEdgeConflicts.end());
          }
        }

        if (timestep < max_path_len - 1) {
          vertex_t u = getAgentLocation(ag, timestep);
          vertex_t v = getAgentLocation(ag, timestep+1);
          auto e = boost::edge(u, v, *m_searchGraph);
          if (e.second) {
            const auto& edgeConflicts = (*m_searchGraph)[e.first].generalizedEdgeConflicts;
            resTableEdges[timestep].insert(e.first);
            resTableEdges[timestep].insert(edgeConflicts.begin(), edgeConflicts.end());

            const auto& edgeVertexConflicts = (*m_searchGraph)[e.first].generalizedEdgeVertexConflicts;
            resTable[timestep].insert(edgeVertexConflicts.begin(), edgeVertexConflicts.end());
            resTable[timestep+1].insert(edgeVertexConflicts.begin(), edgeVertexConflicts.end());
          }
        }
      }
    }
  }
}

// Compute the number of pairs of agents colliding (h_3 in ECBS's paper)
int ECBSSearch::computeNumOfCollidingAgents() {
  //  cout << "   *-*-* Computed number of colliding agents: " << endl;
  int retVal = 0;
  for (int a1 = 0; a1 < m_num_of_agents; a1++) {
    for (int a2 = a1+1; a2 < m_num_of_agents; a2++) {
      size_t max_path_length = m_paths[a1]->size() > m_paths[a2]->size() ? m_paths[a1]->size() : m_paths[a2]->size();
      for (size_t timestep = 0; timestep < max_path_length; timestep++) {
        //        cout << "   A1:" << getAgentLocation(a1, timestep) << ", A2:" <<  getAgentLocation(a2, timestep) << ", T:" << timestep;
        if ( getAgentLocation(a1, timestep) == getAgentLocation(a2, timestep) ||
        // if (m_searchGraph->dist(getAgentLocation(a1, timestep), getAgentLocation(a2, timestep)) < 0.35 ||
             switchedLocations(a1, a2, timestep) ||
             generalConflict(a1, a2, timestep) ||
             generalEdgeConflict(a1, a2, timestep) ||
             generalEdgeVertexConflict(a1, a2, timestep)) {
          retVal++;
          // break to the outer (a1) loop
          timestep = max_path_length;
          a2 = m_num_of_agents;
          //          cout << " !BOOM! ";
        }
      }
    }
  }
  //  cout << "   *-*-* Computed number of colliding agents returns: " << retVal << endl;
  return retVal;
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////
ECBSSearch::ECBSSearch(
  const searchGraph_t* searchGraph,
  const std::vector<Agent>& agents,
  double e_w,
  double e_f,
  const std::vector< std::vector<uint32_t> >& initialPaths,
  bool tweak_g_val)
  : m_focal_w(e_f)
  , m_focal_list_threshold(0)
  , m_min_sum_f_vals(0)
  , m_paths()
  , m_paths_found_initially()
  , m_solution_found(false)
  , m_solution_cost(-1)
  , m_dummy_start(nullptr)
  , m_start_locations()
  , m_goal_locations()
  , m_searchGraph(searchGraph)
  , m_num_of_agents(0)
  , m_HL_num_expanded(0)
  , m_HL_num_generated(0)
  , m_LL_num_expanded(0)
  , m_LL_num_generated(0)
  , m_open_list()
  , m_focal_list()
  , m_allNodes_table()
  , m_empty_node(nullptr)
  , m_deleted_node(nullptr)
  , m_heuristics()
  , m_search_engines()
  , m_ll_min_f_vals_found_initially()
  , m_ll_min_f_vals()
  , m_paths_costs_found_initially()
  , m_paths_costs()
  , m_earliest_conflict()
  {
  // check input
  m_num_of_agents = agents.size();
  // a) distinct initial locations (not on obstacle)
  std::set<int> startSet;
  for (const auto& agent : agents) {
    if (startSet.find(agent.start) != startSet.end()) {
      throw std::runtime_error("start locations are not unique!");
    }
    startSet.insert(agent.start);
  }
  // b) distinct goal locations (not on obstacle)
  std::set<int> goalSet;
  for (const auto& agent : agents) {
    if (goalSet.find(agent.goal) != goalSet.end()) {
      throw std::runtime_error("goal locations are not unique!");
    }
    goalSet.insert(agent.goal);
  }

  m_ll_min_f_vals.resize(m_num_of_agents);
  m_paths_costs.resize(m_num_of_agents);
  m_ll_min_f_vals_found_initially.resize(m_num_of_agents);
  m_paths_costs_found_initially.resize(m_num_of_agents);
  m_heuristics.resize(m_num_of_agents);
  m_search_engines.resize(m_num_of_agents);
  for (int i = 0; i < m_num_of_agents; i++) {
    const auto& agent = agents[i];
    int init_loc = agent.start;
    int goal_loc = agent.goal;
    m_heuristics[i] = new ComputeHeuristic(m_searchGraph, i, goal_loc, e_w);
    m_search_engines[i] = new SingleAgentECBS(i, init_loc, goal_loc, m_heuristics[i], m_searchGraph, e_w);
  }

  // initialize allNodes_table (hash table)
  m_empty_node = new ECBSNode();
  m_empty_node->time_generated = -2;
  m_empty_node->agent_id = -2;
  m_deleted_node = new ECBSNode();
  m_deleted_node->time_generated = -3;
  m_deleted_node->agent_id = -3;
  m_allNodes_table.set_empty_key(m_empty_node);
  m_allNodes_table.set_deleted_key(m_deleted_node);

  // initialize all initial paths to NULL
  m_paths_found_initially.resize(m_num_of_agents, nullptr);

  // initialize paths_found_initially
  // if (initialPaths.size() > 0)
  // {
    // initialize from existing paths
    for (int i = 0; i < m_num_of_agents; i++) {
      // a) check if existing path is actually valid
      bool valid = false;
      float cost = 0;
      float min_f_val = std::numeric_limits<float>::max();
      if (   i < initialPaths.size()
          && initialPaths[i].size() > 0
          && initialPaths[i].front() == agents[i].start
          && initialPaths[i].back() == agents[i].goal)
      {
        valid = true;
        for (size_t t = 0; t < initialPaths[i].size() - 1; ++t) {
          uint32_t loc1 = initialPaths[i][t];
          uint32_t loc2 = initialPaths[i][t+1];
          if (loc1 != loc2)
          {
            auto e = boost::edge(loc1, loc2, *m_searchGraph);
            if (!e.second) {
              valid = false;
              std::cout << "edge not in searchgraph" << std::endl;
              break;
            } else {
              cost += 1.0; //(*m_searchGraph)[e.first].length;
            }
          } else {
            cost += 1.0;
          }
          float hval = m_heuristics[i]->getHValue(loc1);
          float gval = cost;
          float fval = hval + gval;
          min_f_val = std::min(min_f_val, fval);
        }
        uint32_t loc = initialPaths[i].back();
        float hval = m_heuristics[i]->getHValue(loc);
        float gval = cost;
        float fval = hval + gval;
        min_f_val = std::min(min_f_val, fval);
      } else {
        // std::cout << "path does not start with startNode or end with goal node" << std::endl;
        // std::cout << "start: " << agents[i].start << " goal: " << agents[i].goal << std::endl;
        // std::cout << "path: ";
        // if (i < initialPaths.size()) {
        //   for (uint32_t loc : initialPaths[i]) {
        //     std::cout << loc << ",";
        //   }
        // }
        // std::cout << std::endl;
      }

      if (valid)
      {
        // std::cout << "Plan for agent " << i << " is VALID" << std::endl;

        m_paths_found_initially[i] = new vector<uint32_t>(initialPaths[i]);

        m_ll_min_f_vals_found_initially[i] = min_f_val;
        m_paths_costs_found_initially[i] = cost;
      } else {
        // std::cout << "Plan for agent " << i << " is NOT valid" << std::endl;

        //    cout << "Computing initial path for agent " << i << endl; fflush(stdout);
        m_paths = m_paths_found_initially;
        size_t max_plan_len = getPathsMaxLength();
        std::vector< std::unordered_set<vertex_t> > resTable(max_plan_len);
        std::vector< std::set<edge_t> > resTableEdges(max_plan_len);
        updateReservationTable(resTable, resTableEdges, max_plan_len, i);
        //    cout << "*** CALCULATING INIT PATH FOR AGENT " << i << ". Reservation Table[MAP_SIZE x MAX_PLAN_LEN]: " << endl;
        //    printResTable(res_table, max_plan_len);
        if (!m_search_engines[i]->findPath(e_f, NULL, resTable, resTableEdges, max_plan_len )) {
          cout << "NO SOLUTION EXISTS";
        }
        m_paths_found_initially[i] = new vector<uint32_t > (*(m_search_engines[i]->getPath()));
        m_ll_min_f_vals_found_initially[i] = m_search_engines[i]->minFval();
        m_paths_costs_found_initially[i] = m_search_engines[i]->pathCost();
        m_LL_num_expanded += m_search_engines[i]->numExpanded();
        m_LL_num_generated += m_search_engines[i]->numGenerated();
      }
    }

    // std::cout << "min_f_vals:" << std::endl;
    // for (int i = 0; i < m_num_of_agents; i++) {
    //   std::cout << m_ll_min_f_vals_found_initially[i] << ",";
    // }
    // std::cout << std::endl;

    // std::cout << "cost:" << std::endl;
    // for (int i = 0; i < m_num_of_agents; i++) {
    //   std::cout << m_paths_costs_found_initially[i] << ",";
    // }
    // std::cout << std::endl;


  // }
  // else {
  //   // initialize from scratch

  //   for (int i = 0; i < m_num_of_agents; i++) {
  //     //    cout << "Computing initial path for agent " << i << endl; fflush(stdout);
  //     m_paths = m_paths_found_initially;
  //     size_t max_plan_len = getPathsMaxLength();
  //     std::vector< std::unordered_set<uint32_t> > resTable(max_plan_len);
  //     updateReservationTable(resTable, max_plan_len, i);
  //     //    cout << "*** CALCULATING INIT PATH FOR AGENT " << i << ". Reservation Table[MAP_SIZE x MAX_PLAN_LEN]: " << endl;
  //     //    printResTable(res_table, max_plan_len);
  //     if (!m_search_engines[i]->findPath(e_f, NULL, resTable, max_plan_len )) {
  //       cout << "NO SOLUTION EXISTS";
  //     }
  //     m_paths_found_initially[i] = new vector<uint32_t > (*(m_search_engines[i]->getPath()));
  //     m_ll_min_f_vals_found_initially[i] = m_search_engines[i]->minFval();
  //     m_paths_costs_found_initially[i] = m_search_engines[i]->pathCost();
  //     m_LL_num_expanded += m_search_engines[i]->numExpanded();
  //     m_LL_num_generated += m_search_engines[i]->numGenerated();
  //   }
  // }

  m_paths = m_paths_found_initially;
  m_ll_min_f_vals = m_ll_min_f_vals_found_initially;
  m_paths_costs = m_paths_costs_found_initially;

  // generate dummy start and update data structures
  m_dummy_start = new ECBSNode();
  m_dummy_start->agent_id = -1;
  m_dummy_start->g_val = 0;
  for (int i = 0; i < m_num_of_agents; i++) {
    m_dummy_start->g_val += m_paths_costs[i];
  }
  m_dummy_start->ll_min_f_val = 0;
  m_dummy_start->sum_min_f_vals = compute_hl_lower_bound();
  m_dummy_start->open_handle = m_open_list.push(m_dummy_start);
  m_dummy_start->focal_handle = m_focal_list.push(m_dummy_start);
  m_HL_num_generated++;
  m_dummy_start->time_generated = m_HL_num_generated;
  m_allNodes_table[m_dummy_start] = m_dummy_start;

  m_min_sum_f_vals = m_dummy_start->sum_min_f_vals;
  m_focal_list_threshold = m_focal_w * m_dummy_start->sum_min_f_vals;

  //  cout << "Paths in START (high-level) node:" << endl;
  //  printPaths();
  //  cout << "SUM-MIN-F-VALS: " << dummy_start->sum_min_f_vals << endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool ECBSSearch::runECBSSearch() {
  // set timer
  std::clock_t start;
  double duration;
  start = std::clock();

  // start is already in the open_list
  while ( !m_focal_list.empty() && !m_solution_found ) {
    // break after 5 min
    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    if (duration > 300) {  // timeout after 1 minute
      cout << "TIMEOUT  ; " << m_solution_cost << " ; " << m_min_sum_f_vals << " ; " <<
          m_HL_num_expanded << " ; " << m_HL_num_generated << " ; " <<
          m_LL_num_expanded << " ; " << m_LL_num_generated << " ; " << duration << endl;
      return false;
    }

    ECBSNode* curr = m_focal_list.top();
    m_focal_list.pop();
    m_open_list.erase(curr->open_handle);
    m_HL_num_expanded++;
    curr->time_expanded = m_HL_num_expanded;
    //    cout << "Expanding: (" << curr << ")" << *curr << " at time:" << HL_num_expanded << endl;

    // takes the paths_found_initially and UPDATE all constrained paths found for agents from curr to dummy_start (and lower-bounds)
    updatePaths(curr, m_dummy_start);
    //    printPaths();

    vector< tuple<int, int, int, int, int, CollisionType, int, int> >* collision_vec = extractCollisions();  // check for collisions on updated paths
#ifndef NDEBUG
    cout << endl << "****** Expanded #" << curr->time_expanded << " with cost " << curr->g_val << " and # Collisions " << collision_vec->size() << " and |FOCAL|=" << m_focal_list.size() << " and focal-threshold=" << m_focal_list_threshold << endl;
#endif
    /*
    cout << "Collision found in the expanded node's paths:" << endl;
    for (vector< tuple<int,int,int,int,int> >::const_iterator it = collision_vec->begin(); it != collision_vec->end(); it++)
      cout << "   A1:" << get<0>(*it) << " ; A2:" << get<1>(*it) << " ; L1:" << get<2>(*it) << " ; L2:" << get<3>(*it) << " ; T:" << get<4>(*it) << endl;
    cout << "Overall Col_Vec.size=" << collision_vec->size() << endl;
     */

    if ( collision_vec->size() == 0 ) {  // found a solution (and finish the while look)
      m_solution_found = true;
      m_solution_cost = curr->g_val;
    } else {  // generate the two successors that resolve one of the conflicts
      int agent1_id, agent2_id, location1, location2, timestep, loc1b, loc2b;
      CollisionType collType;
      tie(agent1_id, agent2_id, location1, location2, timestep, collType, loc1b, loc2b) = m_earliest_conflict;  // choose differently? (used to be collision_vec->at(0))
#ifndef NDEBUG
      cout << "   Earliest collision -- A1:" << agent1_id << " ; A2: " << agent2_id
           << " ; L1:" << location1 << " ; L2:" << location2 << " ; T:" << timestep << endl;
#endif
      ECBSNode* n1 = new ECBSNode();
      ECBSNode* n2 = new ECBSNode();
      n1->agent_id = agent1_id;
      n2->agent_id = agent2_id;
      switch (collType)
      {
      case CollisionTypeVertex:
        {
          // either agent1 should not visit that vertex...
          {
            n1->constraints.push_back(make_tuple(location1, -1, timestep));
            // const auto& conflicts1 = (*m_searchGraph)[location1].generalizedVertexConflicts;
            // for (vertex_t c : conflicts1) {
            //   n1->constraints.push_back(make_tuple(c, -1, timestep));
            // }
            // const auto& conflicts2 = (*m_searchGraph)[location1].generalizedVertexEdgeConflicts;
            // for (edge_t e : conflicts2) {
            //   n1->constraints.push_back(make_tuple(boost::source(e, *m_searchGraph), boost::target(e, *m_searchGraph), timestep));
            // }
          }

          // ..or agent 2 should not visit that vertex.
          {
            n2->constraints.push_back(make_tuple(location1, -1, timestep));
            // const auto& conflicts1 = (*m_searchGraph)[location1].generalizedVertexConflicts;
            // for (vertex_t c : conflicts1) {
            //   n2->constraints.push_back(make_tuple(c, -1, timestep));
            // }
            // const auto& conflicts2 = (*m_searchGraph)[location1].generalizedVertexEdgeConflicts;
            // for (edge_t e : conflicts2) {
            //   n2->constraints.push_back(make_tuple(boost::source(e, *m_searchGraph), boost::target(e, *m_searchGraph), timestep));
            // }
          }
        }
        break;
      case CollisionTypeEdge:
        // either agent1 should not take that edge...
        {
          n1->constraints.push_back(make_tuple(location1, location2, timestep));
          // auto e1 = boost::edge(location1, location2, *m_searchGraph);
          // const auto& conflicts1 = (*m_searchGraph)[e1.first].generalizedEdgeConflicts;
          // for (edge_t e : conflicts1) {
          //   n1->constraints.push_back(make_tuple(boost::source(e, *m_searchGraph), boost::target(e, *m_searchGraph), timestep));
          // }
          // const auto& conflicts2 = (*m_searchGraph)[e1.first].generalizedEdgeVertexConflicts;
          // for (vertex_t v : conflicts2) {
          //   n1->constraints.push_back(make_tuple(v, -1, timestep));
          // }
        }
        // ... or agent 2 should not traverse this edge (in the opposite direction)
        {
          n2->constraints.push_back(make_tuple(location2, location1, timestep));
          // auto e1 = boost::edge(location1, location2, *m_searchGraph);
          // const auto& conflicts1 = (*m_searchGraph)[e1.first].generalizedEdgeConflicts;
          // for (edge_t e : conflicts1) {
          //   n2->constraints.push_back(make_tuple(boost::source(e, *m_searchGraph), boost::target(e, *m_searchGraph), timestep));
          // }
          // const auto& conflicts2 = (*m_searchGraph)[e1.first].generalizedEdgeVertexConflicts;
          // for (vertex_t v : conflicts2) {
          //   n2->constraints.push_back(make_tuple(v, -1, timestep));
          // }
        }
        break;
      case CollisionTypeTwoVertices:
        {
          // either agent1 should not be on that vertex (and all vertices conflicting with it)
          {
            n1->constraints.push_back(make_tuple(location1, -1, timestep));
            // const auto& conflicts1 = (*m_searchGraph)[location1].generalizedVertexConflicts;
            // for (vertex_t c : conflicts1) {
            //   n1->constraints.push_back(make_tuple(c, -1, timestep));
            // }
            // const auto& conflicts2 = (*m_searchGraph)[location1].generalizedVertexEdgeConflicts;
            // for (edge_t e : conflicts2) {
            //   n1->constraints.push_back(make_tuple(boost::source(e, *m_searchGraph), boost::target(e, *m_searchGraph), timestep));
            // }
          }
          // ... or agent 2
          {
            n2->constraints.push_back(make_tuple(location2, -1, timestep));
            // const auto& conflicts1 = (*m_searchGraph)[location2].generalizedVertexConflicts;
            // for (vertex_t c : conflicts1) {
            //   n2->constraints.push_back(make_tuple(c, -1, timestep));
            // }
            // const auto& conflicts2 = (*m_searchGraph)[location2].generalizedVertexEdgeConflicts;
            // for (edge_t e : conflicts2) {
            //   n2->constraints.push_back(make_tuple(boost::source(e, *m_searchGraph), boost::target(e, *m_searchGraph), timestep));
            // }
          }
        }
        break;
      case CollisionTypeGeneralEdge:
        {
          // either agent1 should not be on that edge (and all edges conflicting with it)
          {
            n1->constraints.push_back(make_tuple(location1, location2, timestep));
            // auto e1 = boost::edge(location1, location2, *m_searchGraph);
            // const auto& conflicts1 = (*m_searchGraph)[e1.first].generalizedEdgeConflicts;
            // for (edge_t e : conflicts1) {
            //   n1->constraints.push_back(make_tuple(boost::source(e, *m_searchGraph), boost::target(e, *m_searchGraph), timestep));
            // }
            // const auto& conflicts2 = (*m_searchGraph)[e1.first].generalizedEdgeVertexConflicts;
            // for (vertex_t v : conflicts2) {
            //   n1->constraints.push_back(make_tuple(v, -1, timestep));
            // }
          }

          // ... or agent 2
          {
            n2->constraints.push_back(make_tuple(loc1b, loc2b, timestep));
            // auto e2 = boost::edge(loc1b, loc2b, *m_searchGraph);
            // const auto& conflicts1 = (*m_searchGraph)[e2.first].generalizedEdgeConflicts;
            // for (edge_t e : conflicts1) {
            //   n2->constraints.push_back(make_tuple(boost::source(e, *m_searchGraph), boost::target(e, *m_searchGraph), timestep));
            // }
            // const auto& conflicts2 = (*m_searchGraph)[e2.first].generalizedEdgeVertexConflicts;
            // for (vertex_t v : conflicts2) {
            //   n2->constraints.push_back(make_tuple(v, -1, timestep));
            // }
          }
        }
        break;
      case CollisionTypeGeneralEdgeVertex:
        {
          if (location1 == location2) {
            // std::cout << "ag1 stationary" << std::endl;
            // agent1 is stationary
            // either agent1 avoids that vertex
            {
              n1->constraints.push_back(make_tuple(location1, -1, timestep+1));
              // const auto& conflicts1 = (*m_searchGraph)[location1].generalizedVertexConflicts;
              // for (vertex_t c : conflicts1) {
              //   n1->constraints.push_back(make_tuple(c, -1, timestep+1));
              // }
              // const auto& conflicts2 = (*m_searchGraph)[location1].generalizedVertexEdgeConflicts;
              // for (edge_t e : conflicts2) {
              //   n1->constraints.push_back(make_tuple(boost::source(e, *m_searchGraph), boost::target(e, *m_searchGraph), timestep));
              // }
            }
            /// or agent2 avoids the edge
            {
              n2->constraints.push_back(make_tuple(loc1b, loc2b, timestep));
              // auto e2 = boost::edge(loc1b, loc2b, *m_searchGraph);
              // const auto& conflicts1 = (*m_searchGraph)[e2.first].generalizedEdgeConflicts;
              // for (edge_t e : conflicts1) {
              //   n2->constraints.push_back(make_tuple(boost::source(e, *m_searchGraph), boost::target(e, *m_searchGraph), timestep));
              // }
              // const auto& conflicts2 = (*m_searchGraph)[e2.first].generalizedEdgeVertexConflicts;
              // for (vertex_t v : conflicts2) {
              //   n2->constraints.push_back(make_tuple(v, -1, timestep));
              // }
            }
            // TODO: and all other edges
          } else {
            // agent2 is stationary
            // std::cout << "ag2 stationary" << std::endl;
            // either agent2 avoids that vertex
            {
              n2->constraints.push_back(make_tuple(loc1b, -1, timestep+1));
              // const auto& conflicts1 = (*m_searchGraph)[loc1b].generalizedVertexConflicts;
              // for (vertex_t c : conflicts1) {
              //   n2->constraints.push_back(make_tuple(c, -1, timestep));
              // }
              // const auto& conflicts2 = (*m_searchGraph)[location1].generalizedVertexEdgeConflicts;
              // for (edge_t e : conflicts2) {
              //   n2->constraints.push_back(make_tuple(boost::source(e, *m_searchGraph), boost::target(e, *m_searchGraph), timestep));
              // }
            }
            /// or agent1 avoids the edge
            {
              n1->constraints.push_back(make_tuple(location1, location2, timestep));
              // auto e2 = boost::edge(location1, location2, *m_searchGraph);
              // const auto& conflicts1 = (*m_searchGraph)[e2.first].generalizedEdgeConflicts;
              // for (edge_t e : conflicts1) {
              //   n1->constraints.push_back(make_tuple(boost::source(e, *m_searchGraph), boost::target(e, *m_searchGraph), timestep));
              // }
              // const auto& conflicts2 = (*m_searchGraph)[e2.first].generalizedEdgeVertexConflicts;
              // for (vertex_t v : conflicts2) {
              //   n1->constraints.push_back(make_tuple(v, -1, timestep));
              // }
            }
            // TODO: and all other edges...
          }
        }
        break;
      }
      n1->parent = curr;
      //      cout << "*** Before solving, " << endl << *n1;
      // find all constraints on this agent (recursing to the root) and compute (and store) a path satisfying them. Also updates n1's g_val
      if ( updateECBSNode(n1, m_dummy_start) == true ) {
        // new g_val equals old g_val plus the new path length found for the agent minus its old path length
        n1->g_val = curr->g_val - m_paths_costs[n1->agent_id] + n1->path_cost;
        // update n1's path for computing the num of colliding agents
        const vector<uint32_t >* temp_old_path = m_paths[n1->agent_id];
        m_paths[n1->agent_id] = &(n1->path);
        n1->h_val = computeNumOfCollidingAgents();
        m_paths[n1->agent_id] = temp_old_path;  // restore the old path (for n2)
        // update lower bounds and handles
        n1->sum_min_f_vals = curr->sum_min_f_vals - m_ll_min_f_vals[n1->agent_id] + n1->ll_min_f_val;
        n1->open_handle = m_open_list.push(n1);
        m_HL_num_generated++;
        n1->time_generated = m_HL_num_generated;
        if ( n1->sum_min_f_vals <= m_focal_list_threshold )
          n1->focal_handle = m_focal_list.push(n1);
        m_allNodes_table[n1] = n1;
#ifndef NDEBUG
	cout << endl << "   First node generated for A" << n1->agent_id << ": g-val=" << n1->g_val << " ; h-val=" << n1->h_val << " ; LB=" << n1->sum_min_f_vals << endl;
#endif
      } else {
        delete (n1);
      }
      // same for n2
      //      cout << "*** Before solving, " << endl << *n2;
      if (n2) {
        n2->parent = curr;
        if ( updateECBSNode(n2, m_dummy_start) == true ) {
          n2->g_val = curr->g_val - m_paths_costs[n2->agent_id] + n2->path_cost;
          const vector<uint32_t>* temp_old_path = m_paths[n2->agent_id];
          m_paths[n2->agent_id] = &(n2->path);
          n2->h_val = computeNumOfCollidingAgents();
          m_paths[n2->agent_id] = temp_old_path;
          n2->sum_min_f_vals = curr->sum_min_f_vals - m_ll_min_f_vals[n2->agent_id] + n2->ll_min_f_val;
          n2->open_handle = m_open_list.push(n2);
          m_HL_num_generated++;
          n2->time_generated = m_HL_num_generated;
          if ( n2->sum_min_f_vals <= m_focal_list_threshold ) {
            n2->focal_handle = m_focal_list.push(n2);
          }
          m_allNodes_table[n2] = n2;
  #ifndef NDEBUG
  	cout << endl << "   Second node generated for A" << n2->agent_id << ": g-val=" << n2->g_val << " ; h-val=" << n2->h_val << " ; LB=" << n2->sum_min_f_vals << endl;
  #endif
        } else {
          delete (n2);
        }
      }
      //            cout << "It has found the following paths:" << endl;
      //            printPaths();
      //            cout << "Focal threshold: (before) " << focal_list_threshold;
      if (m_open_list.size() == 0) {
        m_solution_found = false;
        break;
      }
      ECBSNode* open_head = m_open_list.top();
    if ( open_head->sum_min_f_vals > m_min_sum_f_vals ) {
#ifndef NDEBUG
	cout << "  Note -- FOCAL UPDATE!! from |FOCAL|=" << m_focal_list.size() << " with |OPEN|=" << m_open_list.size() << " to |FOCAL|=";
#endif
        m_min_sum_f_vals = open_head->sum_min_f_vals;
        double new_focal_list_threshold = m_min_sum_f_vals * m_focal_w;
        updateFocalList(m_focal_list_threshold, new_focal_list_threshold, m_focal_w);
        m_focal_list_threshold = new_focal_list_threshold;
	    cout << m_focal_list.size() << endl;
      }
      //            cout << " ; (after) " << focal_list_threshold << endl << endl;
    }  // end generating successors
    delete (collision_vec);
  }  // end of while loop

  // get time
  duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

  if (m_solution_found)
    cout << "1 ; ";
  else
    cout << "0 ; ";
  cout << m_solution_cost << " ; " << m_min_sum_f_vals << " ; " <<
      m_HL_num_expanded << " ; " << m_HL_num_generated << " ; " <<
      m_LL_num_expanded << " ; " << m_LL_num_generated << " ; " << duration << endl;
    //    printPaths();

    // std::cout << "min_f_vals:" << std::endl;
    // for (int i = 0; i < m_num_of_agents; i++) {
    //   std::cout << m_ll_min_f_vals[i] << ",";
    // }
    // std::cout << std::endl;

    // std::cout << "cost:" << std::endl;
    // for (int i = 0; i < m_num_of_agents; i++) {
    //   std::cout << m_paths_costs[i] << ",";
    // }
    // std::cout << std::endl;

  return m_solution_found;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ECBSSearch::~ECBSSearch() {
  for (size_t i = 0; i < m_search_engines.size(); i++)
    delete (m_search_engines[i]);
  for (size_t i = 0; i < m_paths_found_initially.size(); i++)
    delete (m_paths_found_initially[i]);
  //  for (size_t i=0; i<paths.size(); i++)
  //    delete (paths[i]);
  releaseClosedListNodes();
  delete (m_empty_node);
  delete (m_deleted_node);
}
