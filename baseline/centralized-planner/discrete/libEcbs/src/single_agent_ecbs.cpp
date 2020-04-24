#include "single_agent_ecbs.h"
#include <cstring>
#include <climits>
#include <vector>
#include <list>
#include <utility>
#include <iostream>

#include <boost/heap/fibonacci_heap.hpp>
#include <google/dense_hash_map>

#include "node.h"

using google::dense_hash_map;      // namespace where class lives by default
using std::cout;
using std::endl;
using boost::heap::fibonacci_heap;


SingleAgentECBS::SingleAgentECBS(
  uint32_t agent,
  uint32_t start_location,
  uint32_t goal_location,
  const ComputeHeuristic* heuristic,
  const searchGraph_t* searchGraph,
  double e_weight,
  bool tweak_g_val)
  : m_agent(agent)
  , m_path()
  , m_path_cost(0)
  , m_start_location(start_location)
  , m_goal_location(goal_location)
  , m_heuristic(heuristic)
  , m_searchGraph(searchGraph)
  , m_num_expanded(0)
  , m_num_generated(0)
  , m_tweak_g_val(tweak_g_val)
  , m_e_weight(e_weight)
  , m_lower_bound(0)
  , m_min_f_val(0)
  , m_num_non_hwy_edges(0)
  , m_open_list()
  , m_focal_list()
  , m_allNodes_table()
  , m_empty_node(nullptr)
  , m_deleted_node(nullptr)
{
  m_empty_node = new Node();
  m_empty_node->loc = -1;
  m_deleted_node = new Node();
  m_deleted_node->loc = -2;
  m_allNodes_table.set_empty_key(m_empty_node);
  m_allNodes_table.set_deleted_key(m_deleted_node);
}

SingleAgentECBS::~SingleAgentECBS() {
  delete m_empty_node;
  delete m_deleted_node;
}


void SingleAgentECBS::updatePath(Node* goal) {
  m_path.clear();
  const Node* curr = goal;
  // cout << "   UPDATING Path for one agent to: ";
  while (curr->timestep != 0) {
    m_path.resize(m_path.size() + 1);
    m_path.back() = curr->loc;
    //    cout << *curr << endl;
    curr = curr->parent;
  }
  m_path.resize(m_path.size() + 1);
  m_path.back() = m_start_location;
  reverse(m_path.begin(), m_path.end());
  m_path_cost = goal->g_val;
  m_num_non_hwy_edges = goal->num_non_hwy_edges;
}

inline void SingleAgentECBS::releaseClosedListNodes(hashtable_t* allNodes_table) {
  hashtable_t::iterator it;
  for (it=allNodes_table->begin(); it != allNodes_table->end(); it++) {
    delete ( (*it).second );  // Node* s = (*it).first; delete (s);
  }
}


// iterate over the constraints ( cons[t] is a list of all constraints for timestep t) and return the latest
// timestep which has a constraint involving the goal location
int SingleAgentECBS::extractLastGoalTimestep(
  uint32_t goal_location,
  const vector< list< pair<uint32_t, uint32_t> > >* cons)
{
  if (cons != NULL) {
    for ( int t = static_cast<int>(cons->size())-1; t > 0; t-- ) {
      for (list< pair<uint32_t, uint32_t> >::const_iterator it = cons->at(t).begin(); it != cons->at(t).end(); ++it) {
        // $$$: in the following if, do we need to check second (maybe cannot happen in edge constraints?)
        if ((*it).first == m_goal_location || (*it).second == m_goal_location) {
          return (t);
        }
      }
    }
  }
  return -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// input: curr_id (location at time next_timestep-1) ; next_id (location at time next_timestep); next_timestep
//        cons[timestep] is a list of <loc1,loc2> of (vertex/edge) constraints for that timestep.
inline bool SingleAgentECBS::isConstrained(
  uint32_t curr_id,
  uint32_t next_id,
  int next_timestep,
  const vector< list< pair<uint32_t, uint32_t> > >* cons)
{
  //  cout << "check if ID="<<id<<" is occupied at TIMESTEP="<<timestep<<endl;
  if (cons == NULL)
    return false;

  // check vertex constraints (being in next_id at next_timestep is disallowed)
  if ( next_timestep < static_cast<int>(cons->size()) ) {
    for ( list< pair<uint32_t, uint32_t> >::const_iterator it = cons->at(next_timestep).begin(); it != cons->at(next_timestep).end(); ++it ) {
      if ( (*it).second == -1 ) {
        if ( (*it).first == next_id ) {
          return true;
        }
      }
    }
  }

  // check edge constraints (the move from curr_id to next_id at next_timestep-1 is disallowed)
  if ( next_timestep > 0 && next_timestep - 1 < static_cast<int>(cons->size()) ) {
    for ( list< pair<uint32_t, uint32_t> >::const_iterator it = cons->at(next_timestep-1).begin(); it != cons->at(next_timestep-1).end(); ++it ) {
      if ( (*it).first == curr_id && (*it).second == next_id ) {
        return true;
      }
    }
  }

  return false;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int SingleAgentECBS::numOfConflictsForStep(
  uint32_t curr_id,
  uint32_t next_id,
  int next_timestep,
  const std::vector< std::unordered_set<vertex_t> >& resTable,
  const std::vector< std::set<edge_t> >& resTableEdges,
  int max_plan_len)
{
  int retVal = 0;
  if (next_timestep >= max_plan_len) {
    // check vertex constraints (being at an agent's goal when he stays there because he is done planning)
    if ( resTable[max_plan_len-1].find(next_id) != resTable[max_plan_len-1].end() ) {
      retVal++;
    }
    // Note -- there cannot be edge conflicts when other agents are done moving
  } else {
    // check vertex constraints (being in next_id at next_timestep is disallowed)
    if ( resTable[next_timestep].find(next_id) != resTable[next_timestep].end() ) {
      retVal++;
    }
    // check edge constraints (the move from curr_id to next_id at next_timestep-1 is disallowed)
    // which means that res_table is occupied with another agent for [curr_id,next_timestep] and [next_id,next_timestep-1]
    if (   resTable[next_timestep].find(curr_id) != resTable[next_timestep].end()
        && resTable[next_timestep-1].find(next_id) != resTable[next_timestep-1].end() ) {
      retVal++;
    }

    // check generalized edge conflicts
    auto e = boost::edge(curr_id, next_id, *m_searchGraph);
    if (resTableEdges[next_timestep-1].find(e.first) != resTableEdges[next_timestep-1].end()) {
      retVal++;
      // std::cout << "generalized edge conflict!" << std::endl;
    }

  }
  //  cout << "#CONF=" << retVal << " ; For: curr_id=" << curr_id << " , next_id=" << next_id << " , next_timestep=" << next_timestep
  //       << " , max_plan_len=" << max_plan_len << endl;
  return retVal;
}

// $$$ -- is there a more efficient way to do that?
void SingleAgentECBS::updateFocalList(
  double old_lower_bound,
  double new_lower_bound,
  double f_weight)
{
  //  cout << "Update Focal: (old_LB=" << old_lower_bound << " ; new_LB=" << new_lower_bound << endl;;
  for (Node* n : m_open_list) {
    //    cout << "   Considering " << n << " , " << *n << endl;
    if ( n->getFVal() > old_lower_bound &&
         n->getFVal() <= new_lower_bound ) {
      //      cout << "      Added (n->f-val=" << n->getFVal() << ")" << endl;
      n->focal_handle = m_focal_list.push(n);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// return true if a path found (and updates vector<int> path) or false if no path exists
bool SingleAgentECBS::findPath(
  double f_weight,
  const vector < list< pair<uint32_t, uint32_t> > >* constraints,
  const std::vector< std::unordered_set<vertex_t> >& resTable,
  const std::vector< std::set<edge_t> >& resTableEdges,
  size_t max_plan_len)
{
  // clear data structures if they had been used before
  // (note -- nodes are deleted before findPath returns)
  m_open_list.clear();
  m_focal_list.clear();
  m_allNodes_table.clear();
  m_num_expanded = 0;
  m_num_generated = 0;

  hashtable_t::iterator it;  // will be used for find()

  // generate start and add it to the OPEN list
  Node* start = new Node(m_start_location,
                         0, m_heuristic->getHValue(m_start_location), NULL, 0,
                         0, false, 0);
  m_num_generated++;
  start->open_handle = m_open_list.push(start);
  start->focal_handle = m_focal_list.push(start);
  start->in_openlist = true;
  m_allNodes_table[start] = start;
  m_min_f_val = start->getFVal();
  m_lower_bound = f_weight * m_min_f_val;

  int lastGoalConsTime = extractLastGoalTimestep(m_goal_location, constraints);

  std::vector<uint32_t> neighbors;
  neighbors.reserve(8);

  while ( !m_focal_list.empty() ) {
    //    cout << "|F|=" << focal_list.size() << " ; |O|=" << open_list.size() << endl;
    Node* curr = m_focal_list.top();
    m_focal_list.pop();
    //    cout << "Current FOCAL bound is " << lower_bound << endl;
    //    cout << "POPPED FOCAL's HEAD: (" << curr << ") " << (*curr) << endl;
    m_open_list.erase(curr->open_handle);
    //    cout << "DELETED" << endl; fflush(stdout);
    curr->in_openlist = false;
    m_num_expanded++;

    // check if the popped node is a goal
    if (   curr->loc == m_goal_location
        && curr->timestep > lastGoalConsTime) {
      updatePath(curr);
      releaseClosedListNodes(&m_allNodes_table);
      return true;
    }

    neighbors.clear();
    neighbors.push_back(curr->loc); // We can wait as well

    for (const auto& e : pair_range(boost::out_edges(curr->loc, *m_searchGraph))) {
      neighbors.push_back(boost::target(e, *m_searchGraph));
    }

    for (uint32_t next_loc : neighbors) {
      int next_timestep = curr->timestep + 1;
      if ( !isConstrained(curr->loc, next_loc, next_timestep, constraints) ) {
        // compute cost to next_id via curr node
        double cost = 1.0;//0.25;//0.1;
        if (curr->loc != next_loc) {
          auto e = boost::edge(curr->loc, next_loc, *m_searchGraph);
          cost = 1.0;//(*m_searchGraph)[e.first].length;
        }
        // if (action == MapLoader::WAIT_ACTION) {
        //   cost = 0.5;
        // }
        double next_g_val = curr->g_val + cost;
        double next_h_val = m_heuristic->getHValue(next_loc);
        int next_internal_conflicts = 0;
        if (max_plan_len > 0) { // check if the reservation table is not empty (that is tha max_length of any other agent's plan is > 0)
          next_internal_conflicts = curr->num_internal_conf + numOfConflictsForStep(curr->loc, next_loc, next_timestep, resTable, resTableEdges, max_plan_len);
        }
        // generate (maybe temporary) node
        Node* next = new Node (next_loc, next_g_val, next_h_val,
                               curr, next_timestep, next_internal_conflicts, false,
                               0);
        //        cout << "   NEXT(" << next << ")=" << *next << endl;
        // try to retrieve it from the hash table
        it = m_allNodes_table.find(next);

        if ( it == m_allNodes_table.end() ) {  // add the newly generated node to open_list and hash table
          //          cout << "   ADDING it as new." << endl;
          next->open_handle = m_open_list.push(next);
          next->in_openlist = true;
          m_num_generated++;
          if (next->getFVal() <= m_lower_bound) {
            next->focal_handle = m_focal_list.push(next);
          }
          m_allNodes_table[next] = next;
        } else {  // update existing node's if needed (only in the open_list)
          delete(next);  // not needed anymore -- we already generated it before
          Node* existing_next = (*it).second;
          //          cout << "Actually next exists. It's address is " << existing_next << endl;
          if (existing_next->in_openlist) {  // if its in the open list
            if ( existing_next->getFVal() > next_g_val + next_h_val ||
                 (existing_next->getFVal() == next_g_val + next_h_val && existing_next->num_internal_conf > next_internal_conflicts) ) {
              // if f-val decreased through this new path (or it remains the same and there's less internal conflicts)
              //              cout << "   UPDATE its f-val in OPEN (decreased or less #conflicts)" << endl;
              //              cout << "   Node state before update: " << *existing_next;
              bool add_to_focal = false;  // check if it was above the focal bound before and now below (thus need to be inserted)
              bool update_in_focal = false;  // check if it was inside the focal and needs to be updated (because f-val changed)
              bool update_open = false;
              if ( (next_g_val + next_h_val) <= m_lower_bound ) {  // if the new f-val qualify to be in FOCAL
                if ( existing_next->getFVal() > m_lower_bound ) {
                  add_to_focal = true;  // and the previous f-val did not qualify to be in FOCAL then add
                } else {
                  update_in_focal = true;  // and the previous f-val did qualify to be in FOCAL then update
                }
              }
              if ( existing_next->getFVal() > next_g_val + next_h_val ) {
                update_open = true;
              }
              // update existing node
              existing_next->g_val = next_g_val;
              existing_next->h_val = next_h_val;
              existing_next->parent = curr;
              existing_next->num_internal_conf = next_internal_conflicts;
              existing_next->num_non_hwy_edges = 0;
              //              cout << "   Node state after update: " << *existing_next;
              if ( update_open ) {
                m_open_list.increase(existing_next->open_handle);  // increase because f-val improved
                //                cout << "     Increased in OPEN" << endl;
              }
              if (add_to_focal) {
                existing_next->focal_handle = m_focal_list.push(existing_next);
                //                cout << "     Inserted to FOCAL" << endl;
              }
              if (update_in_focal) {
                m_focal_list.update(existing_next->focal_handle);  // should we do update? yes, because number of conflicts may go up or down
                //                cout << "     Updated in FOCAL" << endl;
              }
            }
            //            cout << "   Do NOT update in OPEN (f-val for this node increased or stayed the same and has more conflicts)" << endl;
          } else {  // if its in the closed list (reopen)
            if ( existing_next->getFVal() > next_g_val + next_h_val ||
                 (existing_next->getFVal() == next_g_val + next_h_val && existing_next->num_internal_conf > next_internal_conflicts) ) {
              // if f-val decreased through this new path (or it remains the same and there's less internal conflicts)
              //              cout << "   Reinsert it to OPEN" << endl;
              //              cout << "   Node state before update: " << *existing_next;
              existing_next->g_val = next_g_val;
              existing_next->h_val = next_h_val;
              existing_next->parent = curr;
              existing_next->num_internal_conf = next_internal_conflicts;
              existing_next->num_non_hwy_edges = 0;
              existing_next->open_handle = m_open_list.push(existing_next);
              existing_next->in_openlist = true;
              //              cout << "   Node state after update: " << *existing_next;
              if ( existing_next->getFVal() <= m_lower_bound ) {
                existing_next->focal_handle = m_focal_list.push(existing_next);
                //                cout << "     Inserted to FOCAL" << endl;
              }
            }
            //            cout << "   Do NOT reopen" << endl;
          }  // end update a node in closed list
        }  // end update an existing node
      }  // end if case for grid not blocked
    }  // end for loop that generates successors
    // update FOCAL if min f-val increased
    if (m_open_list.size() == 0) { // in case OPEN is empty, no path found...
      return false;
    }
    Node* open_head = m_open_list.top();
    if ( open_head->getFVal() > m_min_f_val ) {
      double new_min_f_val = open_head->getFVal();
      double new_lower_bound = f_weight * new_min_f_val;
      /*
        cout << "LL FOCAL UPDATE! Old-f-min=" << min_f_val << " ; Old-LB=" << lower_bound << endl;
        cout << "OPEN: ";
        for (Node* n : open_list)
        cout << n << " , ";
        cout << endl;
        cout << "FOCAL: ";
        for (Node* n : focal_list)
        cout << n << " , ";
        cout << endl;
      */
      updateFocalList(m_lower_bound, new_lower_bound, f_weight);
      m_min_f_val = new_min_f_val;
      m_lower_bound = new_lower_bound;
      /*
        cout << "   New-f-min=" << min_f_val << " ; New-LB=" << lower_bound << endl;
        cout << "FOCAL: ";
        for (Node* n : focal_list)
        cout << n << " , ";
        cout << endl;
      */
    }
  }  // end while loop
  // no path found
  m_path.clear();
  releaseClosedListNodes(&m_allNodes_table);
  return false;
}

