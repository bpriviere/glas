#include "node.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

using namespace boost;
using namespace std;

Node::Node()
  : loc(0)
  , g_val(0)
  , h_val(0)
  , parent(NULL)
  , timestep(0)
  , num_internal_conf(0)
  , in_openlist(false)
  , num_non_hwy_edges(0)
{
}

Node::Node(
  uint32_t loc,
  double g_val,
  double h_val,
  Node* parent,
  int timestep,
  int num_internal_conf,
  bool in_openlist,
  int num_non_hwy_edges)
  : loc(loc)
  , g_val(g_val)
  , h_val(h_val)
  , parent(parent)
  , timestep(timestep)
  , num_internal_conf(num_internal_conf)
  , in_openlist(in_openlist)
  , num_non_hwy_edges(num_non_hwy_edges)
{
}

Node::~Node() {
}

std::ostream& operator<<(std::ostream& os, const Node& n) {
  if ( n.parent != NULL )
    os << "LOC=" << n.loc << " ; TIMESTEP=" << n.timestep << " ; GVAL=" << n.g_val << " ; HVAL=" << std::setprecision(4) << n.h_val
       << " ; #CONF="<< n.num_internal_conf << " ; PARENT=" << (n.parent)->loc
       << " ; IN_OPEN?" << std::boolalpha << n.in_openlist;
  else
    os << "LOC=" << n.loc << " ; TIMESTEP=" << n.timestep << " ; GVAL=" << n.g_val << " ; HVAL=" << std::setprecision(4) << n.h_val
       << " ; #CONF="<< n.num_internal_conf << " ; ROOT (NO PARENT)";
  return os;
}
/*std::ostream& operator<<(std::ostream& os, const Node* n) {
  os << "LOC=" << n->loc << " ; TIMESTEP=" << n->timestep << " ; GVAL=" << n->g_val << " ; PARENT=" << (n->parent)->loc;
  return os;
  }*/
