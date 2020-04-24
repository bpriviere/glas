#pragma once

#include "libCommon/searchgraph.h"

class NanoFlannSearchgraphAdaptor
{
public:
  NanoFlannSearchgraphAdaptor(
    const searchGraph_t& searchgraph)
    : m_searchgraph(searchgraph)
  {
  }

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const
  {
    return num_vertices(m_searchgraph);
  }

  // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
  inline float kdtree_distance(const float *p1, const size_t idx_p2,size_t /*size*/) const
  {
    const float d0=p1[0]-m_searchgraph[idx_p2].pos.x();
    const float d1=p1[1]-m_searchgraph[idx_p2].pos.y();
    const float d2=p1[2]-m_searchgraph[idx_p2].pos.z();
    return d0*d0+d1*d1+d2*d2;
  }

  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate value, the
  //  "if/else's" are actually solved at compile time.
  inline float kdtree_get_pt(const size_t idx, int dim) const
  {
    if (dim==0) return m_searchgraph[idx].pos.x();
    else if (dim==1) return m_searchgraph[idx].pos.y();
    else return m_searchgraph[idx].pos.z();
  }

  // Optional bounding-box computation: return false to default to a standard bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
  //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

private:
  const searchGraph_t& m_searchgraph;
};
