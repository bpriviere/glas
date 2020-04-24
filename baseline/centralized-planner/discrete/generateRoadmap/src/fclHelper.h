#pragma once

// FCL Headers
#include <fcl/collision.h>

namespace fclHelper
{
  fcl::CollisionGeometry* createCollisionGeometryFromFile(
    const std::string& fileName);

  struct range {
    double min;
    double max;
  };

  std::array<range, 3> boundingBox(
    fcl::CollisionGeometry* geometry);
}
