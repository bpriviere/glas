#include "fclHelper.h"

#include <regex>

// Boost
#include <boost/filesystem.hpp>

// FCL Headers
#include <fcl/collision.h>
#include <fcl/collision_node.h>
#include <fcl/traversal/traversal_node_setup.h>
#include <fcl/continuous_collision.h>

// Assimp
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

// Octomap
#include <octomap/octomap.h>
#include <octomap/OcTree.h>

using namespace std;
using namespace fcl;

namespace fclHelper {

void extractVerticesAndTrianglesRecursive(
  const aiScene *scene,
  const aiNode *node,
  std::vector<Vec3f>& vertices,
  std::vector<Triangle>& triangles)
{
  for (size_t i = 0 ; i < node->mNumMeshes; ++i)
  {
    const aiMesh* a = scene->mMeshes[node->mMeshes[i]];
    size_t offset = vertices.size();

    for (size_t j = 0; j < a->mNumVertices; ++j) {
      vertices.push_back(
        Vec3f(
          a->mVertices[j].x,
          a->mVertices[j].y,
          a->mVertices[j].z));
    }

    for (size_t j = 0 ; j < a->mNumFaces ; ++j) {
      if (a->mFaces[j].mNumIndices == 3) {
        triangles.push_back(Triangle(
          a->mFaces[j].mIndices[0] + offset,
          a->mFaces[j].mIndices[1] + offset,
          a->mFaces[j].mIndices[2] + offset));
      }
    }
  }

  for (unsigned int n = 0; n < node->mNumChildren; ++n) {
    extractVerticesAndTrianglesRecursive(scene, node->mChildren[n], vertices, triangles);
  }
}

CollisionGeometry* createCollisionGeometryFromFile(const std::string& fileName)
{
  std::string floatRegex("[+-]?(\\d*\\.)?\\d+");
  std::regex regexSphere("^<Sphere:(" + floatRegex + ")>$");
  std::regex regexCylinder("^<Cylinder:(" + floatRegex + ")\\,(" + floatRegex + ")>$");

  std::smatch sm;

  std::regex_match(fileName, sm, regexSphere);
  if (sm.size()) {
    double radius = std::stod(sm[1]);
    return new Sphere(radius);
  }

  std::regex_match(fileName, sm, regexCylinder);
  if (sm.size()) {
    // std::cout << sm[0] << "|" << sm[1] << "|" << sm[3] << std::endl;
    double radius = std::stod(sm[1]);
    double lz = std::stod(sm[3]);
    std::cout << "Cylinder with r=" << radius << " lz=" << lz << std::endl;
    return new Cylinder(radius, lz);
  }

  std::string extension = boost::filesystem::extension(fileName);

  if (extension == ".bt") {
    // Assume it is an OcTree
    auto rawtree = new octomap::OcTree(fileName);
    const auto model = new OcTree(std::shared_ptr<const octomap::OcTree>(rawtree));
    model->setUserData(rawtree);
    model->computeLocalAABB();
    return model;
  } else {
    // Assume it is a file supported by Assimp
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(fileName.c_str(),
      aiProcess_Triangulate            |
      aiProcess_JoinIdenticalVertices  |
      aiProcess_SortByPType            |
      aiProcess_OptimizeGraph          |
      aiProcess_OptimizeMeshes);

    std::vector<Vec3f> vertices;
    std::vector<Triangle> triangles;

    // code to set the vertices and triangles
    if (scene && scene->mRootNode) {
      extractVerticesAndTrianglesRecursive(scene, scene->mRootNode, vertices, triangles);

      // add the mesh data into the BVHModel structure
      const auto model = new BVHModel<OBBRSS>();
      model->beginModel();
      model->addSubModel(vertices, triangles);
      model->endModel();
      model->computeLocalAABB();

      return model;
    }
  }

  return nullptr;
}

std::array<range, 3> boundingBox(
  fcl::CollisionGeometry* geometry)
{
  std::array<range, 3> result;
  // for (size_t i = 0; i < 3; ++i) {
  //   result[i].min = geometry->aabb_local.min_[i];
  //   result[i].max = geometry->aabb_local.max_[i];
  // }
  for (auto& r : result) {
    r.min = std::numeric_limits<float>::max();
    r.max = std::numeric_limits<float>::min();
  }

  switch (geometry->getObjectType())
  {
  case fcl::OT_BVH:
    {
      BVHModel<OBBRSS>* model = dynamic_cast<BVHModel<OBBRSS>*>(geometry);
      for (int i = 0; i < model->num_vertices; ++i) {
        const Vec3f& v = model->vertices[i];
        for (size_t d = 0; d < 3; ++d) {
          result[d].min = std::min(result[d].min, v[d]);
          result[d].max = std::max(result[d].max, v[d]);
        }
      }
    }
    break;
  case OT_OCTREE:
    {
      // OcTree* model = dynamic_cast<OcTree*>(geometry);
      // AABB aabb = model->getRootBV();
      // for (size_t i = 0; i < 3; ++i) {
      //   result[i].min = aabb.min_[i];
      //   result[i].max = aabb.max_[i];
      // }
      octomap::OcTree* tree = (octomap::OcTree*)geometry->getUserData();
      tree->getMetricMin(result[0].min, result[1].min, result[2].min);
      tree->getMetricMax(result[0].max, result[1].max, result[2].max);
    }
    break;
  case OT_GEOM:
    {
      geometry->computeLocalAABB();
      for (size_t i = 0; i < 3; ++i) {
        result[i].min = geometry->aabb_local.min_[i];
        result[i].max = geometry->aabb_local.max_[i];
      }
    }
    break;
  default:
    throw std::runtime_error("Unsupported CollisionGeometry type!");
  }
  return result;
}



} // namespace fclHelper
