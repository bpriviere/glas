#include <iostream>
#include <iomanip>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <limits>

// Eigen
#include <Eigen/Core>

// Yaml
#include "yaml-cpp/yaml.h"

// conflict cylinder / swept hull generator
#include "ConflictCylinder.h"

#define SMALL_NUM 0.000001
#define ABS(x) ((x) >= 0 ? (x) : -(x))   //  absolute value
#define MAX(x,y) ((x) > y ? (x) : (y))   //  max
#define MIN(x,y) ((x) < y ? (x) : (y))   //  min

typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> vec3f_t;

struct AgentType{
  //Agent type name
  std::string type;

  //Subgraph file name
  std::string subgraphYaml;

  //Subgraph Range for agent type
  size_t vStart;
  size_t vEnd;
  size_t eStart;
  size_t eEnd;

  //Conflict definition for static objects
  //  unused in annotation code, subgraph version has this implicitly
  // ConflictCylinder envSize;
};

struct AnnotatedVertex {
  AnnotatedVertex(
    const std::string& name,
    const vec3f_t& pos)
    : name(name)
    , pos(pos)
  {
  }
  std::string name;
  vec3f_t pos;
  std::vector<std::unordered_set<size_t> > edgeCollisions;
  std::vector<std::unordered_set<size_t> > vertexCollisions;
  std::set<std::string> typeRestrictions;
};

struct AnnotatedEdge {
  AnnotatedEdge(
    const std::string& name,
    size_t from,
    size_t to)
    : name(name)
    , from(from)
    , to(to)
  {
  }

  std::string name;
  size_t from;
  size_t to;
  std::vector<std::unordered_set<size_t> > edgeCollisions;
  std::vector<std::unordered_set<size_t> > vertexCollisions;
  std::set<std::string> typeRestrictions;
};

struct AnnotatedRoadmap {
  std::vector<AnnotatedVertex> vertices;
  std::vector<AnnotatedEdge> edges;
  std::unordered_map<std::string, size_t> vNameToIdx;

  //Appends a subgraph to the graph. Adds agent type suffix to vertex/edge names
  bool addSubgraph(AgentType& agentType){

    YAML::Node yamlGraph = YAML::LoadFile(agentType.subgraphYaml);

    //subgraph range for agent type
    agentType.vStart = vertices.size();
    agentType.eStart = edges.size();

    for (const auto& node : yamlGraph["vertices"]) {
      const auto& pos = node["pos"];
      std::string name = node["name"].as<std::string>() + "_" + agentType.type;
      vec3f_t p(
        pos[0].as<float>(),
        pos[1].as<float>(),
        pos[2].as<float>());
      vertices.push_back(AnnotatedVertex(name, p));
      vNameToIdx[name] = vertices.size() - 1;
    }

    for (const auto& node : yamlGraph["edges"]) {
      std::string name = node["name"].as<std::string>() + "_" + agentType.type;
      std::string fromName = node["from"].as<std::string>() + "_" + agentType.type;
      std::string toName = node["to"].as<std::string>() + "_" + agentType.type;
      size_t from = vNameToIdx[fromName];
      size_t to = vNameToIdx[toName];
      if (    from >= vertices.size()
           || to >= vertices.size()
           || from == to) {
        std::cerr << "invalid edge! " << node << std::endl;
        continue;
      }
      edges.push_back(AnnotatedEdge(name, from, to));
    }

    //Subgraph range end
    agentType.vEnd = vertices.size();
    agentType.eEnd = edges.size();

    return true;
  }

  bool saveYaml(const std::string& outFile, const std::vector<AgentType>& agentTypes){
    YAML::Emitter out;
    out.SetIndent(2);

    out << YAML::BeginMap;//Begin file

    //Output agent types in this annotated graph file
    //  this is legacy just to support the way ECBS code loads conflicts
    out << YAML::Key << "agentTypes";
    out << YAML::Value << YAML::BeginSeq; //begin types sequence
    for (auto& ag : agentTypes){
      out << YAML::BeginMap; //begin type map
      out << YAML::Key << "type" << YAML::Value << ag.type;
      // out << YAML::Key << "conflictSize";
      // out << YAML::Value << YAML::Flow << YAML::BeginSeq << ag.conflictSize.x() << ag.conflictSize.y() << ag.conflictSize.z() << YAML::EndSeq;
      // out << YAML::Key << "obstacleSize";
      // out << YAML::Value << YAML::Flow << YAML::BeginSeq << ag.obstacleSize.x() << ag.obstacleSize.y() << ag.obstacleSize.z() << YAML::EndSeq;
      out << YAML::EndMap; //end type map
    }
    out << YAML::EndSeq; //end agentTypes sequence


    //Output roadmap with conflicts and restrictions
    out << YAML::Key << "annotatedRoadmap";
    out << YAML::BeginMap; //begin annotation map

    out << YAML::Key << "vertices";
    out << YAML::Value << YAML::BeginSeq; //begin vertex annotations
    for (size_t i = 0; i < vertices.size(); ++i) {
      out << YAML::BeginMap; //begin vertex map

      out << YAML::Key << "name" << YAML::Value << vertices[i].name;
      out << YAML::Key << "pos" << YAML::Value << YAML::Flow << YAML::BeginSeq << vertices[i].pos.x() << vertices[i].pos.y() << vertices[i].pos.z() << YAML::EndSeq;

      out << YAML::Key << "typeRestrictions";
      out << YAML::Value << YAML::Flow << YAML::BeginSeq; // begin type restrictions
      for (auto& typ : vertices[i].typeRestrictions){
        out << typ;
      }
      out << YAML::EndSeq; //end type restrictions

      //for each agent pair, output vertex and edge conflicts for this vertex
      //  Also legacy way to save conflicts, all conflicts are now in first 'type-pair' map
      out << YAML::Key << "conflicts";
      out << YAML::Value << YAML::BeginSeq; //begin type pair sequence
      //int pairId = 0;
      //for (size_t j = 0; j < agentTypes.size(); ++j){
      //  for (size_t k = j; k < agentTypes.size(); ++k){

          out << YAML::BeginMap; //begin conflict map for this pair
          out << YAML::Key << "typePair" << YAML::Value << YAML::Flow << YAML::BeginSeq << "all" << "all"<< YAML::EndSeq;

          out << YAML::Key << "vertexConflicts";
          out << YAML::Value << YAML::Flow << YAML::BeginSeq; //begin v-v conflicts
          for (size_t c : vertices[i].vertexCollisions[0]){
            out << vertices[c].name;
          }
          out << YAML::EndSeq; //end v-v conflicts

          out << YAML::Key << "edgeConflicts";
          out << YAML::Value << YAML::Flow << YAML::BeginSeq; //begin v-e conflicts
          for (size_t c : vertices[i].edgeCollisions[0]){
            out << edges[c].name;
          }
          out << YAML::EndSeq; //end v-e conflicts

          out << YAML::EndMap; //end conflict map for this pair
          //pairId++;
      //  }
      //}
      out << YAML::EndSeq; //end pair sequence

      out << YAML::EndMap; //end vertex map

    }
    out << YAML::EndSeq; // end vertex sequence

    out << YAML::Key << "edges";
    out << YAML::Value << YAML::BeginSeq; //begin edge annotations
    for (size_t i = 0; i < edges.size(); ++i) {
      out << YAML::BeginMap; //begin edge map

      out << YAML::Key << "name" << YAML::Value << edges[i].name;
      out << YAML::Key << "from" << YAML::Value << vertices[edges[i].from].name;
      out << YAML::Key << "to" << YAML::Value << vertices[edges[i].to].name;

      out << YAML::Key << "typeRestrictions";
      out << YAML::Value << YAML::Flow << YAML::BeginSeq; // begin type restrictions
      for (auto& typ : edges[i].typeRestrictions){
        out << typ;
      }
      out << YAML::EndSeq; //end type restrictions

      //for each agent pair, output vertex and edge conflicts for this vertex
      out << YAML::Key << "conflicts";
      out << YAML::Value << YAML::BeginSeq; //begin type pair sequence
      //int pairId = 0;
      //for (size_t j = 0; j < agentTypes.size(); ++j){
      //  for (size_t k = j; k < agentTypes.size(); ++k){
          out << YAML::BeginMap; //begin conflict map for this pair
          out << YAML::Key << "typePair" << YAML::Value << YAML::Flow << YAML::BeginSeq << "all" << "all" << YAML::EndSeq;

          out << YAML::Key << "vertexConflicts";
          out << YAML::Value << YAML::Flow << YAML::BeginSeq; //begin e-v conflicts
          for (size_t c : edges[i].vertexCollisions[0]){
            out << vertices[c].name;
          }
          out << YAML::EndSeq; //end v-v conflicts

          out << YAML::Key << "edgeConflicts";
          out << YAML::Value << YAML::Flow << YAML::BeginSeq; //begin e-e conflicts
          for (size_t c : edges[i].edgeCollisions[0]){
            out << edges[c].name;
          }
          out << YAML::EndSeq; //end v-e conflicts

          out << YAML::EndMap; //end conflict map for this pair
      //  }
        //pairId++;
      //}
      out << YAML::EndSeq; //end pair sequence

      out << YAML::EndMap; //end edge map

    }
    out << YAML::EndSeq; //end edge sequence

    out << YAML::EndMap; //end annotation map

    out << YAML::EndMap; //end file

    //Save output
    std::ofstream ofstr(outFile.c_str());
    ofstr << out.c_str();
    return true;
  }

};
