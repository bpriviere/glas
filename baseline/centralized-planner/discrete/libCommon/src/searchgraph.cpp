#include <iostream>
#include <fstream>

#include "libCommon/searchgraph.h"

// Yaml
#include "yaml-cpp/yaml.h"


position_t nodeAsPos(const YAML::Node& node)
{
  return position_t(
      node[0].as<float>(),
      node[1].as<float>(),
      node[2].as<float>());
}

void loadSearchGraph(
  searchGraph_t& searchGraph,
  std::unordered_map<std::string, vertex_t>& vNameToV,
  std::unordered_map<std::string, edge_t>& eNameToE,
  const std::string& fileName)
{
  YAML::Node config = YAML::LoadFile(fileName);
  config = config["annotatedRoadmap"];
  // std::unordered_map<std::string, vertex_t> vNameToV;
  // std::unordered_map<std::string, edge_t> eNameToE;

  // add vertices
  const auto& vertices = config["vertices"];
  if (vertices) {
    for (const auto& node : vertices) {
      position_t pos = nodeAsPos(node["pos"]);
      std::string name = node["name"].as<std::string>();
      auto v = boost::add_vertex(searchGraph);
      searchGraph[v].name = name;
      searchGraph[v].pos = pos;
      vNameToV[name] = v;
    }

    // add generalized vertex conflicts
    for (const auto& node : vertices) {
      std::string name = node["name"].as<std::string>();
      auto iter = vNameToV.find(name);
      if (iter == vNameToV.end()) {
        std::cerr << "(1) ERROR: Could not find vertex " << name << std::endl;
        continue;
      }
      vertex_t v = iter->second;

      const auto& vertexConflicts = node["conflicts"][0]["vertexConflicts"];
      if (vertexConflicts) {
        for (const auto& cNode : vertexConflicts) {
          std::string cName = cNode.as<std::string>();
          auto cIter = vNameToV.find(cName);
          if (cIter == vNameToV.end()) {
            std::cerr << "(2) ERROR: Could not find vertex " << cName << std::endl;
            continue;
          }
          // std::cout << cName << std::endl;
          vertex_t u = cIter->second;
          searchGraph[v].generalizedVertexConflicts.insert(u);
        }
      }
    }
  }

  // add edges
  const auto& edges = config["edges"];
  if (edges) {
    for (const auto& node : edges) {
      std::string name = node["name"].as<std::string>();
      std::string fromName = node["from"].as<std::string>();
      auto fromIter = vNameToV.find(fromName);

      std::string toName = node["to"].as<std::string>();
      auto toIter = vNameToV.find(toName);
      if (    fromIter == vNameToV.end()
           || toIter == vNameToV.end()
           || fromIter->second == toIter->second) {
        std::cerr << "invalid edge! " << node << std::endl;
        continue;
      }
      auto e = boost::add_edge(fromIter->second, toIter->second, searchGraph);
      searchGraph[e.first].name = name;
      searchGraph[e.first].length = (searchGraph[fromIter->second].pos - searchGraph[toIter->second].pos).norm();
      searchGraph[e.first].isHighway = false;

      eNameToE[name] = e.first;
    }

    // add generalized edge conflicts
    for (const auto& node : edges) {
      std::string name = node["name"].as<std::string>();
      auto iter = eNameToE.find(name);
      if (iter == eNameToE.end()) {
        std::cerr << "(1) ERROR: Could not find edge " << name << std::endl;
        continue;
      }
      edge_t e = iter->second;

      const auto& edgeConflicts = node["conflicts"][0]["edgeConflicts"];
      if (edgeConflicts) {
        for (const auto& cNode : edgeConflicts) {
          std::string cName = cNode.as<std::string>();
          auto cIter = eNameToE.find(cName);
          if (cIter == eNameToE.end()) {
            std::cerr << "(1) ERROR: Could not find edge " << cName << std::endl;
            continue;
          }
          edge_t u = cIter->second;
          searchGraph[e].generalizedEdgeConflicts.insert(u);

          searchGraph[u].generalizedEdgeConflicts.insert(e);
        }
      }

      const auto& vertexConflicts = node["conflicts"][0]["vertexConflicts"];
      if (vertexConflicts) {
        for (const auto& cNode : vertexConflicts) {
          std::string cName = cNode.as<std::string>();
          auto cIter = vNameToV.find(cName);
          if (cIter == vNameToV.end()) {
            std::cerr << "(1) ERROR: Could not find vertex " << cName << std::endl;
            continue;
          }
          vertex_t u = cIter->second;
          searchGraph[e].generalizedEdgeVertexConflicts.insert(u);

          searchGraph[u].generalizedVertexEdgeConflicts.insert(e);
        }
      }
    }
  }

  if (vertices) {
    for (const auto& node : vertices) {
      std::string name = node["name"].as<std::string>();
      auto iter = vNameToV.find(name);
      if (iter == vNameToV.end()) {
        std::cerr << "(1) ERROR: Could not find vertex " << name << std::endl;
        continue;
      }
      vertex_t v = iter->second;


      const auto& edgeConflicts = node["conflicts"][0]["edgeConflicts"];
      if (edgeConflicts) {
        for (const auto& cNode : edgeConflicts) {
          std::string cName = cNode.as<std::string>();
          auto cIter = eNameToE.find(cName);
          if (cIter == eNameToE.end()) {
            std::cerr << "(2) ERROR: Could not find edge " << cName << std::endl;
            continue;
          }
          // std::cout << cName << std::endl;
          edge_t u = cIter->second;
          searchGraph[v].generalizedVertexEdgeConflicts.insert(u);

          // our roadmap does not include vertex conflicts at edges; add them here
          searchGraph[u].generalizedEdgeVertexConflicts.insert(v);
        }
      }
    }
  }

  std::cout << "|V| " << num_vertices(searchGraph) << std::endl;
  std::cout << "|E| " << num_edges(searchGraph) << std::endl;
  size_t vvconflicts = 0;
  size_t veconflicts = 0;
  for (auto vp = boost::vertices(searchGraph); vp.first != vp.second; ++vp.first) {
    vertex_t v = *vp.first;
    vvconflicts += searchGraph[v].generalizedVertexConflicts.size();
    veconflicts += searchGraph[v].generalizedVertexEdgeConflicts.size();
  }
  size_t evconflicts = 0;
  size_t eeconflicts = 0;
  for (auto ep = boost::edges(searchGraph); ep.first != ep.second; ++ep.first) {
    edge_t e = *ep.first;
    evconflicts += searchGraph[e].generalizedEdgeVertexConflicts.size();
    eeconflicts += searchGraph[e].generalizedEdgeConflicts.size();
  }
  std::cout << "avg. Cvv " << vvconflicts / (double)num_vertices(searchGraph) << std::endl;
  std::cout << "avg. Cve " << veconflicts / (double)num_vertices(searchGraph) << std::endl;
  std::cout << "avg. Cev " << evconflicts / (double)num_edges(searchGraph) << std::endl;
  std::cout << "avg. Cee " << eeconflicts / (double)num_edges(searchGraph) << std::endl;
}

void saveSearchGraph(
  const searchGraph_t& searchGraph,
  const std::string& fileName)
{
  YAML::Emitter out;
  out.SetIndent(2);

  out << YAML::BeginMap;

  // vertices
  out << YAML::Key << "vertices";
  out << YAML::Value << YAML::BeginSeq;

  for (const auto& v : pair_range(vertices(searchGraph))) {
    // out << YAML::Anchor("v" + std::to_string(i));
    out << YAML::BeginMap;
    out << YAML::Key << "name";
    out << YAML::Value << searchGraph[v].name;
    out << YAML::Key << "pos";
    const position_t& pos = searchGraph[v].pos;
    out << YAML::Value << YAML::Flow << YAML::BeginSeq << pos.x() << pos.y() << pos.z() << YAML::EndSeq;
    if (searchGraph[v].generalizedVertexConflicts.size()) {
      out << YAML::Key << "conflicts";
      out << YAML::Value << YAML::Flow << YAML::BeginSeq;
      for (vertex_t c : searchGraph[v].generalizedVertexConflicts) {
        //out << c;
        // out << YAML::Alias("v" + std::to_string(c));
        out << searchGraph[c].name;
      }
      out << YAML::EndSeq;
    }
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;

  // edges
  out << YAML::Key << "edges";
  out << YAML::Value << YAML::BeginSeq;

  // for (size_t i = 0; i < edges.size(); ++i) {
  for (const auto& e : pair_range(edges(searchGraph))) {
    // out << YAML::Anchor("e" + std::to_string(i));
    out << YAML::BeginMap;
    out << YAML::Key << "name";
    out << YAML::Value << searchGraph[e].name;
    out << YAML::Key << "from";
    // out << YAML::Value << YAML::Alias("v" + std::to_string(edges[i].from));//edges[i].from;
    out << YAML::Value << searchGraph[source(e, searchGraph)].name;//edges[i].from;
    out << YAML::Key << "to";
    // out << YAML::Value << YAML::Alias("v" + std::to_string(edges[i].to));//edges[i].to;
    out << YAML::Value << searchGraph[target(e, searchGraph)].name;//edges[i].to;
    if (searchGraph[e].generalizedEdgeConflicts.size()) {
      out << YAML::Key << "conflicts";
      out << YAML::Value << YAML::Flow << YAML::BeginSeq;
      for (edge_t c : searchGraph[e].generalizedEdgeConflicts) {
        // out << YAML::Alias("e" + std::to_string(c));
        out << searchGraph[c].name;
        // out << c;
      }
      out << YAML::EndSeq;
    }
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;


  out << YAML::EndMap;

  std::ofstream ofstr(fileName.c_str());
  ofstr << out.c_str();
}
