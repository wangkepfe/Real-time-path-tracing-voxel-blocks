

#include "core/Application.h"

#include <cstring>
#include <iostream>
#include <sstream>

#include "shaders/shader_common.h"

// Parallelogram from footpoint position, spanned by unnormalized vectors vecU and vecV, normal is normalized and on the CCW frontface.
OptixTraversableHandle Application::createParallelogram(float3 const& position, float3 const& vecU, float3 const& vecV, float3 const& normal)
{
  std::vector<VertexAttributes> attributes;

  VertexAttributes attrib;

  // Same for all four vertices in this parallelogram.
  attrib.tangent   = normalize(vecU);
  attrib.normal    = normal;

  attrib.vertex    = position; // left bottom
  attrib.texcoord  = make_float3(0.0f, 0.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = position + vecU; // right bottom
  attrib.texcoord  = make_float3(1.0f, 0.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = position + vecU + vecV; // right top
  attrib.texcoord  = make_float3(1.0f, 1.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = position + vecV; // left top
  attrib.texcoord  = make_float3(0.0f, 1.0f, 0.0f);
  attributes.push_back(attrib);

  std::vector<unsigned int> indices;

  indices.push_back(0);
  indices.push_back(1);
  indices.push_back(2);

  indices.push_back(2);
  indices.push_back(3);
  indices.push_back(0);

  std::cout << "createParallelogram(): Vertices = " << attributes.size() <<  ", Triangles = " << indices.size() / 3 << '\n';

  return createGeometry(attributes, indices);
}
