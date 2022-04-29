

#include "core/Application.h"

#include <cstring>
#include <iostream>
#include <sstream>

// A simple unit cube built from 12 triangles.
OptixTraversableHandle Application::createBox()
{
  float left   = -1.0f;
  float right  =  1.0f;
  float bottom = -1.0f;
  float top    =  1.0f;
  float back   = -1.0f;
  float front  =  1.0f;

  std::vector<VertexAttributes> attributes;

  VertexAttributes attrib;

  // Left.
  attrib.tangent   = make_float3(0.0f, 0.0f, 1.0f);
  attrib.normal    = make_float3(-1.0f, 0.0f, 0.0f);

  attrib.vertex    = make_float3(left, bottom, back);
  attrib.texcoord  = make_float3(0.0f, 0.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(left, bottom, front);
  attrib.texcoord  = make_float3(1.0f, 0.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(left, top, front);
  attrib.texcoord  = make_float3(1.0f, 1.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(left, top, back);
  attrib.texcoord  = make_float3(0.0f, 1.0f, 0.0f);
  attributes.push_back(attrib);

  // Right.
  attrib.tangent   = make_float3(0.0f, 0.0f, -1.0f);
  attrib.normal    = make_float3(1.0f, 0.0f,  0.0f);

  attrib.vertex    = make_float3(right, bottom, front);
  attrib.texcoord  = make_float3(0.0f, 0.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(right, bottom, back);
  attrib.texcoord  = make_float3(1.0f, 0.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(right, top, back);
  attrib.texcoord  = make_float3(1.0f, 1.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(right, top, front);
  attrib.texcoord  = make_float3(0.0f, 1.0f, 0.0f);
  attributes.push_back(attrib);

  // Back.
  attrib.tangent   = make_float3(-1.0f, 0.0f, 0.0f);
  attrib.normal    = make_float3(0.0f, 0.0f, -1.0f);

  attrib.vertex    = make_float3(right, bottom, back);
  attrib.texcoord  = make_float3(0.0f, 0.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(left, bottom, back);
  attrib.texcoord  = make_float3(1.0f, 0.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(left, top, back);
  attrib.texcoord  = make_float3(1.0f, 1.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(right, top, back);
  attrib.texcoord  = make_float3(0.0f, 1.0f, 0.0f);
  attributes.push_back(attrib);

  // Front.
  attrib.tangent   = make_float3(1.0f, 0.0f,  0.0f);
  attrib.normal    = make_float3(0.0f, 0.0f, 1.0f);

  attrib.vertex    = make_float3(left, bottom, front);
  attrib.texcoord  = make_float3(0.0f, 0.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(right, bottom, front);
  attrib.texcoord  = make_float3(1.0f, 0.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(right, top, front);
  attrib.texcoord  = make_float3(1.0f, 1.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(left, top, front);
  attrib.texcoord  = make_float3(0.0f, 1.0f, 0.0f);
  attributes.push_back(attrib);

  // Bottom.
  attrib.tangent   = make_float3(1.0f, 0.0f,  0.0f);
  attrib.normal    = make_float3(0.0f, -1.0f, 0.0f);

  attrib.vertex    = make_float3(left, bottom, back);
  attrib.texcoord  = make_float3(0.0f, 0.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(right, bottom, back);
  attrib.texcoord  = make_float3(1.0f, 0.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(right, bottom, front);
  attrib.texcoord  = make_float3(1.0f, 1.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(left, bottom, front);
  attrib.texcoord  = make_float3(0.0f, 1.0f, 0.0f);
  attributes.push_back(attrib);

  // Top.
  attrib.tangent   = make_float3(1.0f, 0.0f,  0.0f);
  attrib.normal    = make_float3( 0.0f, 1.0f, 0.0f);

  attrib.vertex    = make_float3(left, top, front);
  attrib.texcoord  = make_float3(0.0f, 0.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(right, top, front);
  attrib.texcoord  = make_float3(1.0f, 0.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(right, top, back);
  attrib.texcoord  = make_float3(1.0f, 1.0f, 0.0f);
  attributes.push_back(attrib);

  attrib.vertex    = make_float3(left, top, back);
  attrib.texcoord  = make_float3(0.0f, 1.0f, 0.0f);
  attributes.push_back(attrib);


  std::vector<unsigned int> indices;

  for (unsigned int i = 0; i < 6; ++i)
  {
    const unsigned int idx = i * 4; // Four attributes per box face.

    indices.push_back(idx    );
    indices.push_back(idx + 1);
    indices.push_back(idx + 2);

    indices.push_back(idx + 2);
    indices.push_back(idx + 3);
    indices.push_back(idx    );
  }

  std::cout << "createBox(): Vertices = " << attributes.size() <<  ", Triangles = " << indices.size() / 3 << '\n';

  return createGeometry(attributes, indices);
}
