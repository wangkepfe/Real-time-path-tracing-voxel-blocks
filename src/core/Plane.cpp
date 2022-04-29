

#include "core/Application.h"

#include <cstring>
#include <iostream>
#include <sstream>

#include "shaders/vector_math.h"

#include "core/MyAssert.h"

OptixTraversableHandle Application::createPlane(const unsigned int tessU, const unsigned int tessV, const unsigned int upAxis)
{
  MY_ASSERT(1 <= tessU && 1 <= tessV);

  const float uTile = 2.0f / float(tessU);
  const float vTile = 2.0f / float(tessV);

  float3 corner;

  std::vector<VertexAttributes> attributes;

  VertexAttributes attrib;

  switch (upAxis)
  {
    case 0: // Positive x-axis is the geometry normal, create geometry on the yz-plane.
      corner = make_float3(0.0f, -1.0f, 1.0f); // Lower front corner of the plane. texcoord (0.0f, 0.0f).

      attrib.tangent = make_float3(0.0f, 0.0f, -1.0f);
      attrib.normal  = make_float3(1.0f, 0.0f,  0.0f);

      for (unsigned int j = 0; j <= tessV; ++j)
      {
        const float v = float(j) * vTile;

        for (unsigned int i = 0; i <= tessU; ++i)
        {
          const float u = float(i) * uTile;

          attrib.vertex   = corner + make_float3(0.0f, v, -u);
          attrib.texcoord = make_float3(u * 0.5f, v * 0.5f, 0.0f);

          attributes.push_back(attrib);
        }
      }
      break;

    case 1: // Positive y-axis is the geometry normal, create geometry on the xz-plane.
      corner = make_float3(-1.0f, 0.0f, 1.0f); // left front corner of the plane. texcoord (0.0f, 0.0f).

      attrib.tangent = make_float3(1.0f, 0.0f, 0.0f);
      attrib.normal  = make_float3(0.0f, 1.0f, 0.0f);

      for (unsigned int j = 0; j <= tessV; ++j)
      {
        const float v = float(j) * vTile;

        for (unsigned int i = 0; i <= tessU; ++i)
        {
          const float u = float(i) * uTile;

          attrib.vertex   = corner + make_float3(u, 0.0f, -v);
          attrib.texcoord = make_float3(u * 0.5f, v * 0.5f, 0.0f);

          attributes.push_back(attrib);
        }
      }
      break;

    case 2: // Positive z-axis is the geometry normal, create geometry on the xy-plane.
      corner = make_float3(-1.0f, -1.0f, 0.0f); // Lower left corner of the plane. texcoord (0.0f, 0.0f).

      attrib.tangent = make_float3(1.0f, 0.0f, 0.0f);
      attrib.normal  = make_float3(0.0f, 0.0f, 1.0f);

      for (unsigned int j = 0; j <= tessV; ++j)
      {
        const float v = float(j) * vTile;

        for (unsigned int i = 0; i <= tessU; ++i)
        {
          const float u = float(i) * uTile;

          attrib.vertex   = corner + make_float3(u, v, 0.0f);
          attrib.texcoord = make_float3(u * 0.5f, v * 0.5f, 0.0f);

          attributes.push_back(attrib);
        }
      }
      break;
  }

  std::vector<unsigned int> indices;

  const unsigned int stride = tessU + 1;
  for (unsigned int j = 0; j < tessV; ++j)
  {
    for (unsigned int i = 0; i < tessU; ++i)
    {
      indices.push_back( j      * stride + i    );
      indices.push_back( j      * stride + i + 1);
      indices.push_back((j + 1) * stride + i + 1);

      indices.push_back((j + 1) * stride + i + 1);
      indices.push_back((j + 1) * stride + i    );
      indices.push_back( j      * stride + i    );
    }
  }

  std::cout << "createPlane(" << upAxis << "): Vertices = " << attributes.size() <<  ", Triangles = " << indices.size() / 3 << '\n';

  return createGeometry(attributes, indices);
}
