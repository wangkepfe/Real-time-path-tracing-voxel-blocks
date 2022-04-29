

#include "shaders/app_config.h"

#include "shaders/vector_math.h"

#include "core/Application.h"

#include <cstring>
#include <iostream>
#include <sstream>

#include "core/MyAssert.h"


OptixTraversableHandle Application::createTorus(const unsigned int tessU, const unsigned int tessV, const float innerRadius, const float outerRadius)
{
  MY_ASSERT(3 <= tessU && 3 <= tessV);

  // The torus is a ring with radius outerRadius rotated around the y-axis along the circle with innerRadius.
  /*           +y
      ___       |       ___
    /     \           /     \
   |       |    |    |       |
   |       |         |       |
    \ ___ /     |     \ ___ /
                         <--->
                          outerRadius
                <------->
                innerRadius
  */

  std::vector<VertexAttributes> attributes;
  attributes.reserve((tessU + 1) * (tessV + 1));

  std::vector<unsigned int> indices;
  indices.reserve(8 * tessU * tessV);

  const float u = (float) tessU;
  const float v = (float) tessV;

  float phi_step   = 2.0f * M_PIf / u;
  float theta_step = 2.0f * M_PIf / v;

  // Setup vertices and normals.
  // Generate the torus exactly like the sphere with rings around the origin along the latitudes.
  for (unsigned int latitude = 0; latitude <= tessV; ++latitude) // theta angle
  {
    const float theta    = (float) latitude * theta_step;
    const float sinTheta = sinf(theta);
    const float cosTheta = cosf(theta);

    const float radius = innerRadius + outerRadius * cosTheta;

    for (unsigned int longitude = 0; longitude <= tessU; ++longitude) // phi angle
    {
      const float phi    = (float) longitude * phi_step;
      const float sinPhi = sinf(phi);
      const float cosPhi = cosf(phi);

      VertexAttributes attrib;

      attrib.vertex   = make_float3(radius * cosPhi, outerRadius * sinTheta, radius * -sinPhi);
      attrib.tangent  = make_float3(-sinPhi, 0.0f, -cosPhi);
      attrib.normal   = make_float3(cosPhi * cosTheta, sinTheta, -sinPhi * cosTheta);
      attrib.texcoord = make_float3((float) longitude / u, (float) latitude / v, 0.0f);

      attributes.push_back(attrib);
    }
  }

  // We have generated tessU + 1 vertices per latitude.
  const unsigned int columns = tessU + 1;

  // Setup indices
  for (unsigned int latitude = 0; latitude < tessV; ++latitude )
  {
    for (unsigned int longitude = 0; longitude < tessU; ++longitude)
    {
      indices.push_back( latitude      * columns + longitude    );  // lower left
      indices.push_back( latitude      * columns + longitude + 1);  // lower right
      indices.push_back((latitude + 1) * columns + longitude + 1);  // upper right

      indices.push_back((latitude + 1) * columns + longitude + 1);  // upper right
      indices.push_back((latitude + 1) * columns + longitude    );  // upper left
      indices.push_back( latitude      * columns + longitude    );  // lower left
    }
  }

  std::cout << "createTorus(): Vertices = " << attributes.size() <<  ", Triangles = " << indices.size() / 3 << '\n';

  return createGeometry(attributes, indices);
}
