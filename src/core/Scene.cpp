#include "core/Scene.h"
#include "shaders/MathUtils.h"

namespace jazzfusion
{

OptixTraversableHandle Scene::createGeometry(
    OptixFunctionTable& api,
    OptixDeviceContext& context,
    CUstream cudaStream,
    std::vector<GeometryData>& geometries,
    std::vector<VertexAttributes> const& attributes,
    std::vector<unsigned int> const& indices)
{
    CUdeviceptr d_attributes;
    CUdeviceptr d_indices;

    const size_t attributesSizeInBytes = sizeof(VertexAttributes) * attributes.size();

    CUDA_CHECK(cudaMalloc((void**)&d_attributes, attributesSizeInBytes));
    CUDA_CHECK(cudaMemcpy((void*)d_attributes, attributes.data(), attributesSizeInBytes, cudaMemcpyHostToDevice));

    const size_t indicesSizeInBytes = sizeof(unsigned int) * indices.size();

    CUDA_CHECK(cudaMalloc((void**)&d_indices, indicesSizeInBytes));
    CUDA_CHECK(cudaMemcpy((void*)d_indices, indices.data(), indicesSizeInBytes, cudaMemcpyHostToDevice));

    OptixBuildInput triangleInput = {};

    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(VertexAttributes);
    triangleInput.triangleArray.numVertices = (unsigned int)attributes.size();
    triangleInput.triangleArray.vertexBuffers = &d_attributes;

    triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;

    triangleInput.triangleArray.numIndexTriplets = (unsigned int)indices.size() / 3;
    triangleInput.triangleArray.indexBuffer = d_indices;

    unsigned int triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

    triangleInput.triangleArray.flags = triangleInputFlags;
    triangleInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelBuildOptions = {};

    accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes accelBufferSizes;

    OPTIX_CHECK(api.optixAccelComputeMemoryUsage(context, &accelBuildOptions, &triangleInput, 1, &accelBufferSizes));

    CUdeviceptr d_gas; // This holds the geometry acceleration structure.

    CUDA_CHECK(cudaMalloc((void**)&d_gas, accelBufferSizes.outputSizeInBytes));

    CUdeviceptr d_tmp;

    CUDA_CHECK(cudaMalloc((void**)&d_tmp, accelBufferSizes.tempSizeInBytes));

    OptixTraversableHandle traversableHandle = 0; // This is the GAS handle which gets returned.

    OPTIX_CHECK(api.optixAccelBuild(context, cudaStream,
        &accelBuildOptions, &triangleInput, 1,
        d_tmp, accelBufferSizes.tempSizeInBytes,
        d_gas, accelBufferSizes.outputSizeInBytes,
        &traversableHandle, nullptr, 0));

    CUDA_CHECK(cudaStreamSynchronize(cudaStream));

    CUDA_CHECK(cudaFree((void*)d_tmp));

    // Track the GeometryData to be able to set them in the SBT record GeometryInstanceData and free them on exit.
    GeometryData geometry;

    geometry.indices = d_indices;
    geometry.attributes = d_attributes;
    geometry.numIndices = indices.size();
    geometry.numAttributes = attributes.size();
    geometry.gas = d_gas;

    geometries.push_back(geometry);

    return traversableHandle;
}

OptixTraversableHandle Scene::createBox(OptixFunctionTable& api,
    OptixDeviceContext& context,
    CUstream cudaStream,
    std::vector<GeometryData>& geometries)
{
    float left = -1.0f;
    float right = 1.0f;
    float bottom = -1.0f;
    float top = 1.0f;
    float back = -1.0f;
    float front = 1.0f;

    std::vector<VertexAttributes> attributes;

    VertexAttributes attrib;

    // Left.
    attrib.tangent = make_float3(0.0f, 0.0f, 1.0f);
    attrib.normal = make_float3(-1.0f, 0.0f, 0.0f);

    attrib.vertex = make_float3(left, bottom, back);
    attrib.texcoord = make_float3(0.0f, 0.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(left, bottom, front);
    attrib.texcoord = make_float3(1.0f, 0.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(left, top, front);
    attrib.texcoord = make_float3(1.0f, 1.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(left, top, back);
    attrib.texcoord = make_float3(0.0f, 1.0f, 0.0f);
    attributes.push_back(attrib);

    // Right.
    attrib.tangent = make_float3(0.0f, 0.0f, -1.0f);
    attrib.normal = make_float3(1.0f, 0.0f, 0.0f);

    attrib.vertex = make_float3(right, bottom, front);
    attrib.texcoord = make_float3(0.0f, 0.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(right, bottom, back);
    attrib.texcoord = make_float3(1.0f, 0.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(right, top, back);
    attrib.texcoord = make_float3(1.0f, 1.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(right, top, front);
    attrib.texcoord = make_float3(0.0f, 1.0f, 0.0f);
    attributes.push_back(attrib);

    // Back.
    attrib.tangent = make_float3(-1.0f, 0.0f, 0.0f);
    attrib.normal = make_float3(0.0f, 0.0f, -1.0f);

    attrib.vertex = make_float3(right, bottom, back);
    attrib.texcoord = make_float3(0.0f, 0.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(left, bottom, back);
    attrib.texcoord = make_float3(1.0f, 0.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(left, top, back);
    attrib.texcoord = make_float3(1.0f, 1.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(right, top, back);
    attrib.texcoord = make_float3(0.0f, 1.0f, 0.0f);
    attributes.push_back(attrib);

    // Front.
    attrib.tangent = make_float3(1.0f, 0.0f, 0.0f);
    attrib.normal = make_float3(0.0f, 0.0f, 1.0f);

    attrib.vertex = make_float3(left, bottom, front);
    attrib.texcoord = make_float3(0.0f, 0.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(right, bottom, front);
    attrib.texcoord = make_float3(1.0f, 0.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(right, top, front);
    attrib.texcoord = make_float3(1.0f, 1.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(left, top, front);
    attrib.texcoord = make_float3(0.0f, 1.0f, 0.0f);
    attributes.push_back(attrib);

    // Bottom.
    attrib.tangent = make_float3(1.0f, 0.0f, 0.0f);
    attrib.normal = make_float3(0.0f, -1.0f, 0.0f);

    attrib.vertex = make_float3(left, bottom, back);
    attrib.texcoord = make_float3(0.0f, 0.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(right, bottom, back);
    attrib.texcoord = make_float3(1.0f, 0.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(right, bottom, front);
    attrib.texcoord = make_float3(1.0f, 1.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(left, bottom, front);
    attrib.texcoord = make_float3(0.0f, 1.0f, 0.0f);
    attributes.push_back(attrib);

    // Top.
    attrib.tangent = make_float3(1.0f, 0.0f, 0.0f);
    attrib.normal = make_float3(0.0f, 1.0f, 0.0f);

    attrib.vertex = make_float3(left, top, front);
    attrib.texcoord = make_float3(0.0f, 0.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(right, top, front);
    attrib.texcoord = make_float3(1.0f, 0.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(right, top, back);
    attrib.texcoord = make_float3(1.0f, 1.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = make_float3(left, top, back);
    attrib.texcoord = make_float3(0.0f, 1.0f, 0.0f);
    attributes.push_back(attrib);

    std::vector<unsigned int> indices;

    for (unsigned int i = 0; i < 6; ++i)
    {
        const unsigned int idx = i * 4; // Four attributes per box face.

        indices.push_back(idx);
        indices.push_back(idx + 1);
        indices.push_back(idx + 2);

        indices.push_back(idx + 2);
        indices.push_back(idx + 3);
        indices.push_back(idx);
    }

    std::cout << "createBox(): Vertices = " << attributes.size() << ", Triangles = " << indices.size() / 3 << '\n';

    return createGeometry(api, context, cudaStream, geometries, attributes, indices);
}

OptixTraversableHandle Scene::createSphere(OptixFunctionTable& api,
    OptixDeviceContext& context,
    CUstream cudaStream,
    std::vector<GeometryData>& geometries, const unsigned int tessU, const unsigned int tessV, const float radius, const float maxTheta)
{
    assert(3 <= tessU && 3 <= tessV);

    std::vector<VertexAttributes> attributes;
    attributes.reserve((tessU + 1) * tessV);

    std::vector<unsigned int> indices;
    indices.reserve(6 * tessU * (tessV - 1));

    float phi_step = 2.0f * M_PIf / (float)tessU;
    float theta_step = maxTheta / (float)(tessV - 1);

    // Latitudinal rings.
    // Starting at the south pole going upwards on the y-axis.
    for (unsigned int latitude = 0; latitude < tessV; ++latitude) // theta angle
    {
        float theta = (float)latitude * theta_step;
        float sinTheta = sinf(theta);
        float cosTheta = cosf(theta);

        float texv = (float)latitude / (float)(tessV - 1); // Range [0.0f, 1.0f]

        // Generate vertices along the latitudinal rings.
        // On each latitude there are tessU + 1 vertices.
        // The last one and the first one are on identical positions, but have different texture coordinates!
        // Note that each second triangle connected to the two poles has zero area if the sphere is closed.
        // But since this is also used for open spheres (maxTheta < 1.0f) this is required.
        for (unsigned int longitude = 0; longitude <= tessU; ++longitude) // phi angle
        {
            float phi = (float)longitude * phi_step;
            float sinPhi = sinf(phi);
            float cosPhi = cosf(phi);

            float texu = (float)longitude / (float)tessU; // Range [0.0f, 1.0f]

            // Unit sphere coordinates are the normals.
            float3 normal = make_float3(cosPhi * sinTheta,
                -cosTheta, // -y to start at the south pole.
                -sinPhi * sinTheta);
            VertexAttributes attrib;

            attrib.vertex = normal * radius;
            attrib.tangent = make_float3(-sinPhi, 0.0f, -cosPhi);
            attrib.normal = normal;
            attrib.texcoord = make_float3(texu, texv, 0.0f);

            attributes.push_back(attrib);
        }
    }

    // We have generated tessU + 1 vertices per latitude.
    const unsigned int columns = tessU + 1;

    // Calculate indices.
    for (unsigned int latitude = 0; latitude < tessV - 1; ++latitude)
    {
        for (unsigned int longitude = 0; longitude < tessU; ++longitude)
        {
            indices.push_back(latitude * columns + longitude);           // lower left
            indices.push_back(latitude * columns + longitude + 1);       // lower right
            indices.push_back((latitude + 1) * columns + longitude + 1); // upper right

            indices.push_back((latitude + 1) * columns + longitude + 1); // upper right
            indices.push_back((latitude + 1) * columns + longitude);     // upper left
            indices.push_back(latitude * columns + longitude);           // lower left
        }
    }

    std::cout << "createSphere(): Vertices = " << attributes.size() << ", Triangles = " << indices.size() / 3 << '\n';

    return createGeometry(api, context, cudaStream, geometries, attributes, indices);
}

// Parallelogram from footpoint position, spanned by unnormalized vectors vecU and vecV, normal is normalized and on the CCW frontface.
OptixTraversableHandle Scene::createParallelogram(OptixFunctionTable& api,
    OptixDeviceContext& context,
    CUstream cudaStream,
    std::vector<GeometryData>& geometries, float3 const& position, float3 const& vecU, float3 const& vecV, float3 const& normal)
{
    std::vector<VertexAttributes> attributes;

    VertexAttributes attrib;

    // Same for all four vertices in this parallelogram.
    attrib.tangent = normalize(vecU);
    attrib.normal = normal;

    attrib.vertex = position; // left bottom
    attrib.texcoord = make_float3(0.0f, 0.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = position + vecU; // right bottom
    attrib.texcoord = make_float3(1.0f, 0.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = position + vecU + vecV; // right top
    attrib.texcoord = make_float3(1.0f, 1.0f, 0.0f);
    attributes.push_back(attrib);

    attrib.vertex = position + vecV; // left top
    attrib.texcoord = make_float3(0.0f, 1.0f, 0.0f);
    attributes.push_back(attrib);

    std::vector<unsigned int> indices;

    indices.push_back(0);
    indices.push_back(1);
    indices.push_back(2);

    indices.push_back(2);
    indices.push_back(3);
    indices.push_back(0);

    std::cout << "createParallelogram(): Vertices = " << attributes.size() << ", Triangles = " << indices.size() / 3 << '\n';

    return createGeometry(api, context, cudaStream, geometries, attributes, indices);
}

OptixTraversableHandle Scene::createTorus(OptixFunctionTable& api,
    OptixDeviceContext& context,
    CUstream cudaStream,
    std::vector<GeometryData>& geometries, const unsigned int tessU, const unsigned int tessV, const float innerRadius, const float outerRadius)
{
    assert(3 <= tessU && 3 <= tessV);

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

    const float u = (float)tessU;
    const float v = (float)tessV;

    float phi_step = 2.0f * M_PIf / u;
    float theta_step = 2.0f * M_PIf / v;

    // Setup vertices and normals.
    // Generate the torus exactly like the sphere with rings around the origin along the latitudes.
    for (unsigned int latitude = 0; latitude <= tessV; ++latitude) // theta angle
    {
        const float theta = (float)latitude * theta_step;
        const float sinTheta = sinf(theta);
        const float cosTheta = cosf(theta);

        const float radius = innerRadius + outerRadius * cosTheta;

        for (unsigned int longitude = 0; longitude <= tessU; ++longitude) // phi angle
        {
            const float phi = (float)longitude * phi_step;
            const float sinPhi = sinf(phi);
            const float cosPhi = cosf(phi);

            VertexAttributes attrib;

            attrib.vertex = make_float3(radius * cosPhi, outerRadius * sinTheta, radius * -sinPhi);
            attrib.tangent = make_float3(-sinPhi, 0.0f, -cosPhi);
            attrib.normal = make_float3(cosPhi * cosTheta, sinTheta, -sinPhi * cosTheta);
            attrib.texcoord = make_float3((float)longitude / u, (float)latitude / v, 0.0f);

            attributes.push_back(attrib);
        }
    }

    // We have generated tessU + 1 vertices per latitude.
    const unsigned int columns = tessU + 1;

    // Setup indices
    for (unsigned int latitude = 0; latitude < tessV; ++latitude)
    {
        for (unsigned int longitude = 0; longitude < tessU; ++longitude)
        {
            indices.push_back(latitude * columns + longitude);           // lower left
            indices.push_back(latitude * columns + longitude + 1);       // lower right
            indices.push_back((latitude + 1) * columns + longitude + 1); // upper right

            indices.push_back((latitude + 1) * columns + longitude + 1); // upper right
            indices.push_back((latitude + 1) * columns + longitude);     // upper left
            indices.push_back(latitude * columns + longitude);           // lower left
        }
    }

    std::cout << "createTorus(): Vertices = " << attributes.size() << ", Triangles = " << indices.size() / 3 << '\n';

    return createGeometry(api, context, cudaStream, geometries, attributes, indices);
}

OptixTraversableHandle Scene::createPlane(OptixFunctionTable& api,
    OptixDeviceContext& context,
    CUstream cudaStream,
    std::vector<GeometryData>& geometries, const unsigned int tessU, const unsigned int tessV, const unsigned int upAxis)
{
    assert(1 <= tessU && 1 <= tessV);

    const float uTile = 2.0f / float(tessU);
    const float vTile = 2.0f / float(tessV);

    float3 corner;

    std::vector<VertexAttributes> attributes;

    VertexAttributes attrib;

    switch (upAxis)
    {
    case 0:                                      // Positive x-axis is the geometry normal, create geometry on the yz-plane.
        corner = make_float3(0.0f, -1.0f, 1.0f); // Lower front corner of the plane. texcoord (0.0f, 0.0f).

        attrib.tangent = make_float3(0.0f, 0.0f, -1.0f);
        attrib.normal = make_float3(1.0f, 0.0f, 0.0f);

        for (unsigned int j = 0; j <= tessV; ++j)
        {
            const float v = float(j) * vTile;

            for (unsigned int i = 0; i <= tessU; ++i)
            {
                const float u = float(i) * uTile;

                attrib.vertex = corner + make_float3(0.0f, v, -u);
                attrib.texcoord = make_float3(u * 0.5f, v * 0.5f, 0.0f);

                attributes.push_back(attrib);
            }
        }
        break;

    case 1:                                      // Positive y-axis is the geometry normal, create geometry on the xz-plane.
        corner = make_float3(-1.0f, 0.0f, 1.0f); // left front corner of the plane. texcoord (0.0f, 0.0f).

        attrib.tangent = make_float3(1.0f, 0.0f, 0.0f);
        attrib.normal = make_float3(0.0f, 1.0f, 0.0f);

        for (unsigned int j = 0; j <= tessV; ++j)
        {
            const float v = float(j) * vTile;

            for (unsigned int i = 0; i <= tessU; ++i)
            {
                const float u = float(i) * uTile;

                attrib.vertex = corner + make_float3(u, 0.0f, -v);
                attrib.texcoord = make_float3(u * 0.5f, v * 0.5f, 0.0f);

                attributes.push_back(attrib);
            }
        }
        break;

    case 2:                                       // Positive z-axis is the geometry normal, create geometry on the xy-plane.
        corner = make_float3(-1.0f, -1.0f, 0.0f); // Lower left corner of the plane. texcoord (0.0f, 0.0f).

        attrib.tangent = make_float3(1.0f, 0.0f, 0.0f);
        attrib.normal = make_float3(0.0f, 0.0f, 1.0f);

        for (unsigned int j = 0; j <= tessV; ++j)
        {
            const float v = float(j) * vTile;

            for (unsigned int i = 0; i <= tessU; ++i)
            {
                const float u = float(i) * uTile;

                attrib.vertex = corner + make_float3(u, v, 0.0f);
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
            indices.push_back(j * stride + i);
            indices.push_back(j * stride + i + 1);
            indices.push_back((j + 1) * stride + i + 1);

            indices.push_back((j + 1) * stride + i + 1);
            indices.push_back((j + 1) * stride + i);
            indices.push_back(j * stride + i);
        }
    }

    std::cout << "createPlane(" << upAxis << "): Vertices = " << attributes.size() << ", Triangles = " << indices.size() / 3 << '\n';

    return createGeometry(api, context, cudaStream, geometries, attributes, indices);
}

} // namespace jazzfusion