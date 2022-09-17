#include "BlockMesher.h"
#include <unordered_map>

namespace vox
{

void BlockMesher::addVoxel(const Voxel& voxel, uint8_t x, uint8_t y, uint8_t z)
{
    QuadFace facesToAdd[6] =
    {
        { x, y, z, AxisX },
        { x, y, z, AxisY },
        { x, y, z, AxisZ },
        { (uint8_t)(x + 1), y, z, AxisX },
        { x, (uint8_t)(y + 1), z, AxisY },
        { x, y, (uint8_t)(z + 1), AxisZ },
    };

    for (int faceId = 0; faceId < 6; ++faceId)
    {
        auto& it = faces.find(facesToAdd[faceId]);

        if (it != faces.end())
        {
            // If the face exits, remove the face
            faces.erase(it);
        }
        else
        {
            // If the face does not exist, create this face
            faces.insert(facesToAdd[faceId]);
        }
    }
}

void BlockMesher::facesToMesh()
{
    std::unordered_map<QuadFace, int, QuadFaceHasher, QuadFaceEqualOperator> uniqueVertices;

    int uniqueVertexIndex = 0;

    for (const auto& face : faces)
    {
        std::vector<QuadFace> faceVertices(4);

        if (face.axis == AxisX)
        {
            faceVertices[0] = { (uint8_t)(face.x + 0), (uint8_t)(face.y + 0), (uint8_t)(face.z + 0), 0 };
            faceVertices[1] = { (uint8_t)(face.x + 0), (uint8_t)(face.y + 1), (uint8_t)(face.z + 0), 0 };
            faceVertices[2] = { (uint8_t)(face.x + 0), (uint8_t)(face.y + 0), (uint8_t)(face.z + 1), 0 };
            faceVertices[3] = { (uint8_t)(face.x + 0), (uint8_t)(face.y + 1), (uint8_t)(face.z + 1), 0 };
        }
        else if (face.axis == AxisY)
        {
            faceVertices[0] = { (uint8_t)(face.x + 0), (uint8_t)(face.y + 0), (uint8_t)(face.z + 0), 0 };
            faceVertices[1] = { (uint8_t)(face.x + 1), (uint8_t)(face.y + 0), (uint8_t)(face.z + 0), 0 };
            faceVertices[2] = { (uint8_t)(face.x + 0), (uint8_t)(face.y + 0), (uint8_t)(face.z + 1), 0 };
            faceVertices[3] = { (uint8_t)(face.x + 1), (uint8_t)(face.y + 0), (uint8_t)(face.z + 1), 0 };
        }
        else if (face.axis == AxisZ)
        {
            faceVertices[0] = { (uint8_t)(face.x + 0), (uint8_t)(face.y + 0), (uint8_t)(face.z + 0), 0 };
            faceVertices[1] = { (uint8_t)(face.x + 1), (uint8_t)(face.y + 0), (uint8_t)(face.z + 0), 0 };
            faceVertices[2] = { (uint8_t)(face.x + 0), (uint8_t)(face.y + 1), (uint8_t)(face.z + 0), 0 };
            faceVertices[3] = { (uint8_t)(face.x + 1), (uint8_t)(face.y + 1), (uint8_t)(face.z + 0), 0 };
        }

        std::vector<size_t> uniqueFaceVertexIndices(4);

        for (int i = 0; i < 4; ++i)
        {
            if (uniqueVertices.find(faceVertices[i]) != uniqueVertices.end())
            {
                // If the vertex exists, save the unique vertex index
                uniqueFaceVertexIndices[i] = uniqueVertices[faceVertices[i]];
            }
            else
            {
                // If the vertex does not exist, create a new vertex, then save the index

                uniqueFaceVertexIndices[i] = uniqueVertexIndex;

                uniqueVertices[faceVertices[i]] = uniqueVertexIndex;

                jazzfusion::VertexAttributes vertex;
                vertex.vertex = jazzfusion::Float3((float)faceVertices[i].x, (float)faceVertices[i].y, (float)faceVertices[i].z);
                attributes.push_back(vertex);

                uniqueVertexIndex++;
            }
        }

        indices.push_back(uniqueFaceVertexIndices[0]);
        indices.push_back(uniqueFaceVertexIndices[1]);
        indices.push_back(uniqueFaceVertexIndices[2]);
        indices.push_back(uniqueFaceVertexIndices[1]);
        indices.push_back(uniqueFaceVertexIndices[2]);
        indices.push_back(uniqueFaceVertexIndices[3]);
    }
}

void BlockMesher::process()
{
    voxelChunk.foreach([&](const Voxel& voxel, int x, int y, int z)
        {
            if (voxel.id == 1)
            {
                addVoxel(voxel, x, y, z);
            }
        });

    facesToMesh();
}

void BlockMesher::update(const Voxel& voxel, int x, int y, int z)
{
    addVoxel(voxel, x, y, z);

    attributes.clear();
    indices.clear();

    facesToMesh();
}

}