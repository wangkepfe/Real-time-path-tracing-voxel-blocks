#include <vector>
#include <cmath>

// Compute the 1D Halton value at index 'i' (1-based) for a given base
inline float halton(int i, int base)
{
    float result = 0.0f;
    float f = 1.0f / base;
    int index = i;
    while (index > 0)
    {
        result += f * (index % base);
        index /= base;
        f /= base;
    }
    return result;
}

// Generate a 2D Halton sequence of length N.
// Returns a flat vector: [x0, y0, x1, y1, ..., xN-1, yN-1].
inline std::vector<float> GenerateHalton2D(int N)
{
    std::vector<float> seq;
    seq.reserve(2 * N);
    for (int i = 1; i <= N; ++i)
    {
        // base 2 for the x‑coordinate, base 3 for the y‑coordinate
        seq.push_back(halton(i, 2));
        seq.push_back(halton(i, 3));
    }
    return seq;
}

// Concentric‑disk mapping (Shirley & Chiu)
inline void ConcentricSampleDisk(float u, float v, float &x, float &y)
{
    // bring u,v from [0,1] to [-1,1]
    float a = 2.0f * u - 1.0f;
    float b = 2.0f * v - 1.0f;

    if (a == 0.0f && b == 0.0f)
    {
        x = 0.0f;
        y = 0.0f;
        return;
    }

    float r, phi;
    if (std::abs(a) > std::abs(b))
    {
        r = a;
        phi = (M_PI / 4.0f) * (b / a);
    }
    else
    {
        r = b;
        phi = (M_PI / 2.0f) - (M_PI / 4.0f) * (a / b);
    }

    x = r * std::cos(phi);
    y = r * std::sin(phi);
}

// Generate N points on the unit disk by warping a 2D Halton sequence
inline std::vector<float> GenerateDiskHalton2D(int N)
{
    std::vector<float> seq;
    seq.reserve(2 * N);

    for (int i = 1; i <= N; ++i)
    {
        // get the first two Halton dims (base 2, base 3)
        float u = halton(i, 2);
        float v = halton(i, 3);

        // warp into disk
        float x, y;
        ConcentricSampleDisk(u, v, x, y);

        seq.push_back(x);
        seq.push_back(y);
    }

    return seq;
}

// Generate N points on the unit disk by rejection sampling from a 2D Halton sequence
inline std::vector<float> GenerateDiskHalton2DRejection(int N)
{
    std::vector<float> seq;
    seq.reserve(2 * N);

    int count = 0; // number of accepted points
    int idx = 1;   // Halton index

    while (count < N)
    {
        // sample (u,v) in [0,1]^2 via Halton (bases 2 and 3)
        float u = halton(idx, 2);
        float v = halton(idx, 3);

        // map to [-1,1]^2
        float x = 2.0f * u - 1.0f;
        float y = 2.0f * v - 1.0f;

        // accept if inside unit circle
        if (x * x + y * y <= 1.0f)
        {
            seq.push_back(x);
            seq.push_back(y);
            ++count;
        }

        ++idx;
    }

    return seq;
}