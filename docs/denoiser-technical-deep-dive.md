# RELAX-Based Real-Time Path Tracing Denoiser

## Overview

Implementation of NVIDIA's RELAX denoiser for real-time path tracing with separate diffuse and specular channels. Uses temporal accumulation with virtual motion reprojection for specular reflections and edge-preserving A-trous wavelet filtering.

## Pipeline Architecture

```
Input: Noisy diffuse + specular radiance, G-buffer data
├── Temporal Accumulation
│   ├── Surface Motion Reprojection (diffuse + specular)
│   └── Virtual Motion Reprojection (specular only)
├── Anti-Firefly Pass (RCRS bilateral filtering)
├── History Fix (variance stabilization)
├── A-trous Wavelet Filtering (5 passes, step sizes: 1,2,4,8,16)
└── Output: Denoised diffuse + specular
```

## Temporal Accumulation

### Surface Motion Based Reprojection

Reconstructs previous frame world positions using camera motion and depth:
```cpp
Float3 prevWorldPos = currentWorldPos + cameraDelta;
Float2 prevUV = prevCamera.worldDirectionToUV(normalize(prevWorldPos - prevCamera.pos));
```

**Disocclusion Detection**: Uses plane distance threshold in world space:
- `threshold = disocclusionThreshold * frustumSize`
- `valid = abs(dot(posDiff, normal)) < threshold`

**Bicubic vs Bilinear**: Validates 12-tap bicubic footprint, falls back to weighted bilinear if any tap fails depth test.

### Virtual Motion Reprojection (Specular)

**Core Principle**: Specular reflections track virtual surfaces at reflection hit distance rather than geometric surface motion.

**Virtual Position Calculation**:
```cpp
Float3 V = -normalize(currentViewVector);
Float3 reflectionDir = reflect(-V, currentNormal);
Float3 virtualWorldPos = currentWorldPos + reflectionDir * hitDistFocused;
```

**Why Reflection Direction**: Specular samples represent radiance from directions determined by surface normals and view angles. When camera moves, the effective "virtual surface" the reflection samples moves along the reflection ray, not with geometric surface motion.

**Thin Lens Equation**: `hitDistFocused = hitDist / max(1.0 + hitDist * curvature, 0.001)`
- Accounts for surface curvature effects on reflection focus
- Prevents numerical instability for infinite hit distances

**Validation**: Requires all 4 bilinear taps valid (stricter than surface motion) due to reflection ray sensitivity to geometric changes.

## Anti-Firefly Pass

**Purpose**: Cross Bilateral Rank-Conditioned Rank-Selection (RCRS) filter to remove temporal fireflies - bright outlier pixels that survive temporal accumulation.

**Core Algorithm**:
```cpp
// Poisson disk sampling in 4-pixel radius
for each sample in poissonDisk[16]:
    bilateralWeight = depthWeight * normalWeight * planeWeight * roughnessWeight
    luminanceWeight = exp(-|centerLuma - sampleLuma| / (sigmaMa * centerLuma))
    finalWeight = bilateralWeight * luminanceWeight
```

**Edge Preservation**:
- **Depth Weight**: `exp(-|depthDiff| / (sigmaZ * centerDepth))`
- **Normal Weight**: `pow(dot(centerNormal, sampleNormal), 1/sigmaN)`
- **Plane Weight**: `exp(-planeDistance / threshold)` - prevents cross-surface bleeding
- **Roughness Weight**: `exp(-|roughnessDiff| * 8)` - maintains surface properties

**Energy Preservation**: Blends with original color when filtered result changes luminance by >20% to prevent over-darkening.

## A-trous Wavelet Filtering

### Edge-Preserving Weights

**Geometry Weight**: `exp(-|dot(samplePos - centerPos, normal)| / threshold)`
- Preserves edges where surface geometry changes significantly
- Threshold scaled by view distance for perspective consistency

**Normal Weight Calculation**:

*Diffuse*: Simple angle-based rejection
```cpp
angle = acos(dot(centerNormal, sampleNormal))
weight = smoothstep(0, angleThreshold, angle)
```

*Specular*: View-vector dependent weighting
```cpp
cosaN = dot(centerNormal, sampleNormal)
cosaV = dot(centerViewVector, sampleViewVector)
cosa = min(cosaN, cosaV)
weight = function(acos(cosa), adaptiveThreshold)
```

**Rationale**: Specular reflections depend on both surface orientation and viewing direction. A surface with identical normal but different viewing angle produces different reflection directions, requiring rejection.

### Adaptive Parameters

**History-Based Relaxation**:
- `relaxation = saturate(historyLength / 5.0)`
- Low history → wider acceptance angles
- Prevents over-rejection during disocclusion recovery

**Confidence-Driven Adaptation**:
```cpp
angle *= 10.0 - 9.0 * confidence
luminanceWeight *= (1.0 - confidenceRelaxation)
```
- Low confidence increases spatial support
- Trades noise reduction for potential blur

**Roughness-Dependent Specular Weighting**:
- `lobeAngle = atan(roughness² * 0.75 / (1.0 - 0.75 + ε))`
- Rough surfaces → wider acceptance cones
- Mirror surfaces → strict directional filtering

### Step Size Progression

Uses powers of 2: [1, 2, 4, 8, 16] with Gaussian kernel weights [0.44198, 0.27901].

**Random Offsets**: For steps > 4, adds `±stepSize/2` random offset to break up ringing artifacts from regular sampling patterns.

## Mathematical Foundations

### Luminance Edge Stopping

**Variance-Normalized Threshold**:
```cpp
phiLInv = 1.0 / max(1e-4, phiL * sqrt(variance))
weight = exp(-|centerLum - sampleLum| * phiLInv)
```

**Adaptive Phi Calculation**:
- Specular: `phiL = basePhi * lerp(0.3, 1.0, roughness)`
- Accounts for roughness-dependent variance characteristics

### Confidence Metrics

**Specular Reprojection Confidence**:
- Surface motion confidence: `dot(centerNormal, prevNormal) > 0`
- Virtual motion confidence: `hitDistanceConsistency * normalConsistency`
- Combined via: `lerp(surfaceConfidence, virtualConfidence, virtualHistoryAmount)`

**Virtual History Amount**: Determines blend weight between surface and virtual motion
```cpp
virtualAmount = specularDominantDirection.confidence 
              * backfaceCheck 
              * normalWeight 
              * roughnessWeight 
              * lookingBackValidation
```

## Implementation Details

### Memory Layout
- Ping-pong buffers for illumination + 2nd moment
- Separate fast/responsive accumulation buffers
- History length buffer for adaptive parameters

### Numerical Stability
- Epsilon guards: `max(value, 1e-6)` for division operations
- Variance clamping: `max(variance, 0.0)` prevents negative values
- Weight normalization: `weight = sum < 1e-4 ? 0 : weight/sum`

### Performance Optimizations
- Shared memory for 3x3 normal/roughness data
- Early termination on geometry weight < 1e-4
- Material ID comparison via integer ops
- Precomputed kernel weights in constant memory

## Failure Cases & Mitigations

**Virtual Motion Breakdown**: When reflection rays miss previous frame → fallback to surface motion
**History Contamination**: Aggressive disocclusion detection prevents propagation of invalid samples  
**Temporal Lag**: Looking-back validation using 2+ frame history consistency checks
**Fireflies**: Variance boosting for zero-sample pixels: `variance += specVarianceBoost * (1.0 - confidence)`