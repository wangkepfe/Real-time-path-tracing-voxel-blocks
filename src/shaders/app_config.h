

// This header with defines is included in all shaders
// to be able to switch different code paths at a central location.
// Changing any setting here will rebuild the whole solution.

#pragma once

#ifndef APP_CONFIG_H
#define APP_CONFIG_H

// Switch between full optimization and full debug information for module compilation and program linking.
#define USE_MAX_OPTIMIZATION 1

#define RT_DEFAULT_MAX 1.e27f

// Scales the m_sceneEpsilonFactor to give the effective SystemParameter::sceneEpsilon.
#define SCENE_EPSILON_SCALE 1.0e-7f

// Prevent that division by very small floating point values results in huge values, for example dividing by pdf.
#define DENOMINATOR_EPSILON 1.0e-6f

// 0 == Brute force path tracing without next event estimation (direct lighting). // Debug setting to compare lighting results.
// 1 == Next event estimation per path vertex (direct lighting) and using MIS with power heuristic. // Default.
#define USE_NEXT_EVENT_ESTIMATION 0

#endif // APP_CONFIG_H
