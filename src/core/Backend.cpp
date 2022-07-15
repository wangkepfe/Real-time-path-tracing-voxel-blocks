#include "core/Backend.h"
#include "util/DebugUtils.h"
#include "core/OptixRenderer.h"
#include "core/UI.h"
#include "postprocessing/PostProcessor.h"
#include "denoising/Denoiser.h"
#include "core/InputHandler.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>

namespace jazzfusion
{

static void errorCallback(int error, const char* description)
{
    std::cerr << "Error: " << error << ": " << description << '\n';
}

void Backend::init()
{
    m_width = 3840;
    m_height = 2160;

    m_dynamicResolution = false;

    m_minRenderWidth = 480;
    m_minRenderHeight = 270;

    m_maxRenderWidth = 1920;
    m_maxRenderHeight = 1080;

    m_maxRenderWidth = 960;
    m_maxRenderHeight = 540;

    // m_maxRenderWidth = 3840;
    // m_maxRenderHeight = 2160;

    glfwSetErrorCallback(errorCallback);
    if (!glfwInit())
    {
        throw std::runtime_error("GLFW failed to initialize.");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    m_window = glfwCreateWindow(m_width, m_height, "JazzFusion Renderer", NULL, NULL);
    if (!m_window)
    {
        throw std::runtime_error("glfwCreateWindow() failed.");
    }

    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if (glfwRawMouseMotionSupported())
    {
        glfwSetInputMode(m_window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }
    glfwSetKeyCallback(m_window, InputHandler::KeyCallback);
    glfwSetCursorPosCallback(m_window, InputHandler::CursorPosCallback);

    glfwMakeContextCurrent(m_window);
    if (glewInit() != GL_NO_ERROR)
    {
        throw std::runtime_error("GLEW failed to initialize.");
    }

    // Initialize DevIL once.
    ilInit();

    initOpenGL();
    initInterop();

    dumpSystemInformation();

    CUDA_CHECK(cudaFree(0));
    cuCtxGetCurrent(&m_cudaContext);
    CUDA_CHECK(cudaStreamCreate(&m_cudaStream));

    auto& renderer = OptixRenderer::Get();

    renderer.setWidth(m_maxRenderWidth);
    renderer.setHeight(m_maxRenderHeight);

    m_frameNum = 0;
}

void Backend::mainloop()
{
    auto& ui = UI::Get();
    auto& renderer = OptixRenderer::Get();
    auto& postProcessor = PostProcessor::Get();
    auto& denoiser = Denoiser::Get();
    auto& inputHandler = InputHandler::Get();

    m_timer.init();

    while (!glfwWindowShouldClose(m_window))
    {
        glfwPollEvents();

        float minFrameTimeAllowed = 1000.0f / m_maxFpsAllowed;
        m_timer.updateWithLimiter(minFrameTimeAllowed);

        dynamicResolution();

        inputHandler.update();

        renderer.render();

        denoiser.run(renderer.getWidth(), renderer.getHeight(), m_historyRenderWidth, m_historyRenderHeight);

        mapInteropBuffer();
        postProcessor.run(m_interopBuffer, renderer.getWidth(), renderer.getHeight(), m_width, m_height);
        unmapInteropBuffer();

        ui.update();

        display();

        ui.render();

        glfwSwapBuffers(m_window);

        m_frameNum++;
        m_accumulationCounter++;
    }
}

void Backend::dynamicResolution()
{
    auto& renderer = OptixRenderer::Get();
    float deltaTime = m_timer.getDeltaTime();

    int renderWidth = renderer.getWidth();
    int renderHeight = renderer.getHeight();

    m_historyRenderWidth = renderWidth;
    m_historyRenderHeight = renderHeight;

    if (m_dynamicResolution == false)
    {
        renderWidth = m_maxRenderWidth;
        renderHeight = m_maxRenderHeight;
    }
    else
    {
        float targetFrameTimeHigh = 1000.0f / (m_targetFPS - 2.0f);
        float targetFrameTimeLow = 1000.0f / (m_targetFPS + 2.0f);

        if (targetFrameTimeHigh < deltaTime || targetFrameTimeLow > deltaTime)
        {
            float ratio = (1000.0f / m_targetFPS) / deltaTime;
            ratio = sqrtf(ratio);
            renderWidth *= ratio;
        }

        // Safe resolution
        renderWidth = renderWidth + ((renderWidth % 16 < 8) ? (-renderWidth % 16) : (16 - renderWidth % 16));
        renderWidth = clampi(renderWidth, m_minRenderWidth, m_maxRenderWidth);
        renderHeight = (renderWidth / 16) * 9;
    }

    renderer.setWidth(renderWidth);
    renderer.setHeight(renderHeight);
    renderer.getCamera().resolution = Float2(renderWidth, renderHeight);

    static float timerCounter = 0.0f;
    timerCounter += deltaTime;
    if (timerCounter > 1000.0f)
    {
        timerCounter -= 1000.0f;
        m_currentFPS = 1000.0f / deltaTime;
        m_currentRenderWidth = renderWidth;
    }
}

void Backend::clear()
{
    CUDA_CHECK(cudaStreamSynchronize(m_cudaStream));
    CUDA_CHECK(cudaStreamDestroy(m_cudaStream));

    CUDA_CHECK(cudaGraphicsUnregisterResource(m_cudaGraphicsResource));
    glDeleteBuffers(1, &m_pbo);

    glDeleteBuffers(1, &m_vboAttributes);
    glDeleteBuffers(1, &m_vboIndices);
    glDeleteProgram(m_glslProgram);

    ilShutDown();
    glfwTerminate();
}

void Backend::initOpenGL()
{
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    glViewport(0, 0, m_width, m_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

    glGenTextures(1, &m_hdrTexture);
    assert(m_hdrTexture != 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    // GLSL shaders objects and program.
    m_glslVS = 0;
    m_glslFS = 0;
    m_glslProgram = 0;

    m_positionLocation = -1;
    m_texCoordLocation = -1;

    static const std::string vsSource =
        "#version 330                                   \n"
        "layout(location = 0) in vec2 attrPosition;     \n"
        "layout(location = 1) in vec2 attrTexCoord;     \n"
        "out vec2 varTexCoord;                          \n"
        "void main()                                    \n"
        "{                                              \n"
        "    gl_Position = vec4(attrPosition, 0.0, 1.0);\n"
        "    varTexCoord = attrTexCoord;                \n"
        "}                                              \n";

    static const std::string fsSource =
        "#version 330                                                                      \n"
        "uniform sampler2D samplerHDR;                                                     \n"
        "uniform float gain;                                                               \n"
        "uniform float maxWhite;                                                           \n"
        "in vec2 varTexCoord;                                                              \n"
        "layout(location = 0, index = 0) out vec4 outColor;                                \n"
        "void main()                                                                       \n"
        "{                                                                                 \n"
        "    vec3 color = texture(samplerHDR, varTexCoord).rgb;                            \n"
        "    float lum = dot(color, vec3(0.2126f, 0.7152f, 0.0722f));                      \n"
        "    color = color * gain * (1.0f + (lum / (maxWhite * maxWhite))) / (1.0f + lum); \n"
        "    color = pow(color, vec3(1.0f / 2.2f));                                        \n"
        "    outColor = vec4(color, 1.0f);                                                 \n"
        "}                                                                                 \n";

    GLint vsCompiled = 0;
    GLint fsCompiled = 0;

    m_glslVS = glCreateShader(GL_VERTEX_SHADER);
    if (m_glslVS)
    {
        GLsizei len = (GLsizei)vsSource.size();
        const GLchar* vs = vsSource.c_str();
        glShaderSource(m_glslVS, 1, &vs, &len);
        glCompileShader(m_glslVS);

        glGetShaderiv(m_glslVS, GL_COMPILE_STATUS, &vsCompiled);
        assert(vsCompiled);
    }

    m_glslFS = glCreateShader(GL_FRAGMENT_SHADER);
    if (m_glslFS)
    {
        GLsizei len = (GLsizei)fsSource.size();
        const GLchar* fs = fsSource.c_str();
        glShaderSource(m_glslFS, 1, &fs, &len);
        glCompileShader(m_glslFS);

        glGetShaderiv(m_glslFS, GL_COMPILE_STATUS, &fsCompiled);
        assert(fsCompiled);
    }

    m_glslProgram = glCreateProgram();
    if (m_glslProgram)
    {
        GLint programLinked = 0;

        if (m_glslVS && vsCompiled)
        {
            glAttachShader(m_glslProgram, m_glslVS);
        }
        if (m_glslFS && fsCompiled)
        {
            glAttachShader(m_glslProgram, m_glslFS);
        }

        glLinkProgram(m_glslProgram);

        glGetProgramiv(m_glslProgram, GL_LINK_STATUS, &programLinked);
        assert(programLinked);

        if (programLinked)
        {
            glUseProgram(m_glslProgram);

            m_positionLocation = glGetAttribLocation(m_glslProgram, "attrPosition");
            assert(m_positionLocation != -1);

            m_texCoordLocation = glGetAttribLocation(m_glslProgram, "attrTexCoord");
            assert(m_texCoordLocation != -1);

            glUniform1f(glGetUniformLocation(m_glslProgram, "gain"), m_toneMapGain);
            glUniform1f(glGetUniformLocation(m_glslProgram, "maxWhite"), m_toneMapMaxWhite);

            glUseProgram(0);
        }
    }

    // Two hardcoded triangles in the identity matrix pojection coordinate system with 2D texture coordinates.
    const float attributes[16] =
    {
        // vertex2f,   texcoord2f
        -1.0f, -1.0f, 0.0f, 0.0f,
        1.0f, -1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, 0.0f, 1.0f
    };

    unsigned int indices[6] =
    {
        0, 1, 2,
        2, 3, 0
    };

    glGenBuffers(1, &m_vboAttributes);
    assert(m_vboAttributes != 0);

    glGenBuffers(1, &m_vboIndices);
    assert(m_vboIndices != 0);

    // Setup the vertex arrays from the interleaved vertex attributes.
    glBindBuffer(GL_ARRAY_BUFFER, m_vboAttributes);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)sizeof(float) * 16, (GLvoid const*)attributes, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vboIndices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)sizeof(unsigned int) * 6, (const GLvoid*)indices, GL_STATIC_DRAW);

    glVertexAttribPointer(m_positionLocation, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (GLvoid*)0);
    glVertexAttribPointer(m_texCoordLocation, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (GLvoid*)(sizeof(float) * 2));
}

void Backend::display()
{
    // Bind texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei)m_width, (GLsizei)m_height, 0, GL_RGBA, GL_FLOAT, (void*)0);

    // Bind buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_vboAttributes);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vboIndices);
    glEnableVertexAttribArray(m_positionLocation);
    glEnableVertexAttribArray(m_texCoordLocation);

    glUseProgram(m_glslProgram);

    glUniform1f(glGetUniformLocation(m_glslProgram, "gain"), m_toneMapGain);
    glUniform1f(glGetUniformLocation(m_glslProgram, "maxWhite"), m_toneMapMaxWhite);

    glDrawElements(GL_TRIANGLES, (GLsizei)6, GL_UNSIGNED_INT, (const GLvoid*)0);

    glUseProgram(0);

    glDisableVertexAttribArray(m_positionLocation);
    glDisableVertexAttribArray(m_texCoordLocation);
}

void Backend::initInterop()
{
    m_pbo = 0;
    glGenBuffers(1, &m_pbo);
    assert(m_pbo != 0);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cudaGraphicsResource, m_pbo, cudaGraphicsRegisterFlagsNone));

    size_t size;

    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&m_interopBuffer, &size, m_cudaGraphicsResource));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream));

    assert(m_width * m_height * sizeof(float) * 4 <= size);
}

void Backend::mapInteropBuffer()
{
    size_t size;
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&m_interopBuffer, &size, m_cudaGraphicsResource));
}

void Backend::unmapInteropBuffer()
{
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream));
}

void Backend::dumpSystemInformation()
{
    int versionDriver = 0;
    CUDA_CHECK(cudaDriverGetVersion(&versionDriver));

    // The version is returned as (1000 * major + 10 * minor).
    int major = versionDriver / 1000;
    int minor = (versionDriver - major * 1000) / 10;
    std::cout << "Driver Version  = " << major << "." << minor << '\n';

    int versionRuntime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&versionRuntime));

    // The version is returned as (1000 * major + 10 * minor).
    major = versionRuntime / 1000;
    minor = (versionRuntime - major * 1000) / 10;
    std::cout << "Runtime Version = " << major << "." << minor << '\n';

    int countDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&countDevices));
    std::cout << "Device Count    = " << countDevices << '\n';

    for (int i = 0; i < countDevices; ++i)
    {
        cudaDeviceProp properties;

        CUDA_CHECK(cudaGetDeviceProperties(&properties, i));

        m_deviceProperties.push_back(properties);

        std::cout << "Device " << i << ": " << properties.name << '\n';
#if 1 // Condensed information
        std::cout << "  SM " << properties.major << "." << properties.minor << '\n';
        std::cout << "  Total Mem = " << properties.totalGlobalMem << '\n';
        std::cout << "  ClockRate [kHz] = " << properties.clockRate << '\n';
        std::cout << "  MaxThreadsPerBlock = " << properties.maxThreadsPerBlock << '\n';
        std::cout << "  SM Count = " << properties.multiProcessorCount << '\n';
        std::cout << "  Timeout Enabled = " << properties.kernelExecTimeoutEnabled << '\n';
        std::cout << "  TCC Driver = " << properties.tccDriver << '\n';
#else // Dump every property.
        // std::cout << "name[256] = " << properties.name << '\n';
        std::cout << "uuid = " << properties.uuid.bytes << '\n';
        std::cout << "totalGlobalMem = " << properties.totalGlobalMem << '\n';
        std::cout << "sharedMemPerBlock = " << properties.sharedMemPerBlock << '\n';
        std::cout << "regsPerBlock = " << properties.regsPerBlock << '\n';
        std::cout << "warpSize = " << properties.warpSize << '\n';
        std::cout << "memPitch = " << properties.memPitch << '\n';
        std::cout << "maxThreadsPerBlock = " << properties.maxThreadsPerBlock << '\n';
        std::cout << "maxThreadsDim[3] = " << properties.maxThreadsDim[0] << ", " << properties.maxThreadsDim[1] << ", " << properties.maxThreadsDim[0] << '\n';
        std::cout << "maxGridSize[3] = " << properties.maxGridSize[0] << ", " << properties.maxGridSize[1] << ", " << properties.maxGridSize[2] << '\n';
        std::cout << "clockRate = " << properties.clockRate << '\n';
        std::cout << "totalConstMem = " << properties.totalConstMem << '\n';
        std::cout << "major = " << properties.major << '\n';
        std::cout << "minor = " << properties.minor << '\n';
        std::cout << "textureAlignment = " << properties.textureAlignment << '\n';
        std::cout << "texturePitchAlignment = " << properties.texturePitchAlignment << '\n';
        std::cout << "deviceOverlap = " << properties.deviceOverlap << '\n';
        std::cout << "multiProcessorCount = " << properties.multiProcessorCount << '\n';
        std::cout << "kernelExecTimeoutEnabled = " << properties.kernelExecTimeoutEnabled << '\n';
        std::cout << "integrated = " << properties.integrated << '\n';
        std::cout << "canMapHostMemory = " << properties.canMapHostMemory << '\n';
        std::cout << "computeMode = " << properties.computeMode << '\n';
        std::cout << "maxTexture1D = " << properties.maxTexture1D << '\n';
        std::cout << "maxTexture1DMipmap = " << properties.maxTexture1DMipmap << '\n';
        std::cout << "maxTexture1DLinear = " << properties.maxTexture1DLinear << '\n';
        std::cout << "maxTexture2D[2] = " << properties.maxTexture2D[0] << ", " << properties.maxTexture2D[1] << '\n';
        std::cout << "maxTexture2DMipmap[2] = " << properties.maxTexture2DMipmap[0] << ", " << properties.maxTexture2DMipmap[1] << '\n';
        std::cout << "maxTexture2DLinear[3] = " << properties.maxTexture2DLinear[0] << ", " << properties.maxTexture2DLinear[1] << ", " << properties.maxTexture2DLinear[2] << '\n';
        std::cout << "maxTexture2DGather[2] = " << properties.maxTexture2DGather[0] << ", " << properties.maxTexture2DGather[1] << '\n';
        std::cout << "maxTexture3D[3] = " << properties.maxTexture3D[0] << ", " << properties.maxTexture3D[1] << ", " << properties.maxTexture3D[2] << '\n';
        std::cout << "maxTexture3DAlt[3] = " << properties.maxTexture3DAlt[0] << ", " << properties.maxTexture3DAlt[1] << ", " << properties.maxTexture3DAlt[2] << '\n';
        std::cout << "maxTextureCubemap = " << properties.maxTextureCubemap << '\n';
        std::cout << "maxTexture1DLayered[2] = " << properties.maxTexture1DLayered[0] << ", " << properties.maxTexture1DLayered[1] << '\n';
        std::cout << "maxTexture2DLayered[3] = " << properties.maxTexture2DLayered[0] << ", " << properties.maxTexture2DLayered[1] << ", " << properties.maxTexture2DLayered[2] << '\n';
        std::cout << "maxTextureCubemapLayered[2] = " << properties.maxTextureCubemapLayered[0] << ", " << properties.maxTextureCubemapLayered[1] << '\n';
        std::cout << "maxSurface1D = " << properties.maxSurface1D << '\n';
        std::cout << "maxSurface2D[2] = " << properties.maxSurface2D[0] << ", " << properties.maxSurface2D[1] << '\n';
        std::cout << "maxSurface3D[3] = " << properties.maxSurface3D[0] << ", " << properties.maxSurface3D[1] << ", " << properties.maxSurface3D[2] << '\n';
        std::cout << "maxSurface1DLayered[2] = " << properties.maxSurface1DLayered[0] << ", " << properties.maxSurface1DLayered[1] << '\n';
        std::cout << "maxSurface2DLayered[3] = " << properties.maxSurface2DLayered[0] << ", " << properties.maxSurface2DLayered[1] << ", " << properties.maxSurface2DLayered[2] << '\n';
        std::cout << "maxSurfaceCubemap = " << properties.maxSurfaceCubemap << '\n';
        std::cout << "maxSurfaceCubemapLayered[2] = " << properties.maxSurfaceCubemapLayered[0] << ", " << properties.maxSurfaceCubemapLayered[1] << '\n';
        std::cout << "surfaceAlignment = " << properties.surfaceAlignment << '\n';
        std::cout << "concurrentKernels = " << properties.concurrentKernels << '\n';
        std::cout << "ECCEnabled = " << properties.ECCEnabled << '\n';
        std::cout << "pciBusID = " << properties.pciBusID << '\n';
        std::cout << "pciDeviceID = " << properties.pciDeviceID << '\n';
        std::cout << "pciDomainID = " << properties.pciDomainID << '\n';
        std::cout << "tccDriver = " << properties.tccDriver << '\n';
        std::cout << "asyncEngineCount = " << properties.asyncEngineCount << '\n';
        std::cout << "unifiedAddressing = " << properties.unifiedAddressing << '\n';
        std::cout << "memoryClockRate = " << properties.memoryClockRate << '\n';
        std::cout << "memoryBusWidth = " << properties.memoryBusWidth << '\n';
        std::cout << "l2CacheSize = " << properties.l2CacheSize << '\n';
        std::cout << "maxThreadsPerMultiProcessor = " << properties.maxThreadsPerMultiProcessor << '\n';
        std::cout << "streamPrioritiesSupported = " << properties.streamPrioritiesSupported << '\n';
        std::cout << "globalL1CacheSupported = " << properties.globalL1CacheSupported << '\n';
        std::cout << "localL1CacheSupported = " << properties.localL1CacheSupported << '\n';
        std::cout << "sharedMemPerMultiprocessor = " << properties.sharedMemPerMultiprocessor << '\n';
        std::cout << "regsPerMultiprocessor = " << properties.regsPerMultiprocessor << '\n';
        std::cout << "managedMemory = " << properties.managedMemory << '\n';
        std::cout << "isMultiGpuBoard = " << properties.isMultiGpuBoard << '\n';
        std::cout << "multiGpuBoardGroupID = " << properties.multiGpuBoardGroupID << '\n';
        std::cout << "singleToDoublePrecisionPerfRatio = " << properties.singleToDoublePrecisionPerfRatio << '\n';
        std::cout << "pageableMemoryAccess = " << properties.pageableMemoryAccess << '\n';
        std::cout << "concurrentManagedAccess = " << properties.concurrentManagedAccess << '\n';
        std::cout << "computePreemptionSupported = " << properties.computePreemptionSupported << '\n';
        std::cout << "canUseHostPointerForRegisteredMem = " << properties.canUseHostPointerForRegisteredMem << '\n';
        std::cout << "cooperativeLaunch = " << properties.cooperativeLaunch << '\n';
        std::cout << "cooperativeMultiDeviceLaunch = " << properties.cooperativeMultiDeviceLaunch << '\n';
        std::cout << "pageableMemoryAccessUsesHostPageTables = " << properties.pageableMemoryAccessUsesHostPageTables << '\n';
        std::cout << "directManagedMemAccessFromHost = " << properties.directManagedMemAccessFromHost << '\n';
#endif
    }
}

}