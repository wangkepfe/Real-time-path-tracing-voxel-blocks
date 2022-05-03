#include "core/Backend.h"
#include "core/Options.h"
#include "core/Application.h"
#include "core/DebugUtils.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace jazzfusion {

static void errorCallback(int error, const char* description)
{
    std::cerr << "Error: "<< error << ": " << description << '\n';
}

void Backend::init()
{
    m_width = 1920;
    m_height = 1080;

    glfwSetErrorCallback(errorCallback);
    if (!glfwInit())
    {
        throw std::runtime_error("GLFW failed to initialize.");
    }

    m_window = glfwCreateWindow(m_width, m_height, "JazzFusion Renderer", NULL, NULL);
    if (!m_window)
    {
        throw std::runtime_error("glfwCreateWindow() failed.");
    }

    glfwMakeContextCurrent(m_window);
    if (glewInit() != GL_NO_ERROR)
    {
        throw std::runtime_error("GLEW failed to initialize.");
    }

    // Initialize DevIL once.
    ilInit();

    initOpenGL();
    initInterop();
}

void Backend::mainloop()
{
    auto& app = Application::Get();

    while (!glfwWindowShouldClose(m_window))
    {
        glfwPollEvents();

        reshape();

        mapInteropBuffer();
        app.m_systemParameter.outputBuffer = m_interopBuffer;
        app.render();
        unmapInteropBuffer();


        app.guiNewFrame();
        app.guiWindow();
        app.guiEventHandler();

        display();

        app.guiRender();

        glfwSwapBuffers(m_window);
    }
}

void Backend::clear()
{
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
        "#version 330                                                                                  \n"
        "uniform sampler2D samplerHDR;                                                                 \n"
        "uniform vec3  colorBalance;                                                                   \n"
        "uniform float invWhitePoint;                                                                  \n"
        "uniform float burnHighlights;                                                                 \n"
        "uniform float saturation;                                                                     \n"
        "uniform float crushBlacks;                                                                    \n"
        "uniform float invGamma;                                                                       \n"
        "in vec2 varTexCoord;                                                                          \n"
        "layout(location = 0, index = 0) out vec4 outColor;                                            \n"
        "void main()                                                                                   \n"
        "{                                                                                             \n"
        "    vec3 hdrColor = texture(samplerHDR, varTexCoord).rgb;                                     \n"
        "    vec3 ldrColor = invWhitePoint * colorBalance * hdrColor;                                  \n"
        "    ldrColor *= (ldrColor * burnHighlights + 1.0) / (ldrColor + 1.0);                         \n"
        "    float luminance = dot(ldrColor, vec3(0.3, 0.59, 0.11));                                   \n"
        "    ldrColor = max(mix(vec3(luminance), ldrColor, saturation), 0.0);                          \n"
        "    luminance = dot(ldrColor, vec3(0.3, 0.59, 0.11));                                         \n"
        "    if (luminance < 1.0)                                                                      \n"
        "    {                                                                                         \n"
        "        ldrColor = max(mix(pow(ldrColor, vec3(crushBlacks)), ldrColor, sqrt(luminance)), 0.0);\n"
        "    }                                                                                         \n"
        "    ldrColor = pow(ldrColor, vec3(invGamma));                                                 \n"
        "    outColor = vec4(ldrColor, 1.0);                                                           \n"
        "}                                                                                             \n";

    GLint vsCompiled = 0;
    GLint fsCompiled = 0;

    m_glslVS = glCreateShader(GL_VERTEX_SHADER);
    if (m_glslVS)
    {
        GLsizei len = (GLsizei)vsSource.size();
        const GLchar *vs = vsSource.c_str();
        glShaderSource(m_glslVS, 1, &vs, &len);
        glCompileShader(m_glslVS);

        glGetShaderiv(m_glslVS, GL_COMPILE_STATUS, &vsCompiled);
        assert(vsCompiled);
    }

    m_glslFS = glCreateShader(GL_FRAGMENT_SHADER);
    if (m_glslFS)
    {
        GLsizei len = (GLsizei)fsSource.size();
        const GLchar *fs = fsSource.c_str();
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

            // TODO: move to tone mapper
            m_gamma = 2.2f;
            m_colorBalance = make_float3(1.0f, 1.0f, 1.0f);
            m_whitePoint = 1.0f;
            m_burnHighlights = 0.8f;
            m_crushBlacks = 0.2f;
            m_saturation = 1.2f;
            m_brightness = 0.8f;

            glUniform1i(glGetUniformLocation(m_glslProgram, "samplerHDR"), 0); // Always using texture image unit 0 for the display texture.
            glUniform1f(glGetUniformLocation(m_glslProgram, "invGamma"), 1.0f / m_gamma);
            glUniform3f(glGetUniformLocation(m_glslProgram, "colorBalance"), m_colorBalance.x, m_colorBalance.y, m_colorBalance.z);
            glUniform1f(glGetUniformLocation(m_glslProgram, "invWhitePoint"), m_brightness / m_whitePoint);
            glUniform1f(glGetUniformLocation(m_glslProgram, "burnHighlights"), m_burnHighlights);
            glUniform1f(glGetUniformLocation(m_glslProgram, "crushBlacks"), m_crushBlacks + m_crushBlacks + 1.0f);
            glUniform1f(glGetUniformLocation(m_glslProgram, "saturation"), m_saturation);

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
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)sizeof(float) * 16, (GLvoid const *)attributes, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vboIndices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)sizeof(unsigned int) * 6, (const GLvoid *)indices, GL_STATIC_DRAW);

    glVertexAttribPointer(m_positionLocation, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (GLvoid *)0);
    glVertexAttribPointer(m_texCoordLocation, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (GLvoid *)(sizeof(float) * 2));
}

void Backend::display()
{
    // Bind texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_hdrTexture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei)m_width, (GLsizei)m_height, 0, GL_RGBA, GL_FLOAT, (void *)0);

    // Bind buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_vboAttributes);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vboIndices);
    glEnableVertexAttribArray(m_positionLocation);
    glEnableVertexAttribArray(m_texCoordLocation);

    glUseProgram(m_glslProgram);

    glDrawElements(GL_TRIANGLES, (GLsizei)6, GL_UNSIGNED_INT, (const GLvoid *)0);

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
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&m_interopBuffer, &size, m_cudaGraphicsResource));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream));

    assert(m_width * m_height * sizeof(float) * 4 <= size);
}

void Backend::mapInteropBuffer()
{
    size_t size;
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&m_interopBuffer, &size, m_cudaGraphicsResource));
}

void Backend::unmapInteropBuffer()
{
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream));
}

void Backend::reshape()
{
    int width;
    int height;
    glfwGetFramebufferSize(m_window, &width, &height);

    if ((width != 0 && height != 0) && (m_width != width || m_height != height))
    {
        m_width = width;
        m_height = height;

        glViewport(0, 0, m_width, m_height);

        CUDA_CHECK(cudaGraphicsUnregisterResource(m_cudaGraphicsResource));

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cudaGraphicsResource, m_pbo, cudaGraphicsRegisterFlagsNone));

        size_t size;

        CUDA_CHECK(cudaGraphicsMapResources(1, &m_cudaGraphicsResource, m_cudaStream));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&m_interopBuffer, &size, m_cudaGraphicsResource));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource, m_cudaStream));

        assert(m_width * m_height * sizeof(float) * 4 <= size);


        auto& app = Application::Get();
        app.m_width = width;
        app.m_height = height;
        app.m_systemParameter.outputBuffer = m_interopBuffer;
        app.m_pinholeCamera.setViewport(m_width, m_height);
        app.restartAccumulation();
    }
}

}