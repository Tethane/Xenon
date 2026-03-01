#define GL_GLEXT_PROTOTYPES
#include "display/window.h"
#include <GLFW/glfw3.h>
#include <cstdio>
#include <vector>

namespace xn {

static const char* v_shader_src = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
)";

static const char* f_shader_src = R"(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D screenTexture;
    void main() {
        // Simple gamma correction (approx 2.2)
        vec3 col = texture(screenTexture, TexCoord).rgb;
        FragColor = vec4(pow(col, vec3(1.0/2.2)), 1.0);
    }
)";

Window::Window(int width, int height, const std::string& title)
    : width_(width), height_(height) {
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to init GLFW\n");
        return;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window_ = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
    if (!window_) {
        std::fprintf(stderr, "Failed to create GLFW window\n");
        return;
    }

    glfwMakeContextCurrent(window_);
    
    // Setup simple full-screen quad shader
    auto compile = [](unsigned int type, const char* src) {
        unsigned int s = glCreateShader(type);
        glShaderSource(s, 1, &src, NULL);
        glCompileShader(s);
        return s;
    };
    unsigned int vs = compile(GL_VERTEX_SHADER, v_shader_src);
    unsigned int fs = compile(GL_FRAGMENT_SHADER, f_shader_src);
    shader_program_ = glCreateProgram();
    glAttachShader(shader_program_, vs);
    glAttachShader(shader_program_, fs);
    glLinkProgram(shader_program_);
    glDeleteShader(vs);
    glDeleteShader(fs);

    setup_quad();

    // Create texture for blitting
    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

Window::~Window() {
    if (window_) glfwDestroyWindow(window_);
    glfwTerminate();
}

bool Window::should_close() const {
    return glfwWindowShouldClose(window_);
}

void Window::poll_events() {
    glfwPollEvents();
}

void Window::display(TripleSwapchain& swapchain) {
    float* data = swapchain.get_read_buffer();
    
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGB, GL_FLOAT, data);

    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shader_program_);
    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

void Window::swap_buffers() {
    glfwSwapBuffers(window_);
}

void Window::setup_quad() {
    float quad[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
}

} // namespace xn
