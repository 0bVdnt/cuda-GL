#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <vector>

#include "kernel.h" // CUDA kernel wrapper

// Window dimensions
const int WIDTH = 1024;
const int HEIGHT = 768;
const int GRID_W = 128;
const int GRID_H = 128;
const int NUM_POINTS = GRID_W * GRID_H;

int main() {
	// 1. Initialize GLFW
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return -1;
	}

	// 2. Create a window
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "CUDA + OpenGL Interop", NULL, NULL);
	if (!window) {
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// 3. Initialize GLEW
	if (glewInit() != GLEW_OK) {
		std::cerr << "Failed to initialize GLEW" << std::endl;
		return -1;
	}

	// 4. Initialize CUDA and link it to the OpenGL context
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0); // Use the first CUDA-capable device
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaSetDevice failed! Check if the GPU is CUDA-capable ?" << std::endl;
		return -1;
	}

	// 5. Create the Vertex Buffer Object (VBO) in OpenGL
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	// Allocate memory for the VBO
	unsigned int size = NUM_POINTS * sizeof(float4);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// 6. Register the VBO with CUDA
	struct cudaGraphicsResource* cuda_vbo_resource;
	cudaStatus = cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsRegisterFlagsNone);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaGraphicsGLRegisterBuffer failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		return -1;
	}

	// Create initial positions
	std::vector<float4> initial_positions(NUM_POINTS);
	for (int j = 0; j < GRID_H; ++j) {
		for (int i = 0; i < GRID_W; ++i) {
			float u = i / (float)(GRID_W - 1);
			float v = j / (float)(GRID_H - 1);
			initial_positions[j * GRID_W + i] = make_float4(
				(u * 2.0f - 1.0f), // x in [-1, 1]
				(v * 2.0f - 1.0f), // y in [-1, 1]
				0.0f,              // z (will be used for color)
				1.0f               // w
			);
		}
	}
	// Copy initial data to VBO
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferSubData(GL_ARRAY_BUFFER, 0, size, initial_positions.data());
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	// 7. Render Loop
	while (!glfwWindowShouldClose(window)) {
		// CUDA : Update the VBO data
		float4* d_ptr; // CUDA device pointer

		// Map the VBO for writing by CUDA
		cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
		size_t num_bytes;
		cudaGraphicsResourceGetMappedPointer((void**)&d_ptr, &num_bytes, cuda_vbo_resource);

		// Call the CUDA kernel to modify the data
		launch_kernel(d_ptr, GRID_W, GRID_H, (float)glfwGetTime());

		// Unmap the VBO
		cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

		// OpenGL : Render the VBO
		glClear(GL_COLOR_BUFFER_BIT);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableClientState(GL_VERTEX_ARRAY);
		// Passing the color through the z-coordinate for simplicity
		glEnableClientState(GL_COLOR_ARRAY);

		// x, y from float4
		glVertexPointer(2, GL_FLOAT, sizeof(float4), (void*)0);
		// z (red), w (unused) from float4 for color. Make it (z, z, z) for grayscale.
		glColorPointer(3, GL_FLOAT, sizeof(float4), (void*)(sizeof(float) * 2));

		glDrawArrays(GL_POINTS, 0, NUM_POINTS);

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);

		// Swap buffers and poll events
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// 8. Cleanup
	cudaGraphicsUnregisterResource(cuda_vbo_resource);
	glDeleteBuffers(1, &vbo);
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}