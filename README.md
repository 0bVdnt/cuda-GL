# CUDA + OpenGL Project

This is an example project demonstrating CUDA-OpenGL interoperability.

## Prerequisites
1.  An NVIDIA GPU with the latest drivers.
2.  Visual Studio 2022 with the "Desktop development with C++" workload.
3.  The latest **NVIDIA CUDA Toolkit**.

# Sample Output
![cuda-GL](./Assets/cuda-GL_inaction.gif)

## Setup

1.  Clone this repository.
2.  Download **GLFW** (64-bit pre-compiled binaries) and extract it.
3.  Download **GLEW** (binaries) and extract it.
4.  Open the `cuGL.vcxproj` file in a text editor and update the paths in the `<AdditionalIncludeDirectories>` and `<AdditionalLibraryDirectories>` sections to point to where you extracted GLEW and GLFW.
5.  Open the `.sln` file in Visual Studio and build.