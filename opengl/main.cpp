#include <GLFW/glfw3.h>
#include <stdlb.h>
#include <stdio.h>

int main(void) {
  GLFWwindow* window;
  if(!glfwInit())
    exit(EXIT_FAILURE);
  window = glfwCreateWindow(640,480, "Chapter 1: Simple GLFW example", NULL, NULL);
  if(!window) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }
  glfwMakeContextCurrent(window);

}
