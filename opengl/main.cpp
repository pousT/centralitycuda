#include <GLFW/glfw3.h>
#include <stdlb.h>
#include <stdio.h>

int main(void) {
  //init window
  GLFWwindow* window;
  if(!glfwInit())
    exit(EXIT_FAILURE);
  window = glfwCreateWindow(640,480, "Chapter 1: Simple GLFW example", NULL, NULL);
  if(!window) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }
  glfwMakeContextCurrent(window);
  // while not closed
  while(!glfwWindowShouldClose(window)) {
    //set up view
    float ratio;
    int width, height;

    glfwGetFrameBufferSize(window, &width, &height);
    ratio = (float)width/(float)height;
    glViewport(0,0, width, height);
    glClear(GL_COLOR_BUFFER_BIT);
    //set up camera matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // draw a rotating triangle
    glRotatef((float)glfwGetTime() * 50.f, 0.f, 0.f, 1.f);
    glBegin(GL_TRIANGLES);
    glColor3f(1.f, 0.f, 0.f);
    glVertex3f(-0.6f, -0.4f, 0.f);
    glColor3f(0.f, 1.f, 0.f);
    glVertex3f(0.6f, -0.4f, 0.f);
    glColor3f(0.f, 0.f, 1.f);
    glVertex3f(0.f, 0.6f, 0.f);
    glEnd();

    glfwSwapBuffers(window);
    glfwPollEvents();


  }
     glfwDestroyWindow(window);
     glfwTerminate();
     exit(EXIT_SUCCESS);
}
