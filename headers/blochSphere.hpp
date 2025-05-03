/**
* @file blochSphere.hpp
*/
#ifndef BLOCHSPHERE_HPP
#define BLOCHSPHERE_HPP

#include <GL/glut.h>
#include <cmath>

/**
 * @brief The angle of rotation around the X-axis.
 * This variable represents the current rotation angle of the Bloch sphere around the X-axis.
 */
extern float angleX ;
/**
 * @brief The angle of rotation around the Z-axis.
 * This variable represents the current rotation angle of the Bloch sphere around the Z-axis.
 */
extern float angleY ;
/**
 * @brief The last recorded mouse X position during a drag event.
 * This variable is used to track the mouse's horizontal position to enable interactive rotation of the Bloch sphere.
 */
extern float angleZ ;
/**
 * @brief The last recorded mouse X position during a drag event.
 * 
 * This variable is used to track the mouse's horizontal position to enable interactive rotation of the Bloch sphere.
 */
extern int lastMouseX;
/**
 * @brief The last recorded mouse Y position during a drag event.
 * This variable is used to track the mouse's vertical position to enable interactive rotation of the Bloch sphere.
 */
extern int lastMouseY;
/**
 * @brief A flag to indicate whether the mouse is currently being dragged.
 * This boolean is set to true when the mouse button is pressed and dragged, allowing the rotation of the Bloch sphere.
 */
extern bool isDragging;
/**
 * @brief The polar angle (theta) of the vector on the Bloch sphere.
 * This variable determines the latitude of the vector on the Bloch sphere, defining its position in spherical coordinates.
 */
extern float theta;
/**
 * @brief The azimuthal angle (phi) of the vector on the Bloch sphere.
 * This variable determines the longitude of the vector on the Bloch sphere, defining its position in spherical coordinates.
 */
extern float phi;

void drawWireSphere(float radius, int slices, int stacks); 
void drawText(const char* text);
void drawAxes(); 
void drawAxisLabels(); 
void drawBoldArrow(float theta, float phi); 
void renderScene();
void keyboard(unsigned char key, int x, int y); 
void mouseMotion(int x, int y);
void mouseClick(int button, int state, int x, int y); 
void initOpenGL();
void bloch_sphere(int argc, char** argv); 
#endif
