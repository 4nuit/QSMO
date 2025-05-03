/**
* @file blochSphere.cpp
*/

#include "../headers/blochSphere.hpp"

/**
 * @brief Draws a wireframe sphere using OpenGL's GLUquadric object.
 * 
 * This function creates a wireframe sphere with the specified radius, slices, and stacks, and renders it in the scene.
 * 
 * @param radius The radius of the sphere.
 * @param slices The number of longitudinal slices for the sphere.
 * @param stacks The number of latitudinal stacks for the sphere.
 */
void drawWireSphere(float radius, int slices, int stacks) {
    GLUquadric* quad = gluNewQuadric();
    gluQuadricDrawStyle(quad, GLU_LINE); // Set wireframe style
    gluSphere(quad, radius, slices, stacks);
    gluDeleteQuadric(quad);
}

/**
 * @brief Draws a string of text at the current raster position.
 * This function renders each character of the input string on the screen using the GLUT bitmap font. 
 * @param text The string of text to be rendered.
 */
void drawText(const char* text) {
    while (*text) {
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, *text);
        text++;
    }
}

/**
 * @brief Draws the X, Y, and Z axes of the Bloch sphere.
 * This function uses OpenGL lines to draw the coordinate axes (X, Y, Z) in three different colors (red, green, blue).
 */
void drawAxes() {
    glBegin(GL_LINES);
    
    // X-axis (red)
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(1.0f, 0.0f, 0.0f);
    
    // Y-axis (green)
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, -1.0f, 0.0f);
    glVertex3f(0.0f, 1.0f, 0.0f);
    
    // Z-axis (blue)
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, -1.0f);
    glVertex3f(0.0f, 0.0f, 1.0f);
    
    glEnd();
}

/**
 * @brief Draws labels for the X, Y, and Z axes of the Bloch sphere.
 * This function renders labels for the axes, such as "|+>", "|->", "|0>", and "|1>", at the respective axes in the OpenGL scene.
 */
void drawAxisLabels() {
    // X-axis label
    glColor3f(1.0f, 0.0f, 0.0f);  // Red for X-axis
    glRasterPos3f(1.1f, 0.0f, 0.0f);  // Position for X label
    drawText("|+>");

    glColor3f(1.0f, 0.0f, 0.0f);  // Red for X-axis
    glRasterPos3f(-1.1f, 0.0f, 0.0f);  // Position for X label
    drawText("|->");
    
    // Y-axis label
    glColor3f(0.0f, 1.0f, 0.0f);  // Green for Y-axis
    glRasterPos3f(0.0f, 1.1f, 0.0f);  // Position for Y label
    glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, 'Y');
    
    // Z-axis label
    glColor3f(0.0f, 0.0f, 1.0f);  // Blue for Z-axis
    glRasterPos3f(0.0f, 0.0f, 1.1f);  // Position for Z label
    drawText("|0>");
    
    // Z-axis label
    glColor3f(0.0f, 0.0f, 1.0f);  // Blue for Z-axis
    glRasterPos3f(0.0f, 0.0f, -1.1f);  // Position for Z label
    drawText("|1>");
}

/**
 * @brief Draws a bold black arrow on the Bloch sphere.
 * This function renders an arrow from the origin (0, 0, 0) to a point on the Bloch sphere determined by the spherical coordinates (theta, phi).
 * The arrow is rendered with a thick line and an optional arrowhead at the tip.
 * @param theta The polar angle of the arrow in spherical coordinates.
 * @param phi The azimuthal angle of the arrow in spherical coordinates.
 */
void drawBoldArrow(float theta, float phi) {
    // Convert spherical coordinates to Cartesian for arrow tip
    float x = sin(theta) * cos(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(theta);

    // Set color to black for the arrow
    glColor3f(0.0f, 0.0f, 0.0f);
    
    // Make the arrow bold by increasing line width
    glLineWidth(3.0f);
    
    // Draw the arrow from origin (0, 0, 0) to (x, y, z)
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);  // Arrow start (origin)
    glVertex3f(x, y, z);            // Arrow end (tip)
    glEnd();
    
    // Optionally, draw the arrow head
    glPointSize(10.0f);
    glBegin(GL_POINTS);
    glVertex3f(x, y, z);  // Draw point at arrow tip to make it more visible
    glEnd();
}

/**
 * @brief Renders the entire OpenGL scene, including the Bloch sphere, axes, and arrow.
 * This function is called every time the scene needs to be redrawn, applying any rotation transformations and rendering the elements of the Bloch sphere scene.
 */
void renderScene() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    
    // Move the scene back
    glTranslatef(0.0f, 0.0f, -5.0f);
    
    // Apply the rotation from the mouse interaction
    glRotatef(angleX, 1.0f, 0.0f, 0.0f);
    glRotatef(angleY, 0.0f, 1.0f, 0.0f);
    glRotatef(angleZ, 0.0f, 0.0f, 1.0f);
    
    // Draw the axes
    drawAxes();
    
    // Label the axes
    drawAxisLabels();
    
    // Draw the Bloch sphere (wireframe) with thin lines
    glColor3f(0.0f, 0.0f, 0.0f);  // Black for wireframe sphere
    glLineWidth(1.0f);  // Thin line for the wireframe
    drawWireSphere(1.0f, 20, 20);
    
    // Theta and phi are global variables
    drawBoldArrow(theta, phi);
    
    glutSwapBuffers();
}

/**
 * @brief Handles keyboard input for rotating the Bloch sphere.
 * This function updates the rotation angles for the sphere based on the keyboard input. The user can rotate the sphere by pressing the 'w', 'a', 's', or 'd' keys.
 * @param key The ASCII value of the pressed key.
 * @param x The x-coordinate of the mouse when the key is pressed.
 * @param y The y-coordinate of the mouse when the key is pressed.
 */
void keyboard(unsigned char key, int x, int y) {
    const float angleStep = 5.0f;
    if (key == 'w') angleX -= angleStep;  // Rotate upward
    if (key == 's') angleX += angleStep;  // Rotate downward
    if (key == 'a') angleY -= angleStep;  // Rotate left
    if (key == 'd') angleY += angleStep;  // Rotate right
    glutPostRedisplay();  // Redraw the scene
}

/**
 * @brief Handles mouse motion to rotate the Bloch sphere.
 * This function updates the rotation angles of the sphere as the mouse is dragged. It allows for interactive rotation by moving the mouse in the window.
 * @param x The new x-coordinate of the mouse.
 * @param y The new y-coordinate of the mouse.
 */
void mouseMotion(int x, int y) {
    if (isDragging) {
        angleY += (x - lastMouseX) * 0.2f;
        angleX += (y - lastMouseY) * 0.2f;
        lastMouseX = x;
        lastMouseY = y;
        glutPostRedisplay();  // Redraw the scene
    }
}

/**
 * @brief Handles mouse click events for starting or stopping the drag.
 * This function is triggered when the user clicks the mouse. It starts or stops the rotation of the sphere based on mouse dragging.
 * @param button The mouse button that was clicked.
 * @param state The state of the mouse button (pressed or released).
 * @param x The x-coordinate of the mouse at the time of the click.
 * @param y The y-coordinate of the mouse at the time of the click.
 */
void mouseClick(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        isDragging = true;
        lastMouseX = x;
        lastMouseY = y;
    } else if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
        isDragging = false;
    }
}

/**
 * @brief Initializes OpenGL settings for the Bloch sphere rendering.
 * This function sets up OpenGL parameters, including enabling depth testing, setting the background color, and configuring the projection matrix for 3D rendering.
 */
void initOpenGL() {
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.97f, 0.91f, 0.81f, 1.0f);  // Champagne background
    glMatrixMode(GL_PROJECTION);
    gluPerspective(45.0, 1.0, 1.0, 10.0);
    glMatrixMode(GL_MODELVIEW);
}

/**
 * @brief Main function for setting up and running the Bloch sphere application using GLUT.
 * This function initializes GLUT, sets up the window, and enters the GLUT main loop, where the rendering and input handling takes place.
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line argument strings.
 */
void bloch_sphere(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 800);
    glutCreateWindow("Movable Wireframe Bloch Sphere with Arrow and Labels");
    initOpenGL();
    glutDisplayFunc(renderScene);
    glutKeyboardFunc(keyboard);  // Set keyboard callback
    glutMouseFunc(mouseClick);   // Set mouse click callback
    glutMotionFunc(mouseMotion); // Set mouse motion callback
    glutMainLoop();
}

