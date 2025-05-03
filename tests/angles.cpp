#ifndef TEST_ANGLES_CPP
#define TEST_ANGLES_CPP

#include <sycl/ext/intel/fpga_extensions.hpp>
#include "../headers/kernels.hpp"
#include "../headers/blochSphere.hpp"

// Variables to store rotation angles
float angleX = -45.0f;  // Start viewpoint in front of North and South poles
float angleY = -20.0f;
float angleZ = 0.0f;
int lastMouseX, lastMouseY;
bool isDragging = false;

// Default: Draw a bold black arrow (example theta=45 degrees, phi=60 degrees)
float theta = M_PI / 4;  // 45 degrees in radians
float phi = M_PI / 3;    // 60 degrees in radians

/*
 * @brief compute Bloch sphere'angles (theta and phi) corresponding to a given state (i.e. alpha|0> + beta|1>)
 * @param alpha The amplitude of state |0>
 * @param bete The amplitude of state |1>
 * @param theta_out The computed theta angle on the Bloch Sphere
 * @param phi_out The computed phi angle on the Bloch Sphere  
 *
 */
void state2angle(Complex<float> alpha, Complex<float> beta, float & theta_out, float & phi_out){
	float magnitude_alpha = Absol_v(alpha);
	float magnitude_beta = Absol_v(beta);

	// Ensure the state is normalized (alpha^2 + beta^2 = 1)
	double norm = std::sqrt(magnitude_alpha * magnitude_alpha + magnitude_beta * magnitude_beta);
	if (std::abs(norm - 1.0) > 1e-6) {
		std::cerr << "Warning: The state is not normalized!" << std::endl;
	}

	// Compute the polar angle (theta)
	theta_out = 2 * std::acos(magnitude_alpha);

	// Compute the azimuthal angle (phi)
	if (Absol_v(magnitude_alpha) > 1e-6) {  // To avoid division by zero when alpha is 0
		phi_out = Arg_v(beta) - Arg_v(alpha);
	} else {
		phi_out = Arg_v(beta);  // If alpha is real, just use the argument of beta
	}
	// Normalize phi to be in the range [0, 2*pi]
	if (phi_out < 0) {
		phi_out += 2 * M_PI;
	}
}

#endif
