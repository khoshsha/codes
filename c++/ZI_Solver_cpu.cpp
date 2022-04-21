
#include <iostream>
#include <stdio.h>
#include <string>
#include <math.h>
using namespace std;

int solver(float topography[][4], int nx, int ny, float rain, double maxt) {
	double nw = 0.01, A = 1.0, dt = 0.1, dw = 1.0, t = 0.0;
	rain = rain / 3600000.0;
	cout << "nx=" << nx << " ny=" << ny << endl;
	double oldh[nx][ny];
	double newh[nx][ny];
	double zwx[nx][ny], zwy[nx][ny];
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			oldh[i][j] = 0.0;
			newh[i][j] = 0.0;
			zwx[i][j] = 0.0;
			zwy[i][j] = 0.0;
		}
	}

	while (t <= maxt) {
		cout << "t=" << t << endl;

		//gradients row to row
		for (int i = 0; i < nx; i++) {
			for (int j = 0; j < (ny - 1); j++) {
				zwx[i][j] = (topography[i][j + 1] + oldh[i][j + 1] - topography[i][j] - oldh[i][j]) / (dw);
			}
		}

		//gradients column to column
		for (int i = 0; i < (nx - 1); i++) {
			for (int j = 0; j < ny; j++) {
				zwy[i][j] = (topography[i + 1][j] + oldh[i + 1][j] - topography[i][j] - oldh[i][j]) / (dw);
			}
		}




		float gradient = 0.0, up = 0.0, down = 0.0, right = 0.0, left = 0.0;
		float p = 1.666;
		for (int i = 0; i < nx; i++) {
			for (int j = 0; j < ny; j++) {

				//up neg
				if (i != 0 && zwx[i - 1][j] != 0) {
					up = ((pow(oldh[i][j], p))) * (zwx[i - 1][j] / (nw * sqrt(abs(zwx[i - 1][j]))));
				}
				else {
					up = 0.0;
				}

				//down pos
				if (i != nx && zwx[i][j] != 0) {

					down = ((pow(oldh[i][j], p))) * zwx[i][j] / (nw * sqrt(abs(zwx[i][j])));

				}
				else {
					down = 0.0;
				}

				//right pos
				if (j != ny && zwy[i][j] != 0) {
					right = ((pow(oldh[i][j], p))) * zwy[i][j] / (nw * sqrt(abs(zwy[i][j])));
				}
				else {
					right = 0.0;
				}

				//left pos
				if (j != 0 && zwy[i][j - 1] != 0) {
					left = ((pow(oldh[i][j], p))) * zwy[i][j - 1] / (nw * sqrt(abs(zwy[i][j - 1])));
				}
				else {
					left = 0.0;
				}

				gradient = -up + down + right - left;

				newh[i][j] = oldh[i][j] + (dt * rain) - ((dt / A) * gradient);
			}
		}
		for (int i = 0; i < nx; i++) {
			for (int j = 0; j < ny; j++) {
				cout << newh[i][j] << " ";
				oldh[i][j] = newh[i][j];
			}
			cout << "\n";
		}
		t = t + dt;
	}
	return 0;
}

int main() {
	float r = 1.0;
	float base[3][4] = { {3.0,3.0,3.0,3.0}, {3.0,2.0,2.0,3.0} , {2.0,1.0,1.0,3.0} };
	int nrow = 3, ncol = 4;
	double time = 1;

	solver(base, nrow, ncol, r, time);


	return 0;
}

