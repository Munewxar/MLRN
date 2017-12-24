#include <armadillo>
#include "CImg.h"
#include <iostream>
#include <vector>
#include <cmath>

void lNetwork();
void doNecessaryOps(int curObj);
void adjWmatrix();
double bColPixel(double number);
double nColor(double number);
void bNewImg(int w, int h);
void bWmatrix();
void dIntoRects(cimg_library::CImg<double> img);
void getInfo();

int n, m, p, e, rgb = 3;
double alpha;
arma::mat X, _X, Y, dX, W, _W;
std::vector<arma::mat> commonVector;

int main()
{
	cimg_library::CImg<double> img("icon.bmp");

	dIntoRects(img);
	bWmatrix();
	lNetwork();
	bNewImg(img.width(), img.height());

	getchar();
	return 0;
}

void lNetwork()
{
	double E;
	int it = 1, N = n*m*rgb;

	do {
		E = 0;

		for (int i = 0; i < commonVector.size(); i++) {
			doNecessaryOps(i);
			adjWmatrix();
		}

		for (int i = 0; i < commonVector.size(); i++) {
			doNecessaryOps(i);

			for (int j = 0; j < N; j++) {
				E += pow(dX(0, j), 2);
			}
		}

		std::cout << "current iteration: " << it << "\tcurrent mistake: " << E << std::endl;
		it++;
	} while (E > e);

	double z = (N*commonVector.size()*1.0) / ((N + commonVector.size())*p + 2);
	std::cout << "compression ratio = " << z << std::endl;
}

void doNecessaryOps(int curObj)
{
	X = commonVector[curObj];
	Y = X * W;
	_X = Y * _W;
	dX = _X - X;
}

void adjWmatrix()
{
	W = W - alpha * X.t() * dX * _W.t();
	_W = _W - alpha * Y.t() * dX;
}

double bColPixel(double number) 
{
	double pixel = (number + 1) / 2 * 255;

	if (pixel < 0)
		pixel = 0;
	if (pixel > 255)
		pixel = 255;

	return pixel;
}

double nColor(double number)
{
	double valCol = 2 * number / 255 - 1;
	return valCol;
}

void bNewImg(int w, int h)
{
	cimg_library::CImg<double> img(w, h, 1, 3);
	int s = 0;

	for (int i = 0; i < w; i += m) {
		for (int j = 0; j < h; j += n) {
			X = commonVector[s];
			s++;
			Y = X * W;
			_X = Y * _W;
			int in = 0;

			for (int k = i; k < i + m; k++) {
				for (int s = j; s < j + n; s++)
				{
					img(k, s, 0, 0) = bColPixel(_X(0, in++));
					img(k, s, 0, 1) = bColPixel(_X(0, in++));
					img(k, s, 0, 2) = bColPixel(_X(0, in++));
				}
			}
		}
	}
	img.display();
	img.save("outImg.bmp");
}

void bWmatrix()
{
	W.set_size(n*m*rgb, p);
	for (int i = 0; i < n*m*rgb; i++)
		for (int j = 0; j < p; j++) {
			//double s = (double)(rand()) / RAND_MAX * 2 - 1;
			double s = (double)((rand() % 100 / 100 * 2) - 1);
			W(i, j) = 0.1 * s;
		}
	_W = W.t();
}

void dIntoRects(cimg_library::CImg<double> img)
{
	int in = 0;
	for (int i = 0; i < img.width(); i += m) {
		for (int j = 0; j < img.height(); j += n) {
			arma::mat matrix;
			matrix.set_size(1, n*m*rgb);
			in = 0;
			for (int k = i; k < i + m; k++) {
				for (int s = j; s < j + n; s++) {
					double cRed = img(k, s, 0, 0);
					matrix(0, in++) = nColor(cRed);

					double cGreen = img(k, s, 0, 1);
					matrix(0, in++) = nColor(cGreen);

					double cBlue = img(k, s, 0, 2);
					matrix(0, in++) = nColor(cBlue);
				}
			}
			commonVector.push_back(matrix);
		}
	}

}

void getInfo()
{
	std::cout << "Enter number of rows, columns, amount of neirons, max mistake, learning factor: ";
	std::cin >> m >> n >> p >> e >> alpha;
}