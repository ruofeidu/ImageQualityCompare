#define _CRT_SECURE_NO_WARNINGS
#ifdef _DEBUG 
#pragma comment(lib, "opencv_world331d.lib")
#else 
#pragma comment(lib, "opencv_world331.lib")
#endif
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

// height and width
int h[2], w[2];

// mat and grayscale
Mat m[2], g[2];

// file names
string s[2];

int block_size = 1;
bool mask = false; 

void quit() {
	system("pause");
	exit(EXIT_FAILURE);
}

#define C1 (float) (0.01 * 255 * 0.01  * 255)
#define C2 (float) (0.03 * 255 * 0.03  * 255)

// Sigma on block_size
double sigma(Mat & m, int i, int j, int block_size) {
	double sd = 0;

	Mat m_tmp = m(Range(i, i + block_size), Range(j, j + block_size));
	Mat m_squared(block_size, block_size, CV_64F);

	multiply(m_tmp, m_tmp, m_squared);

	// E(x)
	double avg = mean(m_tmp)[0];
	// E(x²)
	double avg_2 = mean(m_squared)[0];

	sd = sqrt(avg_2 - avg * avg);
	return sd;
}

// Covariance
double cov(Mat & m0, Mat & m1, int i, int j, int block_size) {
	Mat m3 = Mat::zeros(block_size, block_size, m0.depth());
	Mat m0_tmp = m0(Range(i, i + block_size), Range(j, j + block_size));
	Mat m1_tmp = m1(Range(i, i + block_size), Range(j, j + block_size));

	multiply(m0_tmp, m1_tmp, m3);

	double avg_ro = mean(m3)[0]; // E(XY)
	double avg_r = mean(m0_tmp)[0]; // E(X)
	double avg_o = mean(m1_tmp)[0]; // E(Y)

	double sd_ro = avg_ro - avg_o * avg_r; // E(XY) - E(X)E(Y)
	return sd_ro;
}

// Mean squared error
double mse(Mat & m0, Mat & m1, bool grayscale = false, bool rooted = false) {
	double res = 0;
	int H = m0.rows, W = m0.cols, blanks = 0;
#pragma omp parallel for
	for (int i = 0; i < H; i++)
		for (int j = 0; j < W; j++) {
			if (grayscale) {
				double p0 = m0.at<double>(i, j), p1 = m1.at<double>(i, j);
				if (mask) {
					if ((p0 > 254.0) || (p0 < 1.0)) {
						++blanks;
						continue;
					}
				}
				double diff = abs(p0 - p1);
				res += diff * diff;
			}
			else {
				Vec3b p0 = m0.at<Vec3b>(i, j);
				Vec3b p1 = m1.at<Vec3b>(i, j);
				if (mask) {
					if ((p0.val[0] > 254 && p0.val[1] > 254 && p0.val[2] > 254) || (p1.val[0] < 1 && p1.val[1] < 1 && p1.val[2] < 1)) {
						++blanks;
						continue;
					}
				}
				double d0 = abs(p0.val[0] - p1.val[0]);
				double d1 = abs(p0.val[1] - p1.val[1]);
				double d2 = abs(p0.val[2] - p1.val[2]);
				if (rooted) {
					res += sqrt(d0 * d0 + d1 * d1 + d2 * d2) / 255.0 / sqrt(3.0);
				}
				else {
					res += (d0 * d0 + d1 * d1 + d2 * d2) / 3.0;
				}
			}
		}
	res /= H * W - blanks;
	return res;
}

// Rooted mean squared error
double rmse(Mat & m0, Mat & m1) {
	double eqm = 0;
	int height = m0.rows;
	int width = m0.cols;
	int blanks = 0;
#pragma omp parallel for
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			Vec3b p0 = m0.at<Vec3b>(i, j);
			Vec3b p1 = m1.at<Vec3b>(i, j);
			if (mask) {
				if ((p0.val[0] > 254 && p0.val[1] > 254 && p0.val[2] > 254) || (p1.val[0] < 1 && p1.val[1] < 1 && p1.val[2] < 1)) {
					++blanks;
					continue;
				}
			}
			double diff = abs(p0.val[0] - p1.val[0]) + abs(p0.val[1] - p1.val[1]) + abs(p0.val[2] - p1.val[2]);
			eqm += (m0.at<double>(i, j) - m1.at<double>(i, j)) * (m0.at<double>(i, j) - m1.at<double>(i, j));
		}

	eqm /= height * width - blanks;
	return eqm;
}

// Peak signal-to-noise ratio
// https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
double psnr(Mat & m0, Mat & m1, int block_size) {
	int D = 255;
	return (10 * log10((D*D) / mse(m0, m1, true, false)));
}

double ssim(Mat & m0, Mat & m1, int block_size) {
	double ssim = 0;

	int nbBlockPerHeight = m0.rows / block_size;
	int nbBlockPerWidth = m0.cols / block_size;
	double ssim_total = 0;

#pragma omp parallel for
	for (int k = 0; k < nbBlockPerHeight; k++) {
		for (int l = 0; l < nbBlockPerWidth; l++) {
			int m = k * block_size;
			int n = l * block_size;

			double avg_o = mean(m0(Range(k, k + block_size), Range(l, l + block_size)))[0];
			double avg_r = mean(m1(Range(k, k + block_size), Range(l, l + block_size)))[0];
			if (mask) {
				if (avg_o > 254 || avg_o < 1) {
					continue;
				}
				ssim_total += 1;
			}
			double sigma_o = sigma(m0, m, n, block_size);
			double sigma_r = sigma(m1, m, n, block_size);
			double sigma_ro = cov(m0, m1, m, n, block_size);

			ssim += ((2 * avg_o * avg_r + C1) * (2 * sigma_ro + C2)) / ((avg_o * avg_o + avg_r * avg_r + C1) * (sigma_o * sigma_o + sigma_r * sigma_r + C2));
		}
	}

	ssim /= mask ? ssim_total : nbBlockPerHeight * nbBlockPerWidth;
	return ssim;
}

int main(int argc, char *argv[])
{
	vector<string> arguments(argv, argv + argc);
	if (argc < 3) {
		printf("Usage: Compare image_file_name_1 image_file_name_2 [--mask] [--block_size] 2\n");
		printf("Example: Compare a.jpg b.png\n");
		printf("Output: a_b.txt\n");
		quit();
	}

	for (int i = 0; i < 2; ++i) {
		s[i] = arguments[i + 1];
		m[i] = imread(s[i]);
		if (m[i].empty()) {
			printf("Error: Cannot read file %s", s[i].c_str());
			quit();
		}
		cvtColor(m[i], g[i], CV_BGR2GRAY);
		g[i].convertTo(g[i], CV_64F);
		s[i] = s[i].substr(0, s[i].find_last_of('.'));
		h[i] = m[i].rows;
		w[i] = m[i].cols;
	}

	if (argc > 3) {
		for (int i = 3; i < argc; ++i) {
			if (!arguments[i].compare("--mask")) {
				mask = true; 
			}
			if (!arguments[i].compare("--block_size")) {
				if ((i + 1) < argc) {
					block_size = atoi(arguments[i + 1].c_str());
				}
				else {
					printf("Error: Please specify the block size!\n");
					quit();
				}
			}
		}
	}

	if (h[0] != h[1] || w[0] != w[1])
	{
		printf("Error: Images are of different dimensions.\n");
		quit();
	}

	double rmse_val = mse(m[0], m[1], false, true);
	double ssim_val = ssim(g[0], g[1], block_size);
	double psnr_val = psnr(g[0], g[1], block_size);

	string txtFile = s[0] + "_" + s[1] + ".txt";
	FILE* file = fopen(txtFile.c_str(), "w");
	fprintf(file, "%.4f\n%.6f\n%.6f\n", rmse_val * 100, ssim_val, psnr_val);
	fprintf(file, "# RMSE(%%), SSIM, and PSNR (dB).\n");
	return 0; 
}
