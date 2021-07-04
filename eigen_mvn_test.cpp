// eigen_mvn_test.cpp
#include <iostream>
#include "eigen_mvn.h"

void TestPDF()
{
	//Define dimension of problem and requested sample size
	size_t D = 2;
	size_t N = 100;

	// Define the covariance matrix and the mean
	Eigen::VectorXf mu(D);
	Eigen::MatrixXf sig(D, D);

	mu << 0.0f, 0.0f;
	sig << 1.0f, 0.1f,
		   0.1f, 1.0f;

	Mvn mvn(D, N);

	mvn.setParameters(mu, sig);

	Eigen::VectorXf test(2);
	test << 0.0f, 0.0f;
	std:: cout << "Testing PDF0: " << (abs(mvn.pdf(test) - 0.16f) < 1e-4f ? "Passed" : "Failed") << std::endl;

	test << -0.6f, -0.6f;
	std:: cout << "Testing PDF1: " << (abs(mvn.pdf(test) - 0.1153f) <  1e-4f ? "Passed" : "Failed") << std::endl;
}

void TestSampling()
{
	//Define dimension of problem and requested sample size
	size_t D = 6;
	size_t N = 100;

	std::cout << "\nTestSampling: " << D << " dims and " << N << " samples" << std::endl;

	// Define the mean and covariance matrix
	Eigen::VectorXf mu(D);
	Eigen::MatrixXf sig(D,D);

	mu << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f;
	// sig << 1.0,  0.2, -0.4,
	// 	   0.2,  1.0,  0.7,
	// 	  -0.4,  0.7,  1.0;

	sig <<   9.74137e-08f,  9.25364e-09f,  1.07041e-09f,  9.61577e-09f, -9.14284e-08f,  3.35628e-09f,
             9.25374e-09f,  2.13549e-07f,  2.24429e-08f,  2.10761e-07f, -7.76453e-09f, -2.42771e-09f,
             1.07042e-09f,  2.24429e-08f,  7.02041e-09f,  2.21773e-08f, -6.28786e-10f, -7.27438e-10f,
             9.61587e-09f,  2.10761e-07f,  2.21774e-08f,  2.08341e-07f, -8.11999e-09f, -2.30132e-09f,
            -9.14284e-08f, -7.76444e-09f, -6.28776e-10f, -8.11991e-09f,   8.6195e-08f, -3.69762e-09f,
             3.35628e-09f,  -2.4277e-09f, -7.27438e-10f, -2.30131e-09f, -3.69762e-09f,  5.14241e-09f;

	Mvn mvn(D, N);
	mvn.sample(mu, sig);

	//Estimate covariance matrix from samples
	//0) initialize
	Eigen::VectorXf approx_mean(D);
	Eigen::MatrixXf approx_sigma(D, D);
	approx_mean.setZero();
	approx_sigma.setZero();

	// 1) get the means
	for (size_t ii = 0; ii < D; ii++)
	{
		for (size_t jj = 0; jj < N; jj++)
		{
			approx_mean(ii) += mvn.Z[D*jj + ii];
		}
		approx_mean(ii) /= N;
	}
	// 2) compute covariance
	for (size_t ii = 0; ii < D; ii++)
	{
		for (size_t jj = 0; jj < D; jj++)
		{
			for (size_t kk = 0; kk < N; kk++)
			{
				approx_sigma(ii, jj) += (mvn.Z[D*kk + ii] - approx_mean(ii))*(mvn.Z[D*kk + jj] - approx_mean(jj));
			}
			approx_sigma(ii, jj) /= (N-1);
		}
	}

	std::cout << "\nMean: " << mu.transpose() << std::endl;
	std::cout << "Mean_hat: " << approx_mean.transpose() << std::endl << std::endl;

	std::cout << "Sigma:\n" << sig << std::endl << std::endl;
	std::cout << "Sigma_hat:\n" << approx_sigma << std::endl << std::endl;

	// The parameters should convergence as 1/sqrt(N), but nothing is perfect
	float convergChk0 = (5.0f*sqrt(sig.maxCoeff()))/sqrt(N);
	float convergChk1 = (5.0f*sqrt(sig.maxCoeff()))/sqrt(N);
	std::cout << "Mean Estimator Check: " << (approx_mean-mu).cwiseAbs().maxCoeff() << " < " << convergChk0 << std::endl;
	std::cout << "Sigma Estimator Check: " << (approx_sigma-sig).cwiseAbs().maxCoeff() << " < " << convergChk1 << std::endl << std::endl;
}

// TEST(TestMvn, TestResizeSampling)
// {

// }

int main()
{
	TestPDF();

	TestSampling();

	return 0;
}
