#include "eigen_mvn.h"

//PARAMETERS
void Mvn::setMean(const Eigen::VectorXf& mu)
{
	mean  = mu;
}
void Mvn::setCov(const Eigen::MatrixXf& s)
{
	sigma = s;

	// Decompose sigma into Eigen Vectors and Values
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(sigma);
	if (eigensolver.info() != Eigen::Success) std::abort();
	eigVals = eigensolver.eigenvalues();
	eigVecs = eigensolver.eigenvectors();

	//Take the abs of the eigeVals, just in case our covariance matrix is not positive, semi-definite
	eigVals = eigVals.cwiseAbs();

	// create normal distribution number generators from eigVal
	randN.clear();

	for (size_t ii = 0; ii < n; ii++)
	{
		randN.push_back(std::normal_distribution<float>(0.0f, sqrt(eigVals(ii))));
	}
}
void Mvn::setParameters(const Eigen::VectorXf& mu, const Eigen::MatrixXf& s)
{
	//Set Internal variables
	setMean(mu);
	setCov(s);
}

//PDF
float Mvn::pdf(const Eigen::VectorXf& x)
{
	float n = x.rows();
	float sqrt2pi = std::sqrt(2.0f * M_PI);
	float quadform  = (x - mean).transpose() * sigma.inverse() * (x - mean);
	float norm = std::pow(sqrt2pi, - n) *
				  std::pow(sigma.determinant(), - 0.5f);

  return norm * exp(-0.5f * quadform);
}

//Samplers
void Mvn::sample(const Eigen::VectorXf& mu, const Eigen::MatrixXf& s, unsigned int N)
{
	//set parameters
	setParameters(mu, s);

	//Sample values
	sample(N);
}

void Mvn::sample(const Eigen::VectorXf& mu, const Eigen::MatrixXf& s)
{
	//set parameters
	setParameters(mu, s);

	//Sample values
	sample();
}

void Mvn::sample(unsigned int N)
{
	if (N!=m)
	{
		//Update Requested Sample Size
		m = N;

		//Deallocate Y and Z
		delete [] Y;
		delete [] Z;

		//Reallocate Y and Z
		Y = new float[m*n];
		Z = new float[m*n];
	}
	//Sample values
	sample();
}

void Mvn::sample()
{
	// Initialize intermediate, uncorrelated values: Y
	for (size_t ii = 0; ii < m; ii++)
	{
		for (size_t jj = 0; jj < n; jj++)
		{
			Y[n*ii + jj] = randN[jj](gen);
		}
	}

	//Generate correlated values: Z
	float sum;
	for (size_t ii = 0; ii < m; ii++)
	{
		for (size_t jj = 0; jj < n; jj++)
		{
			sum = 0.0f;
			for (size_t kk = 0; kk < n; kk++)
			{	
				//NOTE: Eigen is column-major
				sum += Y[n*ii + kk]*eigVecs(jj, kk);
			}
			//Add in mean
			sum         += mean(jj);
			Z[n*ii + jj] = sum;
		}
	}
}
