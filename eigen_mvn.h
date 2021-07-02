#ifndef EIGEN_MVN_H_
#define EIGEN_MVN_H_

#include <Eigen/Dense>
#include <vector>
#include <random>

/*
This class generates multivariate gaussian samples of arbitrary size and covariance.

For efficiency, the constructor sets the initial dimensions and requested number of samples.

For convenience, samples are stored in a mxn c++ array

EVerything is computed at 32-bit float precision, for compatibility with GPU and embedded hardware
*/

class Mvn
{
public:
	unsigned int n;	//Dimensions
	unsigned int m;	//Requested number of samples
	float* Z;

	//Default Constructor
	Mvn(){};
	Mvn(const unsigned int dims, const unsigned int samples):
		Z(NULL),
		Y(NULL)
	{
		configure(dims, samples);
	};

	~Mvn()
	{
		if (Y!=NULL)
		{
			delete [] Y;
			Y=NULL;
		}
		if (Z!=NULL)
		{
			delete [] Z;
			Z=NULL;
		}
	};

	//configure mvn
	void configure(const unsigned int dims, const unsigned int samples)
	{
		n = dims;
		m = samples;

		//Resize and Set Defaults
		mean.resize(n);
		sigma.resize(n,n);
		eigVecs.resize(n,n);
		eigVals.resize(n);

		mean.setZero(n);
		sigma   = Eigen::MatrixXf::Identity(n,n);
		eigVecs = Eigen::MatrixXf::Identity(n,n);
		eigVals.setOnes(n);

		randN.reserve(n);

		//Allocate for intermediate sampling variables
		if (Y==NULL)
			Y = new float[m*n];

		//Allocate output matrix for initial requested number of samples
		if (Z==NULL)
			Z = new float[m*n];

		//Set seed
		gen.seed(rd());
		//Set Initial parameters
		setParameters(mean, sigma);
	}
	//Parameters
	void setMean(const Eigen::VectorXf& mu);
	void setCov(const Eigen::MatrixXf& s);
	void setParameters(const Eigen::VectorXf& mu, const Eigen::MatrixXf& s);

	//PDF
	float pdf(const Eigen::VectorXf& x);

	//Samplers
	void sample(const Eigen::VectorXf& mu, const Eigen::MatrixXf& s, unsigned int N);
	void sample(const Eigen::VectorXf& mu, const Eigen::MatrixXf& s);
	void sample(unsigned int N);
	void sample();

private:
	Eigen::VectorXf mean;
	Eigen::MatrixXf sigma;

	Eigen::MatrixXf eigVecs;
    Eigen::VectorXf eigVals;

    //Vector of random number generators
    std::random_device rd;
    std::default_random_engine gen;
    std::vector<std::normal_distribution<float>> randN;
    float* Y;	//Temp variable in sampling
};

#endif //EIGEN_MVN_H_