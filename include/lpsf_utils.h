#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>


namespace LPSFUTIL
{

template<typename integer_type>
std::vector<integer_type> arange(integer_type start, integer_type stop)
{
    std::vector<integer_type> v;
    for ( integer_type ii=start; ii<stop; ++ii )
    {
        v.push_back(ii);
    }
    return v;
}

std::vector<Eigen::VectorXd> unpack_MatrixXd_columns( const Eigen::MatrixXd & V )
{
    std::vector<Eigen::VectorXd> vv(V.cols());
    for ( unsigned long int ii=0; ii<V.cols(); ++ii )
    {
        vv[ii] = V.col(ii);
    }
    return vv;
}

Eigen::MatrixXd readMatrix(const char *filename)
//based on https://stackoverflow.com/a/22988866/484944
{
    std::vector<std::vector<double>> buff;

	// Read numbers from file into buffer.
	std::ifstream infile;
	infile.open(filename);
	while (! infile.eof())
	{
		std::string line;
		getline(infile, line);

        std::vector<double> current_row_vector;
        double current_entry[1];
        int cc = 0;
		std::stringstream stream(line);
		while(! stream.eof())
        {
            stream >> current_entry[0];
            current_row_vector.push_back(current_entry[0]);
            cc += 1;
        }

        buff.push_back(current_row_vector);
	}

	infile.close();

    unsigned long int nrow = buff.size() - 1;
    unsigned long int ncol = buff[0].size();

	// Populate matrix with numbers.
	Eigen::MatrixXd result(nrow, ncol);
	for (unsigned long int ii = 0; ii < nrow; ++ii)
		for (unsigned long int jj = 0; jj < ncol; ++jj)
            result(ii,jj) = buff[ii][jj];

	return result;
}

Eigen::MatrixXi matrix_double_to_int(Eigen::MatrixXd M_double)
{
    Eigen::MatrixXi M_int(M_double.rows(), M_double.cols());
    for ( int rr=0; rr<M_double.rows(); ++rr )
    {
    	for ( int cc=0; cc<M_double.cols(); ++cc )
    	{
            M_int(rr,cc) = std::lround(M_double(rr,cc));
    	}
    }
    return M_int;
}

}