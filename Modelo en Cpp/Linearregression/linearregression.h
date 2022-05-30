#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <boost/algorithm/string.hpp>

class LinearRegression
{
public:
    LinearRegression(){}


    std::tuple<Eigen::VectorXd, std::vector<float>> GradienteD(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha, int iteration);
    float   fCostoOLS(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta);
};

#endif // LINEARREGRESSION_H
