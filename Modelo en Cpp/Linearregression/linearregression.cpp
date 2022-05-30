#include "linearregression.h"


#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <boost/algorithm/string.hpp>

float LinearRegression::fCostoOLS(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta)
{
    /*se necesita entrenar el modelo, lo que implica minimizar alguna funciòn de costo (para este caso se ha seleccionado para función de costo
     * OLS, y de esta forma se puede medir la función de hipótesis. Una función de costo es la forma de penalizar al modelo por cometer un
     * error. Se implementa una función del tipo flotante, que toma como entradas "X" y "y" y los valores de theta inicializados (Los
     * valores de theta se fijan inicialmente en cualquier valor para que al iterar según alpha. consiga el menor valor para la función de costo.*/

    /*Se almacena la diferencia elevada al cuadrado (Función de hipótesis. Que representa el error.)*/
    Eigen::MatrixXd diferencia = pow((X * theta - y).array(), 2);
    return (diferencia.sum()/(2*X.rows()));
}

//Se necesita proveer al programa una función para dar al algoritmo un valor inicial para theta, el cual va a cambiar iterativamente hasta que converja
// al valor mínimo de la función de costo. Basciamente describe el Gradiente Descendiente: La idea es calcular el gradiente para la función
//de costo dado por la derivada parcial. La función tendrá un alpha que representa el salto del gradiente. La función tiene como entrada "X", "y",
//"theta", "alpha" y el número de iteraciones que necesita theta actualizada cada vez para que la función converja.*/

std::tuple<Eigen::VectorXd, std::vector<float>> LinearRegression::GradienteD(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha, int iteration) {
    //Se almacena temporalmente los parámetros de theta
    Eigen::MatrixXd tempTheta = theta;
    //Se extrae la cantidad de parámetros.
    int parametros = theta.rows();
    //Valores de costo inicial, se actualizará cada vez con los nuevos pesos (pendientes).
    std::vector<float> costo;
    costo.push_back(fCostoOLS(X, y, theta));
    //Para cada iteración se calcula la función de error. Se multiplica cada feature (X) que calcula el error y se almacena en una variable temporal. Luego se actualiza theta
    //y se calcula de nuevo la función de costo basada en el nuevo valor de theta.
    for (int i = 0; i < iteration; ++i) {
       Eigen::MatrixXd Error = X * theta-y;
       for (int j = 0; j< parametros; ++j) {
           Eigen::MatrixXd X_i = X.col(j);
           Eigen::MatrixXd tempError = Error.cwiseProduct(X_i);
           tempTheta(j,0) = theta(j, 0) - ((alpha/X.rows()) * tempError.sum());
       }
       theta = tempTheta;
       costo.push_back(fCostoOLS(X, y, theta));
    }
    //Se empaqueta la tupla y se retorna.
    return std::make_tuple(theta, costo);
}
