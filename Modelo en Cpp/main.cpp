/*
* Fecha: 28/05/2022
* Autor: Diego Alejandro Bermudez Gonzalez
* Materia: HPC- 1
* Parcial final HPC
*/

#include "Extraccion/extraer.h"
#include "linearregression.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <string.h>

using namespace std;


int main(int argc, char *argv[]){

    cout<<"Examen final HPC"<<endl<<"Presentado por: Diego Bermúdez"<<endl;
    cout<<"Modelo RL"<<endl<<"Universidad Sergio Arboleda"<<endl<<endl;
    /*Para la extraccion de la informacion se elimina la cabecera del dataSet
     *por el tipo de dato de la misma, los datos que se manejaran en el dataSet serán de tipo float
     *el nombre del dataSet será poverty2.csv*/

    /* Se crea un objeto del tipo extraer, para incluir los 3 argumentos que necesita el objeto. */
    extraer extraerData(argv[1], argv[2], argv[3]);

    //Se crea un objeto del tipo LinearRegresion, sin ningún argumento de entrada
    LinearRegression LR;

    // Se requiere probar la lectura del fichero y luego se requiere observar el dataset como un objeto de matriz tipo dataFrame
    std::vector<std::vector<std::string>> dataSET = extraerData.readCSV();

    int filas = dataSET.size()+1;
    int columnas = dataSET[0].size();
    Eigen::MatrixXd MatrizDATAF = extraerData.CSVtoEigen(dataSET, filas, columnas);
    std::cout<<"Se imprime la cantidad de filas y columnas del dataSet"<<std::endl;
    std::cout<<"filas: "<<filas<<std::endl;
    std::cout<<"columnas: "<<columnas<<std::endl;

    /* Se imprime la matriz que contiene los datos del dataset */
    std::cout<<"Brth15to17	Brth18to19	ViolCrime	PovPct"<<filas;
    std::cout<<" Se imprime el dataSet "<<std::endl<<MatrizDATAF<<std::endl;



    std::cout<<"El promedio por columna es: "<<std::endl<<extraerData.promedio(MatrizDATAF)<<std::endl;
    std::cout<<"La desviación estándar por columna es: "<<std::endl<<extraerData.desvEstandar(MatrizDATAF)<<std::endl;

    //Se crea la matriz para almacenar la normalización
    Eigen::MatrixXd matNormal = extraerData.Normalizador(MatrizDATAF);

    //se imprime el dataSet con datos normalizados
    std::cout<<"Los datos normalizados son: "<<std::endl;
    std::cout<<matNormal<<endl;

    //A continuación se dividen entrenamiento y prueba en conjuntos de datos de entrada (matNormal).

    Eigen::MatrixXd X_test, Y_test, X_train, Y_train;

    //Se dividen los datos y el 80% es para entrenamiento.
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> MatrixDividida = extraerData.trainTestSplit(matNormal, 0.8);
    //Se desempaqueta la tupla.
    std::tie(X_train, Y_train, X_test, Y_test) = MatrixDividida;
    //std::cout<< matNormal.rows()<<std::endl;
    //std::cout<<X_test.rows()<<std::endl;
    //std::cout<<X_train.rows()<<std::endl;

    //A continuacioń se hará el primer módulo de Machine Learning. Se hará una clase "RegresiónLineal". Con su correspondiente constructor de argumentos
    //de entrada y métodos para el cálculo del modelo RL. Se tiene en cuenta que el RL es un método estadístico, que define la relación entre las variables
    //independientes y la variable dependiente.
    //La idea principal, es definir una línea recta (Híper plano) con sus coeficientes (pendientes) y punto de corte.
    //Se tienen diferentes métodos para resolver RL. Para este caso se usará el método de los Mínimos Cuadrados Ordinarios. (OLS), por ser un
    //método sencillo y computacionalmente económico. Representa una solución óptima para conjunto de datos no complejos. El dataset a utilizar
    //es el de vinoRojo, el cuál tiene 3 variables (multivariable) independientes. Para ello hemos de implementar el algoritmo del gradiente descendiente,
    //cuyo objetivo principal es minimizar la función de costo.

    //Se define un vector para entrenamiento y para prueba inicializados en unos
    Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(X_test.rows());


    //Se redimensionan las matrices para ser ubicadas en el vector ed Unos: Similar al reshape() de numpy.
    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    X_train.col(X_train.cols()-1) = vectorTrain;

    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    X_test.col(X_test.cols()-1) = vectorTest;

    /* Se define el vector theta que se pasara al algoritmo del gradiente descendiente.
     *Básicamente es un vector de ceros del mismo tamaño del entrenamiento, adicionalmente
     * se pasará alpha y el número de iteraciones
     * */
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());
    float alpha = 0.01;
    int iteraciones = 1000;
    /* A continuación se definen las variables de salida, que representan los coeficientes y el vector
     *de costo */
    Eigen::VectorXd thetaSalida;
    std::vector<float> costo;

    /*Se desempaqueta la tupla como objeto instanciando del gradiente descendiente.
     * */
    std::tuple<Eigen::VectorXd, std::vector<float>> objetoGradiente = LR.GradienteD(X_train,
                                                          Y_train, theta, alpha, iteraciones);
    std::tie(thetaSalida, costo) = objetoGradiente;

    //Se imprime los coeficientes para cada variable.
    //std::cout<<thetaSalida<<std::endl;

    //Se mprime para inspección ocular la función de costo
    /*for (auto var: costo) {
        std::cout<<var<<std::endl;
    }*/


    /* Se almacena la función de costo y las variables Theta a ficheros */

    extraerData.vectorToFile(costo, "costo.txt");
    extraerData.EigenToFile(thetaSalida, "theta.txt");


    /* Se calcula el promedio y la desviación estandar, para calcular las predicciones.
    Es decir, se debe de normalizar para calcular la métrica. */
    auto muData = extraerData.promedio(MatrizDATAF);
    auto muFeatures = muData(0, 3);
    auto escalado = MatrizDATAF.rowwise() - MatrizDATAF.colwise().mean();
    auto sigmaData = extraerData.desvEstandar(escalado);
    auto sigmaFeatures = sigmaData(0, 3);

    Eigen::MatrixXd y_train_hat = (X_train*thetaSalida*sigmaFeatures).array() + muFeatures;
    Eigen::MatrixXd y = MatrizDATAF.col(3).topRows(42);// Datos reales tomados del dataSet (entrenamiento)

    float R2_score = extraerData.R2_score(y, y_train_hat);
    std::cout<<"El valor obtenido por R2_score para los datos de entrenamiento es:"<<R2_score<<std::endl;
    cout<<"Lo que nos indica que el modelo y mas especificamente respecto a los datos de entrenamiento tiene"
          "una fiabilidad media pues hablando de un valor poco mayor al 0.5 "
          "nos indica que sus predicciones tendra correlación con los datos reales"
          "pero sin llegar a un nivel muy cercano en su precisión"<<endl;
    extraerData.EigenToFile(y_train_hat, "y_train_hatCPP.txt");// Se extrae la data a un archivo .txt

    /* Se reutiliza el espacio de la variable y pues solo cambian los datos tomados*/

    Eigen::MatrixXd y_test_hat = (X_test*thetaSalida*sigmaFeatures).array() + muFeatures;
    y = MatrizDATAF.col(3).bottomRows(10);// Se toman los ultimos datos correspondientes al test
    R2_score = extraerData.R2_score(y, y_test_hat);
    std::cout<<"El valor obtenido por R2_score para los datos de prueba es:"<<R2_score<<std::endl;
    cout<<"Lo que nos indica que el modelo y mas especificamente respecto a los datos de prueba tiene"
          "una fiabilidad media pues hablando de un valor poco mayor al 0.7 "
          "nos indica que sus predicciones tendra correlación con los datos reales"
          "pero sin llegar a un nivel muy cercano en su precisión el cual "
          "se encontrará unos puntos por debajo respecto a los datos de entrenamiento"<<endl;
    extraerData.EigenToFile(y_test_hat, "y_test_hatCPP.txt");// Se extrae la data a un archivo .txt


    return EXIT_SUCCESS;
}

