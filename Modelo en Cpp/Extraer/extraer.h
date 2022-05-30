#ifndef EXTRAER_H
#define EXTRAER_H
/*
* Fecha: 28/05/2022
* Autor: Diego Alejandro Bermudez Gonzalez
* Materia: HPC- 1
* Tema: Parcial Final HPC
*
*/
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <boost/algorithm/string.hpp>

class extraer
{

   /*Se presenta el constructor de los argumentos de entrada a la clase extraer*/
   /*Nombre del dataset*/
   std::string setDatos;
   //Separador de columnas
   std::string delimitador;
   //Si tiene cabezera o no el dataset
   bool header;

public:
   extraer(std::string datos, std::string separador, bool head):
       setDatos(datos),
       delimitador(separador),
       header(head){}

    std::vector<std::vector<std::string>> readCSV();
    Eigen::MatrixXd CSVtoEigen(std::vector<std::vector<std::string>> SETdatos,
                                        int filas, int columnas);

    auto promedio(Eigen::MatrixXd datos) ->
    decltype(datos.colwise().mean());
    auto desvEstandar(Eigen::MatrixXd datos) ->
    decltype(((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt());
    Eigen::MatrixXd Normalizador(Eigen::MatrixXd datos);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> trainTestSplit(Eigen::MatrixXd DatosNorm, float sizeTrain);
    void vectorToFile(std::vector<float> dataVector, std::string fileName);
    void EigenToFile(Eigen::MatrixXd dataMatrix, std::string name);
    float R2_score(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);
};

#endif // EXTRAER_H
