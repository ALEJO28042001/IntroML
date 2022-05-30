 /*
* Fecha: 28/05/2022
* Autor: Diego Alejandro Bermudez Gonzalez
* Parcial Final HPC
*/

#include "extraer.h"
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <boost/algorithm/string.hpp>

using namespace std;
/* Primera funciòn miembro: lectura de fichero csv. Se presenta como un vector de vectores del tipo string.
La idea es leer linea por linea y almacenar cada una en un vector de vectores de tipo string*/
std::vector<std::vector<std::string>> extraer::readCSV(){
    //Abrir el fichero para lectura solamente
    std::fstream Fichero(setDatos);
    //Vector de vectores tipo string a entregar por parte de la funciòn.
    std::vector<std::vector<std::string>> datosString;
    //Se itera a travès de cada linea y se divide el contenido dado por el separador provisto por el constructor.
    //Almacenar cada lìnea
    std::string linea = "";
    while(std::getline(Fichero, linea)){
        //Se crea un vector para almacenar la fila
        std::vector<std::string> vectorFila;
        //Se separa segùn el delimitador
        boost::algorithm::split(vectorFila,
                               linea,
                               boost::is_any_of(delimitador));
        datosString.push_back(vectorFila);

    }
    //Se cierra el fichero .csv
    Fichero.close();
    //Se retorna el vector de vectores de tipo String
    return datosString;

}

/* Se implementa la segunda funciòn miembro, la cual tiene como misiòn, transformar el vector de vectores del tipo String
 * en una matriz Eigen. La idea es simular un objeto DataFrame de pandas, para poder manipular los datos. */

Eigen::MatrixXd extraer::CSVtoEigen(std::vector<std::vector<std::string>> SETdatos,
                                    int filas, int columnas){
    Eigen::MatrixXd MatrizDF(columnas, filas);
    //Se hace la pregunta si tiene cabecera o no el vector de vectores.
    //Si tiene cabecera, se debe eliminar.
    if (header != false){
        filas = filas-1;
    }
    /* Se itera sobre cada registro del fichero, a la vez que se almacena en una matrixXd de dimensiòn filas por columnas. Principalmente almacenarà
    * strings (por que llega un vector de vectores del tipo string. La idea es segmentar en funciòn del delimitador*/




    for (int i=0; i<filas; i++){
        for (int j=0; j<columnas; j++){
            MatrizDF(j,i) = atof(SETdatos[i][j].c_str());
        }
    }



    //Se transpone la matriz, puesto que viene por columnas por filas para retornar
    return MatrizDF.transpose();
}

//Función para calcular el promedio
//En C++, la herencia del tipo de dato no es directa (Sobre todo si es a partir de funciones dadas por otras interfaces/clases/bibliotecas). Entonces se declara el tipo
//en una expresión "decltype". Con el fin de tener seguridad de que tipo de dato retornará la función.
auto extraer::promedio(Eigen::MatrixXd datos) ->
decltype(datos.colwise().mean()){
    return datos.colwise().mean();
}

//Función para calcular la desviación estándar.
//Para implementar la desviación estandar:

auto extraer::desvEstandar(Eigen::MatrixXd datos) ->
decltype(((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt()){
    return ((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt();
}

/*A continuación se procede a implementar la función de normalización. La idea fundamental es que los datos,
 * presenten una cercana aproximación al promedio, evitando los valores cuyas magnitudes son muy altas o muy bajas.
 * Por ejemplo los outlayers (valores atípicos) */

Eigen::MatrixXd extraer::Normalizador(Eigen::MatrixXd datos){
    Eigen::MatrixXd datosEscalados = datos.rowwise()  - promedio(datos);


    Eigen::MatrixXd MatrixNormal = datosEscalados.array().rowwise()/desvEstandar(datosEscalados);

    //Se retorna la matrix normalizada
    return MatrixNormal;
}

// Para los algoritmos y o modelos de Machine Learning, se necesita dividir los datos en dos grupos. EL primer grupo, es de
// entrenamiento. Se recomienda que sea aproximadamente el 80% de los datos. El segundo grupo, es para pruebas. Será el resto. El 20%.
// La idea es crear una función que permita deividir los datos en los grupos de entrenamiento y prueba de forma automática. Se requiere
// que la selección de los registros para cada grupo sea aleatoria. Esto garantiza que el resultado del modelo presente una aceptable precisión.


//La función de división a continuación, tomará el porcentaje superior de la matriz dada, para entrenamiento. La parte restante, de la matriz dada,
//para prueba. La funciòn devolverá una tupla de 4 matrices dinámicas, variables independientes: entrenamiento y prueba, variables independientes y dependientes.
//Al utilizar la funciòn en el principal se debe desempaquetar la tupla, para obtener los 4 conjuntos de datos.

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> extraer::trainTestSplit(Eigen::MatrixXd DatosNorm, float sizeTrain){
    //Se crea una variable para obtener el número de filas totales.
    int filasTotales = DatosNorm.rows();
    //Variable para obtener el número de filas de entrenamiento.
    int filasTrain = round(filasTotales * sizeTrain);
    //Variable para obtener el número de filas de prueba.
    int filasTest = filasTotales - filasTrain;

    //Se crea la matriz de entrenamiento: parte superior de la matriz de entrada.
    Eigen::MatrixXd train = DatosNorm.topRows(filasTrain);

    //Del conjunto de entrenamiento y para este caso en especial (Dataset, Winedata), todas las columnas de la izquierda son las variables
    //independientes (Features), y la última columna de la derecha representa la variable dependiente.
    //A continuación, se declara el conjunto de entrenamiento de las variables independientes X.
    Eigen::MatrixXd X_train = train.leftCols(DatosNorm.cols()-1);

    //A continuación se declara el conjunto de entrenamiento de las variables dependientes Y.
    Eigen::MatrixXd Y_train = train.rightCols(1);

    //A continuación se declara el conjunto de datos para prueba.
    Eigen::MatrixXd test = DatosNorm.bottomRows(filasTest);

    //A continuación se declara el conjunto de prueba de las variables independientes X.

    Eigen::MatrixXd X_test = test.leftCols(DatosNorm.cols()-1);

    //A continuación se declara el conjunto de prueba de las variables dependientes Y.

    Eigen::MatrixXd Y_test = test.rightCols(1);

    //Finalmente se devuelve la tupla empaquetada.

    return std::make_tuple(X_train, Y_train, X_test, Y_test);
}

/* A continuación se desarrollan 2 nuevas funciones para convertir de fichero a vector, y pasar de una matriz a fichero. La idea
* principal es almacenar los valores parciales en ficheros por motivos dde seguridad, control y seguimiento de la ejecución
* del algoritmo de la regresión final. */

/*Función para exportar valores de un fichero a un vector. La función del tipo vacío, recibe un vector que contendrá los valores del archivo dado. */

void extraer::vectorToFile(std::vector<float> dataVector, std::string fileName){
    //Se crea un buffer (bus de memoria temporal) como objeto que contiene la data de un fichero.
    std::ofstream BufferFichero(fileName);

    //A continuación se itera sobre el buffer, almacenando cada objeto encontrado, representado por un salto de linea("\n")
    std::ostream_iterator<float> BufferIterator(BufferFichero, "\n" );

    //Se copia la data del iterador(BufferIterator) en el vector
    std::copy(dataVector.begin(), dataVector.end(), BufferIterator);


}


//La siguiente función representa la conversión de una matriz Eigen a fichero.
void extraer::EigenToFile(Eigen::MatrixXd dataMatrix, std::string name){
    //Se crea un buffer (bus de memoria temporal) como objeto que contiene la data de un fichero.
    std::ofstream BufferFichero(name);

    //Se condiciona mientras el fichero este abierto, almacenar los datos, separados por un salto de linea("\n")
    if (BufferFichero.is_open()){
        BufferFichero << dataMatrix << "\n";
    }
}

/* Para determinar que tan bueno es nuestro modelo vamos a crear una función como
 *métrica de rendimiento. La métrica seleccionada es R2_score */

float extraer::R2_score(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){

    auto numerador = pow((y-y_hat).array(), 2).sum();
    auto denominador = pow(y.array()-y.mean(), 2).sum();
    return (1- (numerador/denominador));
}




















