# Introducción a Data Science: Programación Estadística con R
## Autor: Jonathan Martín Valera
### Repositorio creado para el curso de  *Introducción a Data Science: Programación Estadística con R* de coursera
#### Fecha de realización: Febrero 2019

---

# Introducción

## Ventajas
- Gratis
- Software libre
- Multiplataforma

## Desventajas
- Sin sistema integrado de gráficos 3D
- Funcionalidad basada en usuarios.
- Los objetos deben de estar cargados en memoria RAM.

# Encontrar ayuda

R viene con un sistema de ayuda integrado. Hay 5 funciones de utilidad:

- help() - ?
- example()
- help.search() - ??
- library(help="")
- vignette("")

Ejemplos:

- help("read.table"): Despliegue de la ventana de ayuda con el manual e información requerida.
- example("read.table"): Muestra posibles ejemplos para esa opción
- library(help="grDevices"): Buscar ayuda desde la biblioteca o paquete
- vignette("Intro2Matrix"): Genera un pdf con la información del paquete y ejemplos

# Funciones

- `ls()`: Lista el contenido de un objeto.
- `attach(object)`: Cargar objeto en memoria.
- `attributes`: Listar parámetros de una función
- `class(x)`: Lista la clase a la que pertenece x.
- `dim(matrix)`: Indica la dimensión de la matriz
- `sys.time()`: Nos devuelve un objeto con la fecha actual
- `strftime(objectTime, stringFormat)`: Nos devuelve una cadena con la fecha formateada.

# Tipos

  - character
  - numeric
    - integer
    - complex
  - logical

Un **vector** sólo puede contener objetos de la misma clase. Se pueden crear objetos de tipo vector con la función `vector()`.

`c()` se utiliza para crear vectores de objetos. Pe C(0.3,0.9...)

La **lista** se puede ver como un vector de objetos de diferentes clases.

- NaN: not a number
- Inf: infinito


**Coercción:** Genera que todos los objetos de un vector sean de la misma clase.
- **coercción explícita:** Funciones *as.\**
  - as.numeric()
  - as.logical()
  - as.character()
  - as.integer()

# Matrices

    matrix(1:6,nrow=2,ncol=3)

**Los elementos de una matriz se van rellenando por columnas (no por filas como se hace comúnmente)**

Para convertir un vector a matriz utilizando la función `dim()` se puede realizar lo siguiente:

    > m <- 1:10
    output: 1 2 3 4 5 6 7 8 9 10

    > dim(m) <- c(2,5)
    output:
        [1,] [2,] [3,] [4,] [5,]
    [1,] 1    3    5    7    9
    [2,] 2    4    6    8    10


Funciones para generar matrices

`cbind():` Une vectores en forma de columnas para generar la matriz

`rbind():` Une vectores en forma de filas para generar la matriz

**Para poder unir los vectores y formar la matriz, ambos vectores deben de tener el mismo número de elementos**

ejemplo:

      > x <- 5:8
      > y <- 12:15
      > cbind(x,y)

           x y
      [1,] 5 12
      [2,] 6 13
      [3,] 7 14
      [4,] 8 15

      > rbind(x,y)

      [1,] [2,] [3,] [4,]
    x  5    6    7    8
    y  12   13   14   15

- **Multiplicación matricial:** x%\*%y

# Fechas y tiempos

**POSIXct** nos devuelve el número de segundos transcurridos desde el 1 de enero de 1970.

`sys.time()`: Nos devuelve un objeto con la fecha actual.

Para convertir el valor de tiempo en los diferentes formatos, podemos utilizar `as.POSIXct()` y `as.POSIXlt()`. (POSIXct es de tipo double y POSIXlt es de tipo lista).

`strftime(objectTime, stringFormat)` nos devuelve una cadena con la fecha formateada. Ejemplo:

  
