# Introducción a Data Science: Programación Estadística con R
## Autor: Jonathan Martín Valera
### Repositorio creado para *R para principiantes*
#### Fecha de realización: Febrero 2019

---

# Introducción

R es un sistema para ánalisis estadı́sticos y gráficos creado por Ross Ihaka y Robert Gentle-
man. R se distribuye gratuitamente bajo los términos de la GNU General Public Licence.

Una de las caracterı́sticas más sobresalientes de R es su enorme **flexibilidad**.
Mientras que programas más clásicos muestran directamente los resultados de un análisis,
R guarda estos resultados como un “objeto”, de tal manera que se puede hacer un análisis sin necesidad de mostrar su resultado inmediatamente.

R es un lenguaje **Orientado a Objetos**: bajo este complejo término se esconde la simplicidad
y flexibilidad de R.

*Orientado a Objetos* significa que las variables, datos, funciones, resultados, etc., se guardan
en la memoria activa del computador en forma de objetos con un nombre especı́fico. El usuario
puede modificar o manipular estos objetos con operadores (aritméticos, lógicos, y comparativos)
y funciones (que a su vez son objetos).

R es un lenguaje **interpretado** (como Java) y no compilado (como C,
C++, Fortran, Pascal, . . . ), lo cual significa que los comandos escritos en el teclado son ejecutados
directamente sin necesidad de construir ejecutables.

---

# Conceptos antes de empezar

El **nombre de un objeto** debe comenzar con una letra (A-Z and a-z) y puede incluir letras,
dı́gitos (0-9), y puntos (.). R discrimina entre letras mayúsculas y minúsculas para el nombre de
un objeto, de tal manera que x y X se refiere a objetos diferentes.

Un **objeto puede ser creado** con el operador “asignar” el cual se denota como una flecha con
el signo menos y el sı́mbolo “>” o “<” dependiendo de la dirección en que asigna el objeto:

Para **borrar objetos en memoria**, utilizamos la función rm(): rm(x) elimina el objeto x,
rm(x,y) elimina ambos objetos x y y, y rm(list=ls()) elimina todos los objetos en me-
moria; las mismas opciones mencionadas para la función ls() se pueden usar para borrar selec-
tivamente algunos objetos: rm(list=ls(pat="ˆm")).

Para la **ayuda en línea** podemos usar `help(lm)`o
`help("lm")` tiene el mismo efecto. Esta última función se debe usar para acceder a la ayuda con caracteres no-convencionales.

Algunos ejemplos que generalmente pueden ser ejecutados sin abrir la ayuda con la función `examples()`.

Para ver la ayuda en formato html escriba el comando: `help.start()`.

La función `apropos()` encuentra todas aquellas funciones cuyo nombre contiene la palabra dada como argumento para los paquetes cargados en memoria:

![img](https://raw.githubusercontent.com/jmv74211/R/master/images/1.png)

---

# Manejando datos con R

La siguiente tabla resume los tipos de objetos y los datos que representan.

![img](https://raw.githubusercontent.com/jmv74211/R/master/images/2.png)

- Un **vector** es una variable en el significado comunmente asumido.

- Un **factor** es una variable categórica.

- Un **arreglo** es una tabla de dimensión k, y una **matriz** es un caso particular de un arreglo donde k = 2.

- Un **data.frame** (marco o base de datos) es una tabla compuesta de uno o más vectores y/o factores de la misma longitud pero que pueden ser de diferentes tipos.

- Un **ts** es una serie temporal y como tal contiene atributos adicionales tales como frecuencia y fechas.

- Una **lista** puede contener cualquier tipo de objeto incluyendo otras listas!

# Leyendo datos desde un archivo

R utiliza el directorio de trabajo para leer y escribir archivos. Para saber cual es este directorio puede utilizar el comando `getwd()`.

Para cambiar el directorio de trabajo, se utiliza la fucnión `setwd()`; por ejemplo, setwd(“C:/data”) o setwd(“/home/paradis/R”).

R puede leer datos guardados como archivos de texto (ASCII) con las siguientes funciones: `read.table` (con sus variantes, ver abajo), scan y read.fwf.

La función `read.table` crea un marco de datos (’data frame’) y constituye la manera más usual de leer datos en forma tabular. Por ejemplo si tenemos un archivo de nombre data.dat, el comando:

    misdatos <- read.table("data.dat")

creará un marco de datos denominado misdatos, y cada variable recibirá por defecto el nombre V1, V2, . . . y puede ser accedida individualmente escribiendo:

- misdatos$V1, misdatos$V2,
- escribiendo misdatos["V1"], misdatos["V2"]
- misdatos[,1], misdatos[,2 ]

Existen varias opciones con valores por defecto (aquellos usados por R si son omitidos por el usuario) que se detallan en la siguiente tabla:

![img](https://raw.githubusercontent.com/jmv74211/R/master/images/3.png)

![img](https://raw.githubusercontent.com/jmv74211/R/master/images/4.png)

---

# Guardando datos

La función `write.table` guarda el contenido de un objeto en un archivo. El objeto es tı́picamente un marco de datos (’data.frame’), pero puede ser cualquier otro tipo de objeto (vector, matriz,. . . ). Los argumentos y opciones son:

    write.table(x, file = "", append = FALSE, quote = TRUE, sep = " ",
    eol = "\n", na = "NA", dec = ".", row.names = TRUE,
    col.names = TRUE, qmethod = c("escape", "double"))

![img](https://raw.githubusercontent.com/jmv74211/R/master/images/5.png)

Una manera sencilla de escribir los contenidos de un objeto en un archivo es utilizando el
comando

    write(x, file="data.txt")

donde x es el nombre del objeto (que puede ser un vector, una matrix, o un arreglo). Esta función tiene dos opciones: `nc` (o ncol) que define el
número de columnas en el archivo (por defecto nc=1 si x es de tipo caracter, nc=5 para otros tipos), y `append` (lógico) que agrega los datos al archivo sin borrar datos ya existentes (TRUE) o borra cualquier dato que existe en el archivo (FALSE, por defecto).

Para guardar un grupo de objetos de cualquier tipo se puede usar el comando

    save(x, y, z, file= "xyz.RData").

Para facilitar la transferencia de datos entre diferentes máquinas
se pueden utilizar la opción `ascii = TRUE` . Los datos (denominados ahora como un workspace o “espacio de trabajo” en terminologı́a de R) se pueden cargar en memoria más tarde con el comando:

    load("xyz.RData")

La función `save.image()` es una manera corta del comando `save(list=ls(all=TRUE), file=".RData")` (guarda todos los objetos en memoria en el archivo .RData).

---

# Generación de datos

## Secuencias regulares

Una secuencia regular de números enteros, por ejemplo de 1 hasta 30, se puede generar con:

    x <- 1:30

La función `seq()` puede generar secuencias de números reales, donde el primer número indica el principio de la secuencia, el segundo el final y el tercero el incremento que se debe usar para generar la secuencia.

    > seq(1, 5, 0.5)
    [1] 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0

Si se quiere, también es posible introducir datos directamente desde el teclado usando la función `scan()` sin opciones:

    > z <- scan()


La función `rep()` crea un vector con elementos idénticos:

    > rep(1, 30)
    [1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

La función `gl()` (generador de niveles) es muy útil porque genera series regulares de factores.
La función tiene la forma `gl(k, n)` donde k es el número de niveles (o clases), y n es el número
de réplicas en cada nivel. Se pueden usar dos opciones: `length` para especificar el número de
datos producidos, y `labels` para especificar los nombres de los factores. Ejemplos:

![img](https://raw.githubusercontent.com/jmv74211/R/master/images/6.png)

Finalmente, `expand.grid()` crea un marco de datos con todas las combinaciones de vectores o factores proporcionados como argumentos:

![img](https://raw.githubusercontent.com/jmv74211/R/master/images/7.png)

## Secuencias aleatorias

La posibilidad de generar datos aleatorios es bastante útil en estadśtica y R tiene la capacidad de hacer esto para un gran número de funciones y distribuciones.

R tiene la capacidad de hacer esto para un gran número de funciones y distribuciones. Estas funciones son de la forma `rfunc(n, p1, p2, ...)`, donde `func` indica la disribución, `n` es el número de datos generado, y `p1, p2`, . . . son valores que toman los parámetros de la distribución.

![img](https://raw.githubusercontent.com/jmv74211/R/master/images/8.png)

---

# Manipulación de objetos

## Creación de objetos

**Vector:** La función `vector()`, que tiene dos argumentos `mode` y `length`, crea un vector cuyos elementos pueden ser de tipo numérico, lógico o caracter dependiendo del argumento especificado en mode (0, FALSE o “ ” respectivamente). Las siguientes funciones tienen exactamente el mismo efecto y tienen un solo argumento (la longitud del vector): `numeric()`,
`logical()`, y `character()`.

**Factor:** Un factor incluye no solo los valores correspondientes a una variable categórica, sino que también incluye los diferentes niveles posibles de esta variable (inclusive si están presentes en los
datos). La función `factor()` crea un factor con las siguientes opciones:

    factor(x, levels = sort(unique(x), na.last = TRUE),
           labels = levels, exclude = NA, ordered = is.ordered(x))


`levels` especifica los posibles niveles del factor (por defecto los valores únicos de x), `labels` define los nombres de los niveles, `exclude` especifica los valores x que se deben excluir de los niveles, y `ordered` es un argumento lógico que especifica si los niveles del factor están ordenados. Recuerde que x es de tipo numérico o caracter. Ejemplos:

![img](https://raw.githubusercontent.com/jmv74211/R/master/images/9.png)

La función `levels()` extrae los niveles posibles de un factor:

![img](https://raw.githubusercontent.com/jmv74211/R/master/images/10.png)


**Marco de datos (dataframe)**:
 Una matriz es realmente un vector con un atributo adicional (dim) el cual a su vez es un vector numérico de longitud 2, que define el número de filas y columnas de la matriz. Una matriz se puede crear con la función `matrix()`:

    matrix(data = NA, nrow = 1, ncol = 1, byrow = FALSE, dimnames = NULL)

La opción `byrow` indica si los valores en data deben llenar las columnas sucesivamente (por defecto) o las filas (if TRUE). La opción `dimnames` permite asignar nombres a las filas y columnas.

![img](https://raw.githubusercontent.com/jmv74211/R/master/images/11.png)

Otra manera de crear una matriz es dando los valores apropiados al atributo dim (que inicialmente tiene valor NULL):

![img](https://raw.githubusercontent.com/jmv74211/R/master/images/12.png)

**Marco de datos (dataframe)**: Hemos visto que un marco de datos (’data.frame’) se crea de manera implı́cita con la función `read.table()`; también es posible hacerlo con la función `data.frame()`.

*Los vectores incluidos como argumentos deben ser de la misma longitud, o si uno de ellos es más corto que los otros, es “reciclado” un cierto número de veces:*

---

# Glosario de Funciones

- `ls()`: Lista todos los objetos en memoria. Si se
quiere listar solo aquellos objetos que contengan un caracter en particular, se puede usar la opción
`pattern` (que se puede abreviar como pat):

      ls(pat = "m")

- `mode(object)`: Muestra el tipo del objeto.

- `length(object)`: Muestra la longitud del objeto.
