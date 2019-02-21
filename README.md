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

# Glosario de Funciones

- `ls()`: Lista todos los objetos en memoria. Si se
quiere listar solo aquellos objetos que contengan un caracter en particular, se puede usar la opción
`pattern` (que se puede abreviar como pat):

      ls(pat = "m")
