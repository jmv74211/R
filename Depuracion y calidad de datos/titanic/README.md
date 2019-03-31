Preprocesamiento de datos con el dataset
[titanic](https://www.kaggle.com/c/titanic/).

> El hundimiento del Titanic es una de las tragedias marítimas más
> conocidas de la historia. El 15 de abril de 1912, durante su viaje
> inaugural, el Titanic se hundió después de chocar contra un iceberg.
> En el accidente murieron 1502 personas de las 2224 que habían
> embarcado, inluyendo pasajeros y tripulación. Una de las razones por
> las que no se encontraron más supervivientes fue la falta de espacio
> en los barcos salvavidas. Así, aunque la suerte sin duda sonrió a los
> supervivientes, también resultaron más favorecidos algunos grupos de
> personas, como las mujeres, los niños y los pasajeros de la clase
> superior.

**En este problema analizaremos qué tipos de personas tuvieron más
probabilidades de sobrevivir. Para ello, aplicaremos técnicas de
aprendizaje automático que nos permitirán predecir qué pasajeros
sobrevivieron al hundimiento.**

En primer lugar, nos centraremos en el pre-procesamiento de los datos
utilizando [tidyverse](https://www.tidyverse.org), una colección de
paquetes de R para Ciencia de Datos. En el libro *[R for Data
Science](http://r4ds.had.co.nz)* podemos encontrar documentación
detallada sobre [tidyverse](https://www.tidyverse.org).

<br/> **Índice**

-   [Lectura de datos](#Lectura%20de%20datos)
-   [Estado del conjunto de
    datos](#Estado%20del%20conjunto%20de%20datos)
-   [Predictor básico: todos los pasajeros
    mueren](#Predictor%20básico:%20todos%20los%20pasajeros%20mueren)
-   [Predictor refinado: todos los hombres
    mueren](#Predictor%20refinado:%20todos%20los%20hombres%20mueren)
-   [Predictor refinado: todos los hombres mueren, las mujeres en 3ª
    clase que pagan &gt;= 20
    mueren](#Predictor%20refinado:%20todos%20los%20hombres%20mueren,%20las%20mujeres%20en%203ª%20clase%20que%20pagan%20%3E=%2020%20mueren)
-   [Valores perdidos](#Valores%20perdidos)
-   [Valores con ruido](#Valores%20con%20ruido)

Lectura de datos
----------------

Comenzaremos utilizando el fichero
[*train.csv*](https://www.kaggle.com/c/titanic/data) de Kaggle, donde
encontramos los datos de 891 pasajeros y que utilizaremos para crear
nuestro modelo de predicción.

Para lectura de datos, utilizaremos alguna de las variantes de la
función [<tt>read\_</tt>](http://r4ds.had.co.nz/data-import.html). A
continuación, podemos inspeccionar el contenido de la tabla de datos,
que se almacena en formato
[<tt>tibble</tt>](http://r4ds.had.co.nz/tibbles.html).

``` r
library(tidyverse)
```

    ## ── Attaching packages ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse 1.2.1 ──

    ## ✔ ggplot2 3.1.0       ✔ purrr   0.3.1  
    ## ✔ tibble  2.0.1       ✔ dplyr   0.8.0.1
    ## ✔ tidyr   0.8.3       ✔ stringr 1.4.0  
    ## ✔ readr   1.3.1       ✔ forcats 0.4.0

    ## ── Conflicts ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()

``` r
data_raw <- read_csv('data/train.csv')
```

    ## Parsed with column specification:
    ## cols(
    ##   PassengerId = col_double(),
    ##   Survived = col_double(),
    ##   Pclass = col_double(),
    ##   Name = col_character(),
    ##   Sex = col_character(),
    ##   Age = col_double(),
    ##   SibSp = col_double(),
    ##   Parch = col_double(),
    ##   Ticket = col_character(),
    ##   Fare = col_double(),
    ##   Cabin = col_character(),
    ##   Embarked = col_character()
    ## )

``` r
data_raw # str(data_raw) , glimpse(data_raw)
```

    ## # A tibble: 891 x 12
    ##    PassengerId Survived Pclass Name  Sex     Age SibSp Parch Ticket  Fare
    ##          <dbl>    <dbl>  <dbl> <chr> <chr> <dbl> <dbl> <dbl> <chr>  <dbl>
    ##  1           1        0      3 Brau… male     22     1     0 A/5 2…  7.25
    ##  2           2        1      1 Cumi… fema…    38     1     0 PC 17… 71.3 
    ##  3           3        1      3 Heik… fema…    26     0     0 STON/…  7.92
    ##  4           4        1      1 Futr… fema…    35     1     0 113803 53.1 
    ##  5           5        0      3 Alle… male     35     0     0 373450  8.05
    ##  6           6        0      3 Mora… male     NA     0     0 330877  8.46
    ##  7           7        0      1 McCa… male     54     0     0 17463  51.9 
    ##  8           8        0      3 Pals… male      2     3     1 349909 21.1 
    ##  9           9        1      3 John… fema…    27     0     2 347742 11.1 
    ## 10          10        1      2 Nass… fema…    14     1     0 237736 30.1 
    ## # … with 881 more rows, and 2 more variables: Cabin <chr>, Embarked <chr>

Estado del conjunto de datos
----------------------------

Podemos identificar los valores perdidos de la tabla utilizando
<tt>df\_status</tt>, del paquete
[<tt>funModeling</tt>](https://livebook.datascienceheroes.com/exploratory-data-analysis.html#dataset-health-status).

``` r
library(funModeling)
```

    ## Loading required package: Hmisc

    ## Loading required package: lattice

    ## Loading required package: survival

    ## Loading required package: Formula

    ## 
    ## Attaching package: 'Hmisc'

    ## The following objects are masked from 'package:dplyr':
    ## 
    ##     src, summarize

    ## The following objects are masked from 'package:base':
    ## 
    ##     format.pval, units

    ## funModeling v.1.7 :)
    ## Examples and tutorials at livebook.datascienceheroes.com

``` r
df_status(data_raw)
```

    ##       variable q_zeros p_zeros q_na  p_na q_inf p_inf      type unique
    ## 1  PassengerId       0    0.00    0  0.00     0     0   numeric    891
    ## 2     Survived     549   61.62    0  0.00     0     0   numeric      2
    ## 3       Pclass       0    0.00    0  0.00     0     0   numeric      3
    ## 4         Name       0    0.00    0  0.00     0     0 character    891
    ## 5          Sex       0    0.00    0  0.00     0     0 character      2
    ## 6          Age       0    0.00  177 19.87     0     0   numeric     88
    ## 7        SibSp     608   68.24    0  0.00     0     0   numeric      7
    ## 8        Parch     678   76.09    0  0.00     0     0   numeric      7
    ## 9       Ticket       0    0.00    0  0.00     0     0 character    681
    ## 10        Fare      15    1.68    0  0.00     0     0   numeric    248
    ## 11       Cabin       0    0.00  687 77.10     0     0 character    147
    ## 12    Embarked       0    0.00    2  0.22     0     0 character      3

Algunas observaciones interesantes:

-   Los valores de *PassengerId* y *Name* son únicos
-   Existen dos valores diferentes para *Survived*, que es nuestro
    objetivo de clasificación
-   No sobrevivieron 549 pasajeros (61.62%)
-   Aparecen numerosos valores perdidos (*na*) en las variables *Age* y
    *Cabin*

Parte de estas situaciones se pueden identificar y procesar directamente
manipulando la tabla <tt>df\_status</tt>:

``` r
status <- df_status(data_raw)
```

    ##       variable q_zeros p_zeros q_na  p_na q_inf p_inf      type unique
    ## 1  PassengerId       0    0.00    0  0.00     0     0   numeric    891
    ## 2     Survived     549   61.62    0  0.00     0     0   numeric      2
    ## 3       Pclass       0    0.00    0  0.00     0     0   numeric      3
    ## 4         Name       0    0.00    0  0.00     0     0 character    891
    ## 5          Sex       0    0.00    0  0.00     0     0 character      2
    ## 6          Age       0    0.00  177 19.87     0     0   numeric     88
    ## 7        SibSp     608   68.24    0  0.00     0     0   numeric      7
    ## 8        Parch     678   76.09    0  0.00     0     0   numeric      7
    ## 9       Ticket       0    0.00    0  0.00     0     0 character    681
    ## 10        Fare      15    1.68    0  0.00     0     0   numeric    248
    ## 11       Cabin       0    0.00  687 77.10     0     0 character    147
    ## 12    Embarked       0    0.00    2  0.22     0     0 character      3

``` r
## columnas con NAs
na_cols <- status %>%
  filter(p_na > 70) %>%
  select(variable)
## columnas con valores diferentes
dif_cols <- status %>%
  filter(unique > 0.8 * nrow(data_raw)) %>%
  select(variable)

## eliminar columnas
remove_cols <- bind_rows( 
  list(na_cols, dif_cols)
)
# Se borran las variables que ¿no? están en la lista
data_reduced <- data_raw %>%
  select(-one_of(remove_cols$variable))
```

Predictor básico: todos los pasajeros mueren
--------------------------------------------

### Tablas

A continuación, nos centramos en los valores de la variable *Survived*.
Podemos obtener un resumen en forma de tabla utilizando <tt>table</tt>,
que muestra un recuento basado en la variable(s) usada como argumento.
De forma similar, <tt>prop.table</tt> muestra un recuento normalizado al
intervalo \[0, 1\].

``` r
table(data_raw$Survived)
```

    ## 
    ##   0   1 
    ## 549 342

``` r
prop.table(table(data_raw$Survived))
```

    ## 
    ##         0         1 
    ## 0.6161616 0.3838384

### Realizar predicción con datos de test (1)

Dado que alrededor del 60% de los pasajeros mueren, podemos asumir un
clasificador muy sencillo que asigna a todos los pasajeros *Survived =
0*. Con este clasificador esperamos una tasa de acierto correspondiente
del 60%.

Para ello, vamos a leer el fichero *test.csv*, seleccionar solo la
columna *PassengerId* y asignar 0 a *Survived*, utilizando dos
funcionalidades de <tt>dplyr</tt>:

-   [Pipes](http://r4ds.had.co.nz/pipes.html): Permiten encadenar
    operaciones de transformación utlizando <tt>%&gt;%</tt>, de forma
    similar al operador \| en bash.

-   [Funciones de transformación de
    datos](http://r4ds.had.co.nz/transform.html): Permiten generar una
    nueva tabla de datos a partir de la tabla recibida como primer
    argumento o a través del *pipe*. Emplearemos
    [<tt>select</tt>](http://r4ds.had.co.nz/transform.html#select-columns-with-select)
    (para selección de columnas) y
    [<tt>mutate</tt>](http://r4ds.had.co.nz/transform.html#add-new-variables-with-mutate)
    (para generar nuevas columnas o modificar las ya existentes).

La tabla de datos obtenidas, que mostramos utilizando *()*, se guarda en
el fichero *all-died.csv*, que enviaremos como *submission* a la
[competición de Kaggle](https://www.kaggle.com/c/titanic/submit).

``` r
(test <- 
  read_csv('data/test.csv') %>%
  select(one_of('PassengerId')) %>%
  mutate(Survived = 0)
)
```

    ## Parsed with column specification:
    ## cols(
    ##   PassengerId = col_double(),
    ##   Pclass = col_double(),
    ##   Name = col_character(),
    ##   Sex = col_character(),
    ##   Age = col_double(),
    ##   SibSp = col_double(),
    ##   Parch = col_double(),
    ##   Ticket = col_character(),
    ##   Fare = col_double(),
    ##   Cabin = col_character(),
    ##   Embarked = col_character()
    ## )

    ## # A tibble: 418 x 2
    ##    PassengerId Survived
    ##          <dbl>    <dbl>
    ##  1         892        0
    ##  2         893        0
    ##  3         894        0
    ##  4         895        0
    ##  5         896        0
    ##  6         897        0
    ##  7         898        0
    ##  8         899        0
    ##  9         900        0
    ## 10         901        0
    ## # … with 408 more rows

``` r
write_csv(test, "all-died.csv")
```

``` r
# ![Resultados predicción 1](./all-died.csv)
```

Predictor refinado: todos los hombres mueren
--------------------------------------------

### Filtrado de datos

Si filtramos las filas de los pasajeros que sobrevivieron, observamos
que los datos indican una mayor cantidad de mujeres entre los
supervivientes.

``` r
filtered <-
  data_raw %>%
  filter(Survived == 1) %>%
  arrange(Age)
```

De hecho, podemos verlo con la función <tt>table</tt>. En este caso, el
primer argumento es la variable que se utiliza para las filas y el
segundo argumento la variable que se utiliza para las columnas. La tabla
indica que, por ejemplo, de todos los pasajeros del barco, un 52% eran
hombres que murieron.

``` r
prop.table(table(data_raw$Sex, data_raw$Survived))
```

    ##         
    ##                   0          1
    ##   female 0.09090909 0.26150393
    ##   male   0.52525253 0.12233446

### Histogramas

También podemos dibujar un histograma de los datos utilizando
[<tt>ggplot</tt>](http://r4ds.had.co.nz/data-visualisation.html).

[<tt>ggplot</tt>](http://r4ds.had.co.nz/data-visualisation.html) es un
paquete de visualización de datos muy completo que ofrece una gramática
para construir gráficos de una manera semi-declarativa, especificando
las propiedades de las diferentes capas visuales.

Generalmente, en la instrucción <tt>ggplot</tt> especificamos las
características comunes para todo el gráfico, incluyendo el conjunto de
datos que vamos a visualizar. Las capas visuales, como por ejemplo el
histograma <tt>geom\_histogram</tt>, se añaden al gráfico general. Cada
una de las capas puede establecer a su vez diferentes parámetros
visuales mediante el argumento <tt>aes</tt>. En este caso, estamos
indicando que queremos que las barras del histograma se coloreen según
el valor de la variable *Survived* (parámetro *fill*).

``` r
library(ggplot2)
ggplot(data_raw) +
  geom_histogram(aes(x = Age, fill = as.factor(Survived)), binwidth = 1)
```

    ## Warning: Removed 177 rows containing non-finite values (stat_bin).

![](titanic_files/figure-markdown_github/unnamed-chunk-9-1.png) Podemos
modificar el formato del gráfico: paleta de colores, etiquetas, etc.
utilizando <tt>ggthemes</tt> y <tt>scales</tt>.

``` r
library(ggthemes)
library(scales)
```

    ## 
    ## Attaching package: 'scales'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     discard

    ## The following object is masked from 'package:readr':
    ## 
    ##     col_factor

``` r
plotdata <- 
  data_raw %>%
  mutate(Survived = as.factor(Survived))
ggplot(plotdata) +
  geom_histogram(aes(x = Age, fill = Survived), binwidth = 1) +
  labs(title = "Titanic survivors", x = "Age", y = "# Passengers", fill = "Survived") +
  theme_hc() + scale_fill_hc(labels = c('Yes', 'No'))
```

    ## Warning: Removed 177 rows containing non-finite values (stat_bin).

![](titanic_files/figure-markdown_github/unnamed-chunk-10-1.png) Si
mostramos únicamente el histograma para los pasajeros que sobrevivieron,
la relación se ve aún más clara. En el siguiente gráfico hemos
modificado el parámetro *bindwidth*, de forma que agrupamos los
pasajeros por tramos de edad más amplios.

``` r
ggplot(filter(data_raw, Survived == 1)) +
  geom_histogram(aes(x = Age, fill = as.factor(Sex)), binwidth = 15)
```

    ## Warning: Removed 52 rows containing non-finite values (stat_bin).

![](titanic_files/figure-markdown_github/unnamed-chunk-11-1.png)

### Estudio de correlaciones

Para comprobar numéricamente la relación entre las variables, incluyendo
*Sex* y *Survived*, realizamos un estudio de correlaciones con
<tt>correlation\_table</tt>. Atención, porque esta función necesita que
los datos estén expresados de forma numérica.

``` r
correlation_table(data_raw, target='Survived')
```

    ##      Variable Survived
    ## 1    Survived     1.00
    ## 2        Fare     0.27
    ## 3       Parch     0.09
    ## 4 PassengerId     0.03
    ## 5       SibSp    -0.02
    ## 6         Age    -0.08
    ## 7      Pclass    -0.36

``` r
d <- 
  data_raw %>%
  mutate(Sex_Num = ifelse(Sex == 'male', 0, 1))

cor(d$Survived, d$Sex_Num)
```

    ## [1] 0.5433514

### Realizar predicción con datos de test (2)

La predicción de ‘todos los hombres mueren’ es sencilla de obtener
asignando el valor de *Survived* según el valor de *Sex* mediante
<tt>ifelse</tt>.

``` r
(test <- 
  read_csv('data/test.csv') %>%
  mutate(Survived = ifelse(Sex == 'female', 1, 0)) %>%
  select(one_of('PassengerId', 'Survived'))
)
```

    ## Parsed with column specification:
    ## cols(
    ##   PassengerId = col_double(),
    ##   Pclass = col_double(),
    ##   Name = col_character(),
    ##   Sex = col_character(),
    ##   Age = col_double(),
    ##   SibSp = col_double(),
    ##   Parch = col_double(),
    ##   Ticket = col_character(),
    ##   Fare = col_double(),
    ##   Cabin = col_character(),
    ##   Embarked = col_character()
    ## )

    ## # A tibble: 418 x 2
    ##    PassengerId Survived
    ##          <dbl>    <dbl>
    ##  1         892        0
    ##  2         893        1
    ##  3         894        0
    ##  4         895        0
    ##  5         896        1
    ##  6         897        0
    ##  7         898        1
    ##  8         899        0
    ##  9         900        1
    ## 10         901        0
    ## # … with 408 more rows

``` r
write_csv(test, "men-died.csv")
```

Los resultados en Kaggle de esta predicción son algo mejores que los
obtenidos con la predicción anterior.

``` r
# ![Resultados predicción 2](./men-died-submission.png)
```

Predictor refinado: todos los hombres mueren, las mujeres en 3ª clase que pagan &gt;= 20 mueren
-----------------------------------------------------------------------------------------------

### Transformación de datos

En la sección anterior hemos visto que existen otras variables
correladas con *Survived*; además de *Sex*, tenemos *Fare* y *Pclass*.
Ambas son indicativas del nivel económico de los pasajeros.

``` r
cor(d$Pclass, d$Fare)
```

    ## [1] -0.5494996

Podemos estudiar *Pclass* para comprobar si es así, por ejemplo creando
un histograma con esta variable en el eje x. En este caso utilizarmos un
<tt>geom\_bar</tt>, que también por defecto realiza un recuento de
ocurrencias.

``` r
ggplot(data_raw) +
  geom_bar(aes(x = Pclass, fill = as.factor(Survived)))
```

![](titanic_files/figure-markdown_github/unnamed-chunk-16-1.png)
Efectivamente, *Pclass* es también determinante para la supervivencia,
como ya sabíamos que ocurría con *Sex*. Por lo tanto, parece conveniente
estudiar la influencia conjunta de ambas variables en la predicción. Una
forma de hacerlo es construir una tabla que nos muestre las tasas de
supervivencia por sexo y clase. Utilizamos para ello las funciones de
<tt>dplyr</tt>:

-   Resumen: La función
    [<tt>summarise</tt>](http://r4ds.had.co.nz/transform.html#grouped-summaries-with-summarise)
    permite realizar operaciones de resumen sobre el conjunto de datos:
    agregaciones, sumas, etc.

-   Agrupación: La función
    [<tt>group\_by</tt>](http://r4ds.had.co.nz/transform.html#grouping-by-multiple-variables)
    permite agrupar los datos en bloques, a los que se aplica
    individualmente el <tt>summarise</tt>.

A continuación se muestran dos ejemplos sencillos de <tt>group\_by</tt>
y <tt>summarise</tt>, que ilustran respectivamente: (1) cómo obtener la
edad media por clase (2) % de supervivencia respecto al total por clase
y sexo.

``` r
data_raw %>%
  group_by(Pclass) %>%
  summarise(AvgAge = mean(Age, na.rm = TRUE) )
```

    ## # A tibble: 3 x 2
    ##   Pclass AvgAge
    ##    <dbl>  <dbl>
    ## 1      1   38.2
    ## 2      2   29.9
    ## 3      3   25.1

``` r
data_raw %>%
  group_by(Pclass, Sex) %>%
  summarise(Survived_G = sum(Survived) / length(Survived) )
```

    ## # A tibble: 6 x 3
    ## # Groups:   Pclass [3]
    ##   Pclass Sex    Survived_G
    ##    <dbl> <chr>       <dbl>
    ## 1      1 female      0.968
    ## 2      1 male        0.369
    ## 3      2 female      0.921
    ## 4      2 male        0.157
    ## 5      3 female      0.5  
    ## 6      3 male        0.135

Finalmente, se presenta el código para la tabla de tasas de
superviviencia por sexo, clase y precio del billete. Para facilitar el
cálculo, se han creado varios intervalos de precios mediante
[<tt>case\_when</tt>](https://www.rdocumentation.org/packages/dplyr/versions/0.7.3/topics/case_when),
que permite implementar múltiples condiciones de tipo if-else de manera
simple.

``` r
data_raw %>%
  filter(!is.na(Fare)) %>%
  mutate(Fare_Interval = case_when(
    Fare >= 30 ~ '30+',
    Fare >= 20 & Fare < 30 ~ '20-30',
    Fare < 20 & Fare >= 10 ~ '10-20',
    Fare < 10 ~ '<10')) %>%
  group_by(Fare_Interval, Pclass, Sex) %>%
  summarise(Survived_G = sum(Survived) / length(Survived)) %>%
  filter(Survived_G > 0.0) %>%
  arrange(Pclass, desc(Survived_G))
```

    ## # A tibble: 18 x 4
    ## # Groups:   Fare_Interval, Pclass [9]
    ##    Fare_Interval Pclass Sex    Survived_G
    ##    <chr>          <dbl> <chr>       <dbl>
    ##  1 30+                1 female      0.977
    ##  2 20-30              1 female      0.833
    ##  3 20-30              1 male        0.4  
    ##  4 30+                1 male        0.384
    ##  5 30+                2 female      1    
    ##  6 10-20              2 female      0.914
    ##  7 20-30              2 female      0.9  
    ##  8 30+                2 male        0.214
    ##  9 20-30              2 male        0.16 
    ## 10 10-20              2 male        0.159
    ## 11 <10                3 female      0.594
    ## 12 10-20              3 female      0.581
    ## 13 20-30              3 female      0.333
    ## 14 30+                3 male        0.24 
    ## 15 10-20              3 male        0.237
    ## 16 20-30              3 male        0.125
    ## 17 30+                3 female      0.125
    ## 18 <10                3 male        0.112

### Realizar predicción con datos de test (3)

Esta predicción establece que mueren todos los hombres y las mujeres en
tercera clase (que pagaron más de 20$, que se añade como condición
adicional).

``` r
(test <- 
  read_csv('data/test.csv') %>%
  mutate(Survived = case_when(
    Sex == 'female' & Pclass == 3 & Fare >= 20 ~ 0,
    Sex == 'male' ~ 0,
    TRUE ~ 1)) %>%
  select(one_of('PassengerId', 'Survived'))
)
```

    ## Parsed with column specification:
    ## cols(
    ##   PassengerId = col_double(),
    ##   Pclass = col_double(),
    ##   Name = col_character(),
    ##   Sex = col_character(),
    ##   Age = col_double(),
    ##   SibSp = col_double(),
    ##   Parch = col_double(),
    ##   Ticket = col_character(),
    ##   Fare = col_double(),
    ##   Cabin = col_character(),
    ##   Embarked = col_character()
    ## )

    ## # A tibble: 418 x 2
    ##    PassengerId Survived
    ##          <dbl>    <dbl>
    ##  1         892        0
    ##  2         893        1
    ##  3         894        0
    ##  4         895        0
    ##  5         896        1
    ##  6         897        0
    ##  7         898        1
    ##  8         899        0
    ##  9         900        1
    ## 10         901        0
    ## # … with 408 more rows

``` r
write_csv(test, "men-and-some-women-died.csv")
```

``` r
# ![Resultados predicción 3](./men-and-some-women-died-submission.png)
```

Predicción automática
---------------------

Los modelos de predicción anteriores expresan un conjunto de reglas
heurísticas obtenidas mediante análisis exploratorio de los datos (EDA,
en inglés). Para crear un modelo de clasificación de forma automática
utilizaremos [<tt>caret</tt>](http://topepo.github.io/caret/). Este
paquete es un *wrapper* para numerosos algoritmos de aprendizaje
automático, ofreciendo una API simple y unificada.

En este ejemplo utilizamos árboles de regresión (CART, *classification
and regression trees*). Estos árboles admiten un solo parámetro
denominado *cp* y que denota la complejidad permitida para los árboles
resultado –una medida calculada a partir de la profundidad, la amplitud
y el número de variables del árbol.

``` r
library(caret)
```

    ## 
    ## Attaching package: 'caret'

    ## The following object is masked from 'package:survival':
    ## 
    ##     cluster

    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

``` r
data <-
  data_raw %>%
  mutate(Survived = as.factor(ifelse(Survived == 1, 'Yes', 'No'))) %>%
  mutate(Pclass = as.factor(Pclass)) %>%
  mutate(Fare_Interval = as.factor(
    case_when(
      Fare >= 30 ~ 'More.than.30',
      Fare >= 20 & Fare < 30 ~ 'Between.20.30',
      Fare < 20 & Fare >= 10 ~ 'Between.10.20',
      Fare < 10 ~ 'Less.than.10'))) %>%
  select(Survived, Pclass, Sex, Fare_Interval)

# Parámetros
rpartCtrl <- trainControl(classProbs = TRUE)
rpartParametersGrid <- expand.grid(.cp = c(0.01))

# Creación de conjuntos de datos de entrenamiento (70%) y validación (20%)
set.seed(0)
trainIndex <- createDataPartition(data$Survived, p = .7, list = FALSE, times = 1)
train <- data[trainIndex, ] 
val   <- data[-trainIndex, ]

# Aprendizaje del modelo
rpartModel <- train(Survived ~ ., data = train, method = "rpart", metric = "Accuracy", trControl = rpartCtrl, tuneGrid = rpartParametersGrid)
```

Podemos visualizar el modelo de reglas.

``` r
library(partykit)
```

    ## Loading required package: grid

    ## Loading required package: libcoin

    ## Loading required package: mvtnorm

``` r
library(rattle)
```

    ## Rattle: A free graphical interface for data science with R.
    ## Versión 5.2.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
    ## Escriba 'rattle()' para agitar, sacudir y  rotar sus datos.

``` r
rpartModel_party <- as.party(rpartModel$finalModel)
plot(rpartModel_party)
```

![](titanic_files/figure-markdown_github/unnamed-chunk-22-1.png)

``` r
fancyRpartPlot(rpartModel$finalModel)
```

![](titanic_files/figure-markdown_github/unnamed-chunk-22-2.png)

``` r
asRules(rpartModel$finalModel)
```

    ## 
    ##  Rule number: 7 [.outcome=Yes cover=110 (18%) prob=0.95]
    ##    Sexmale< 0.5
    ##    Pclass3< 0.5
    ## 
    ##  Rule number: 27 [.outcome=Yes cover=80 (13%) prob=0.64]
    ##    Sexmale< 0.5
    ##    Pclass3>=0.5
    ##    Fare_IntervalMore.than.30< 0.5
    ##    Fare_IntervalBetween.20.30< 0.5
    ## 
    ##  Rule number: 26 [.outcome=No cover=17 (3%) prob=0.29]
    ##    Sexmale< 0.5
    ##    Pclass3>=0.5
    ##    Fare_IntervalMore.than.30< 0.5
    ##    Fare_IntervalBetween.20.30>=0.5
    ## 
    ##  Rule number: 2 [.outcome=No cover=407 (65%) prob=0.19]
    ##    Sexmale>=0.5
    ## 
    ##  Rule number: 12 [.outcome=No cover=11 (2%) prob=0.00]
    ##    Sexmale< 0.5
    ##    Pclass3>=0.5
    ##    Fare_IntervalMore.than.30>=0.5

Y calcular la precisión del modelo sobre los datos de validación.

``` r
prediction <- predict(rpartModel, val, type = "raw") 
cm_train <- confusionMatrix(prediction, val[["Survived"]])
cm_train
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  No Yes
    ##        No  145  34
    ##        Yes  19  68
    ##                                          
    ##                Accuracy : 0.8008         
    ##                  95% CI : (0.7476, 0.847)
    ##     No Information Rate : 0.6165         
    ##     P-Value [Acc > NIR] : 7.74e-11       
    ##                                          
    ##                   Kappa : 0.5666         
    ##  Mcnemar's Test P-Value : 0.05447        
    ##                                          
    ##             Sensitivity : 0.8841         
    ##             Specificity : 0.6667         
    ##          Pos Pred Value : 0.8101         
    ##          Neg Pred Value : 0.7816         
    ##              Prevalence : 0.6165         
    ##          Detection Rate : 0.5451         
    ##    Detection Prevalence : 0.6729         
    ##       Balanced Accuracy : 0.7754         
    ##                                          
    ##        'Positive' Class : No             
    ## 

``` r
prediction <- predict(rpartModel, val, type = "prob")  # probabilidad de superviviencia
```

Podemos considerar también otras medidas de calidad del clasificador;
por ejemplo, ROC:

``` r
rpartCtrl <- trainControl(verboseIter = F, classProbs = TRUE, summaryFunction = twoClassSummary)
rpartModel <- train(Survived ~ ., data = train, method = "rpart", metric = "ROC", trControl = rpartCtrl, tuneGrid = rpartParametersGrid)

library(pROC)
```

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Attaching package: 'pROC'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
predictionValidationProb <- predict(rpartModel, val, type = "prob")
auc <- roc(val$Survived, predictionValidationProb[["Yes"]], levels = unique(val[["Survived"]]))
roc_validation <- plot.roc(auc, ylim=c(0,1), type = "S" , print.thres = T, main=paste('Validation AUC:', round(auc$auc[[1]], 2)))
```

![](titanic_files/figure-markdown_github/unnamed-chunk-24-1.png) Y
aplicar cross-validation:

``` r
rpartCtrl <- trainControl(verboseIter = T, classProbs = TRUE, summaryFunction = twoClassSummary, method = "cv", number = 10)
rpartModel <- train(Survived ~ ., data = train, method = "rpart", metric = "ROC", trControl = rpartCtrl, tuneGrid = rpartParametersGrid)
```

    ## + Fold01: cp=0.01 
    ## - Fold01: cp=0.01 
    ## + Fold02: cp=0.01 
    ## - Fold02: cp=0.01 
    ## + Fold03: cp=0.01 
    ## - Fold03: cp=0.01 
    ## + Fold04: cp=0.01 
    ## - Fold04: cp=0.01 
    ## + Fold05: cp=0.01 
    ## - Fold05: cp=0.01 
    ## + Fold06: cp=0.01 
    ## - Fold06: cp=0.01 
    ## + Fold07: cp=0.01 
    ## - Fold07: cp=0.01 
    ## + Fold08: cp=0.01 
    ## - Fold08: cp=0.01 
    ## + Fold09: cp=0.01 
    ## - Fold09: cp=0.01 
    ## + Fold10: cp=0.01 
    ## - Fold10: cp=0.01 
    ## Aggregating results
    ## Fitting final model on full training set

``` r
predictionValidationProb <- predict(rpartModel, val, type = "prob")
auc <- roc(val$Survived, predictionValidationProb[["Yes"]], levels = unique(val[["Survived"]]))
roc_validation <- plot.roc(auc, ylim=c(0,1), type = "S" , print.thres = T, main=paste('Validation AUC:', round(auc$auc[[1]], 2)))
```

![](titanic_files/figure-markdown_github/unnamed-chunk-25-1.png)

Podemos utilizar otras técnicas, como Random Forest:

``` r
rfModel <- train(Survived ~ ., data = train, method = "rf", metric = "ROC", trControl = rpartCtrl)
```

    ## + Fold01: mtry=2 
    ## - Fold01: mtry=2 
    ## + Fold01: mtry=4 
    ## - Fold01: mtry=4 
    ## + Fold01: mtry=6 
    ## - Fold01: mtry=6 
    ## + Fold02: mtry=2 
    ## - Fold02: mtry=2 
    ## + Fold02: mtry=4 
    ## - Fold02: mtry=4 
    ## + Fold02: mtry=6 
    ## - Fold02: mtry=6 
    ## + Fold03: mtry=2 
    ## - Fold03: mtry=2 
    ## + Fold03: mtry=4 
    ## - Fold03: mtry=4 
    ## + Fold03: mtry=6 
    ## - Fold03: mtry=6 
    ## + Fold04: mtry=2 
    ## - Fold04: mtry=2 
    ## + Fold04: mtry=4 
    ## - Fold04: mtry=4 
    ## + Fold04: mtry=6 
    ## - Fold04: mtry=6 
    ## + Fold05: mtry=2 
    ## - Fold05: mtry=2 
    ## + Fold05: mtry=4 
    ## - Fold05: mtry=4 
    ## + Fold05: mtry=6 
    ## - Fold05: mtry=6 
    ## + Fold06: mtry=2 
    ## - Fold06: mtry=2 
    ## + Fold06: mtry=4 
    ## - Fold06: mtry=4 
    ## + Fold06: mtry=6 
    ## - Fold06: mtry=6 
    ## + Fold07: mtry=2 
    ## - Fold07: mtry=2 
    ## + Fold07: mtry=4 
    ## - Fold07: mtry=4 
    ## + Fold07: mtry=6 
    ## - Fold07: mtry=6 
    ## + Fold08: mtry=2 
    ## - Fold08: mtry=2 
    ## + Fold08: mtry=4 
    ## - Fold08: mtry=4 
    ## + Fold08: mtry=6 
    ## - Fold08: mtry=6 
    ## + Fold09: mtry=2 
    ## - Fold09: mtry=2 
    ## + Fold09: mtry=4 
    ## - Fold09: mtry=4 
    ## + Fold09: mtry=6 
    ## - Fold09: mtry=6 
    ## + Fold10: mtry=2 
    ## - Fold10: mtry=2 
    ## + Fold10: mtry=4 
    ## - Fold10: mtry=4 
    ## + Fold10: mtry=6 
    ## - Fold10: mtry=6 
    ## Aggregating results
    ## Selecting tuning parameters
    ## Fitting mtry = 2 on full training set

``` r
predictionValidationProb <- predict(rfModel, val, type = "prob")
auc <- roc(val$Survived, predictionValidationProb[["Yes"]], levels = unique(val[["Survived"]]))
roc_validation <- plot.roc(auc, ylim=c(0,1), type = "S" , print.thres = T, main=paste('Validation AUC:', round(auc$auc[[1]], 2)))
```

![](titanic_files/figure-markdown_github/unnamed-chunk-26-1.png)

Valores perdidos
----------------

Internamente,
[<tt>rpart</tt>](https://cran.r-project.org/web/packages/rpart/vignettes/longintro.pdf)
utiliza un procedimiento para estimar los valores perdidos. Otras
técnicas se limitan a omitir las filas con valores perdidos, lo que
significa perder muchos datos significativos. Es conveniente por tanto
gestionar los valores perdidos de una forma más controlada. \#\#\#
MissingDataGUI [MissingDataGUI](https://github.com/chxy/MissingDataGUI)
es una herramienta para explorar y reparar valores perdidos. Si bien su
interfaz gráfica puede facilitar la gestión, la falta de documentación
la hacen difícil de utilizar.

``` r
# library(MissingDataGUI)
# if (interactive()) {
#        MissingDataGUI()
# }
```

### VIM

[VIM](#vim) facilita la visualización de la distribución de los valores
perdidos. Puede utilizarse en combinación con
[<tt>funModeling</tt>](https://livebook.datascienceheroes.com/exploratory-data-analysis.html#dataset-health-status).

``` r
library(VIM)
```

    ## Loading required package: colorspace

    ## 
    ## Attaching package: 'colorspace'

    ## The following object is masked from 'package:pROC':
    ## 
    ##     coords

    ## Loading required package: data.table

    ## 
    ## Attaching package: 'data.table'

    ## The following objects are masked from 'package:dplyr':
    ## 
    ##     between, first, last

    ## The following object is masked from 'package:purrr':
    ## 
    ##     transpose

    ## VIM is ready to use. 
    ##  Since version 4.0.0 the GUI is in its own package VIMGUI.
    ## 
    ##           Please use the package to use the new (and old) GUI.

    ## Suggestions and bug-reports can be submitted at: https://github.com/alexkowa/VIM/issues

    ## 
    ## Attaching package: 'VIM'

    ## The following object is masked from 'package:datasets':
    ## 
    ##     sleep

``` r
# aggr(data_raw, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data_raw), cex.axis=.7, gap=3, ylab=c("Histogram of missing data", "Pattern"))
```

### mice

[MICE](https://www.r-bloggers.com/imputing-missing-data-with-r-mice-package/)
es una de las bibliotecas más completas para realizar imputación de
valores perdidos.

``` r
library(mice)
```

    ## 
    ## Attaching package: 'mice'

    ## The following object is masked from 'package:tidyr':
    ## 
    ##     complete

    ## The following objects are masked from 'package:base':
    ## 
    ##     cbind, rbind

``` r
colnames(data_raw) # valores perdidos: Cabin, Age, Embarked
```

    ##  [1] "PassengerId" "Survived"    "Pclass"      "Name"        "Sex"        
    ##  [6] "Age"         "SibSp"       "Parch"       "Ticket"      "Fare"       
    ## [11] "Cabin"       "Embarked"

``` r
imputation <- mice(data_raw, method = c("", "", "", "", "", "mean", "", "", "", "", "cart", "cart"))
```

    ## 
    ##  iter imp variable
    ##   1   1  Age
    ##   1   2  Age
    ##   1   3  Age
    ##   1   4  Age
    ##   1   5  Age
    ##   2   1  Age
    ##   2   2  Age
    ##   2   3  Age
    ##   2   4  Age
    ##   2   5  Age
    ##   3   1  Age
    ##   3   2  Age
    ##   3   3  Age
    ##   3   4  Age
    ##   3   5  Age
    ##   4   1  Age
    ##   4   2  Age
    ##   4   3  Age
    ##   4   4  Age
    ##   4   5  Age
    ##   5   1  Age
    ##   5   2  Age
    ##   5   3  Age
    ##   5   4  Age
    ##   5   5  Age

    ## Warning: Number of logged events: 5

``` r
imputation
```

    ## Class: mids
    ## Number of multiple imputations:  5 
    ## Imputation methods:
    ## PassengerId    Survived      Pclass        Name         Sex         Age 
    ##          ""          ""          ""          ""          ""      "mean" 
    ##       SibSp       Parch      Ticket        Fare       Cabin    Embarked 
    ##          ""          ""          ""          ""          ""          "" 
    ## PredictorMatrix:
    ##             PassengerId Survived Pclass Name Sex Age SibSp Parch Ticket
    ## PassengerId           0        1      1    0   0   1     1     1      0
    ## Survived              1        0      1    0   0   1     1     1      0
    ## Pclass                1        1      0    0   0   1     1     1      0
    ## Name                  1        1      1    0   0   1     1     1      0
    ## Sex                   1        1      1    0   0   1     1     1      0
    ## Age                   1        1      1    0   0   0     1     1      0
    ##             Fare Cabin Embarked
    ## PassengerId    1     0        0
    ## Survived       1     0        0
    ## Pclass         1     0        0
    ## Name           1     0        0
    ## Sex            1     0        0
    ## Age            1     0        0
    ## Number of logged events:  5 
    ##   it im dep     meth      out
    ## 1  0  0     constant     Name
    ## 2  0  0     constant      Sex
    ## 3  0  0     constant   Ticket
    ## 4  0  0     constant    Cabin
    ## 5  0  0     constant Embarked

``` r
complete(imputation)
```

    ##     PassengerId Survived Pclass
    ## 1             1        0      3
    ## 2             2        1      1
    ## 3             3        1      3
    ## 4             4        1      1
    ## 5             5        0      3
    ## 6             6        0      3
    ## 7             7        0      1
    ## 8             8        0      3
    ## 9             9        1      3
    ## 10           10        1      2
    ## 11           11        1      3
    ## 12           12        1      1
    ## 13           13        0      3
    ## 14           14        0      3
    ## 15           15        0      3
    ## 16           16        1      2
    ## 17           17        0      3
    ## 18           18        1      2
    ## 19           19        0      3
    ## 20           20        1      3
    ## 21           21        0      2
    ## 22           22        1      2
    ## 23           23        1      3
    ## 24           24        1      1
    ## 25           25        0      3
    ## 26           26        1      3
    ## 27           27        0      3
    ## 28           28        0      1
    ## 29           29        1      3
    ## 30           30        0      3
    ## 31           31        0      1
    ## 32           32        1      1
    ## 33           33        1      3
    ## 34           34        0      2
    ## 35           35        0      1
    ## 36           36        0      1
    ## 37           37        1      3
    ## 38           38        0      3
    ## 39           39        0      3
    ## 40           40        1      3
    ## 41           41        0      3
    ## 42           42        0      2
    ## 43           43        0      3
    ## 44           44        1      2
    ## 45           45        1      3
    ## 46           46        0      3
    ## 47           47        0      3
    ## 48           48        1      3
    ## 49           49        0      3
    ## 50           50        0      3
    ## 51           51        0      3
    ## 52           52        0      3
    ## 53           53        1      1
    ## 54           54        1      2
    ## 55           55        0      1
    ## 56           56        1      1
    ## 57           57        1      2
    ## 58           58        0      3
    ## 59           59        1      2
    ## 60           60        0      3
    ## 61           61        0      3
    ## 62           62        1      1
    ## 63           63        0      1
    ## 64           64        0      3
    ## 65           65        0      1
    ## 66           66        1      3
    ## 67           67        1      2
    ## 68           68        0      3
    ## 69           69        1      3
    ## 70           70        0      3
    ## 71           71        0      2
    ## 72           72        0      3
    ## 73           73        0      2
    ## 74           74        0      3
    ## 75           75        1      3
    ## 76           76        0      3
    ## 77           77        0      3
    ## 78           78        0      3
    ## 79           79        1      2
    ## 80           80        1      3
    ## 81           81        0      3
    ## 82           82        1      3
    ## 83           83        1      3
    ## 84           84        0      1
    ## 85           85        1      2
    ## 86           86        1      3
    ## 87           87        0      3
    ## 88           88        0      3
    ## 89           89        1      1
    ## 90           90        0      3
    ## 91           91        0      3
    ## 92           92        0      3
    ## 93           93        0      1
    ## 94           94        0      3
    ## 95           95        0      3
    ## 96           96        0      3
    ## 97           97        0      1
    ## 98           98        1      1
    ## 99           99        1      2
    ## 100         100        0      2
    ## 101         101        0      3
    ## 102         102        0      3
    ## 103         103        0      1
    ## 104         104        0      3
    ## 105         105        0      3
    ## 106         106        0      3
    ## 107         107        1      3
    ## 108         108        1      3
    ## 109         109        0      3
    ## 110         110        1      3
    ## 111         111        0      1
    ## 112         112        0      3
    ## 113         113        0      3
    ## 114         114        0      3
    ## 115         115        0      3
    ## 116         116        0      3
    ## 117         117        0      3
    ## 118         118        0      2
    ## 119         119        0      1
    ## 120         120        0      3
    ## 121         121        0      2
    ## 122         122        0      3
    ## 123         123        0      2
    ## 124         124        1      2
    ## 125         125        0      1
    ## 126         126        1      3
    ## 127         127        0      3
    ## 128         128        1      3
    ## 129         129        1      3
    ## 130         130        0      3
    ## 131         131        0      3
    ## 132         132        0      3
    ## 133         133        0      3
    ## 134         134        1      2
    ## 135         135        0      2
    ## 136         136        0      2
    ## 137         137        1      1
    ## 138         138        0      1
    ## 139         139        0      3
    ## 140         140        0      1
    ## 141         141        0      3
    ## 142         142        1      3
    ## 143         143        1      3
    ## 144         144        0      3
    ## 145         145        0      2
    ## 146         146        0      2
    ## 147         147        1      3
    ## 148         148        0      3
    ## 149         149        0      2
    ## 150         150        0      2
    ## 151         151        0      2
    ## 152         152        1      1
    ## 153         153        0      3
    ## 154         154        0      3
    ## 155         155        0      3
    ## 156         156        0      1
    ## 157         157        1      3
    ## 158         158        0      3
    ## 159         159        0      3
    ## 160         160        0      3
    ## 161         161        0      3
    ## 162         162        1      2
    ## 163         163        0      3
    ## 164         164        0      3
    ## 165         165        0      3
    ## 166         166        1      3
    ## 167         167        1      1
    ## 168         168        0      3
    ## 169         169        0      1
    ## 170         170        0      3
    ## 171         171        0      1
    ## 172         172        0      3
    ## 173         173        1      3
    ## 174         174        0      3
    ## 175         175        0      1
    ## 176         176        0      3
    ## 177         177        0      3
    ## 178         178        0      1
    ## 179         179        0      2
    ## 180         180        0      3
    ## 181         181        0      3
    ## 182         182        0      2
    ## 183         183        0      3
    ## 184         184        1      2
    ## 185         185        1      3
    ## 186         186        0      1
    ## 187         187        1      3
    ## 188         188        1      1
    ## 189         189        0      3
    ## 190         190        0      3
    ## 191         191        1      2
    ## 192         192        0      2
    ## 193         193        1      3
    ## 194         194        1      2
    ## 195         195        1      1
    ## 196         196        1      1
    ## 197         197        0      3
    ## 198         198        0      3
    ## 199         199        1      3
    ## 200         200        0      2
    ## 201         201        0      3
    ## 202         202        0      3
    ## 203         203        0      3
    ## 204         204        0      3
    ## 205         205        1      3
    ## 206         206        0      3
    ## 207         207        0      3
    ## 208         208        1      3
    ## 209         209        1      3
    ## 210         210        1      1
    ## 211         211        0      3
    ## 212         212        1      2
    ## 213         213        0      3
    ## 214         214        0      2
    ## 215         215        0      3
    ## 216         216        1      1
    ## 217         217        1      3
    ## 218         218        0      2
    ## 219         219        1      1
    ## 220         220        0      2
    ## 221         221        1      3
    ## 222         222        0      2
    ## 223         223        0      3
    ## 224         224        0      3
    ## 225         225        1      1
    ## 226         226        0      3
    ## 227         227        1      2
    ## 228         228        0      3
    ## 229         229        0      2
    ## 230         230        0      3
    ## 231         231        1      1
    ## 232         232        0      3
    ## 233         233        0      2
    ## 234         234        1      3
    ## 235         235        0      2
    ## 236         236        0      3
    ## 237         237        0      2
    ## 238         238        1      2
    ## 239         239        0      2
    ## 240         240        0      2
    ## 241         241        0      3
    ## 242         242        1      3
    ## 243         243        0      2
    ## 244         244        0      3
    ## 245         245        0      3
    ## 246         246        0      1
    ## 247         247        0      3
    ## 248         248        1      2
    ## 249         249        1      1
    ## 250         250        0      2
    ## 251         251        0      3
    ## 252         252        0      3
    ## 253         253        0      1
    ## 254         254        0      3
    ## 255         255        0      3
    ## 256         256        1      3
    ## 257         257        1      1
    ## 258         258        1      1
    ## 259         259        1      1
    ## 260         260        1      2
    ## 261         261        0      3
    ## 262         262        1      3
    ## 263         263        0      1
    ## 264         264        0      1
    ## 265         265        0      3
    ## 266         266        0      2
    ## 267         267        0      3
    ## 268         268        1      3
    ## 269         269        1      1
    ## 270         270        1      1
    ## 271         271        0      1
    ## 272         272        1      3
    ## 273         273        1      2
    ## 274         274        0      1
    ## 275         275        1      3
    ## 276         276        1      1
    ## 277         277        0      3
    ## 278         278        0      2
    ## 279         279        0      3
    ## 280         280        1      3
    ## 281         281        0      3
    ## 282         282        0      3
    ## 283         283        0      3
    ## 284         284        1      3
    ## 285         285        0      1
    ## 286         286        0      3
    ## 287         287        1      3
    ## 288         288        0      3
    ## 289         289        1      2
    ## 290         290        1      3
    ## 291         291        1      1
    ## 292         292        1      1
    ## 293         293        0      2
    ## 294         294        0      3
    ## 295         295        0      3
    ## 296         296        0      1
    ## 297         297        0      3
    ## 298         298        0      1
    ## 299         299        1      1
    ## 300         300        1      1
    ## 301         301        1      3
    ## 302         302        1      3
    ## 303         303        0      3
    ## 304         304        1      2
    ## 305         305        0      3
    ## 306         306        1      1
    ## 307         307        1      1
    ## 308         308        1      1
    ## 309         309        0      2
    ## 310         310        1      1
    ## 311         311        1      1
    ## 312         312        1      1
    ## 313         313        0      2
    ## 314         314        0      3
    ## 315         315        0      2
    ## 316         316        1      3
    ## 317         317        1      2
    ## 318         318        0      2
    ## 319         319        1      1
    ## 320         320        1      1
    ## 321         321        0      3
    ## 322         322        0      3
    ## 323         323        1      2
    ## 324         324        1      2
    ## 325         325        0      3
    ## 326         326        1      1
    ## 327         327        0      3
    ## 328         328        1      2
    ## 329         329        1      3
    ## 330         330        1      1
    ## 331         331        1      3
    ## 332         332        0      1
    ## 333         333        0      1
    ## 334         334        0      3
    ## 335         335        1      1
    ## 336         336        0      3
    ## 337         337        0      1
    ## 338         338        1      1
    ## 339         339        1      3
    ## 340         340        0      1
    ## 341         341        1      2
    ## 342         342        1      1
    ## 343         343        0      2
    ## 344         344        0      2
    ## 345         345        0      2
    ## 346         346        1      2
    ## 347         347        1      2
    ## 348         348        1      3
    ## 349         349        1      3
    ## 350         350        0      3
    ## 351         351        0      3
    ## 352         352        0      1
    ## 353         353        0      3
    ## 354         354        0      3
    ## 355         355        0      3
    ## 356         356        0      3
    ## 357         357        1      1
    ## 358         358        0      2
    ## 359         359        1      3
    ## 360         360        1      3
    ## 361         361        0      3
    ## 362         362        0      2
    ## 363         363        0      3
    ## 364         364        0      3
    ## 365         365        0      3
    ## 366         366        0      3
    ## 367         367        1      1
    ## 368         368        1      3
    ## 369         369        1      3
    ## 370         370        1      1
    ## 371         371        1      1
    ## 372         372        0      3
    ## 373         373        0      3
    ## 374         374        0      1
    ## 375         375        0      3
    ## 376         376        1      1
    ## 377         377        1      3
    ## 378         378        0      1
    ## 379         379        0      3
    ## 380         380        0      3
    ## 381         381        1      1
    ## 382         382        1      3
    ## 383         383        0      3
    ## 384         384        1      1
    ## 385         385        0      3
    ## 386         386        0      2
    ## 387         387        0      3
    ## 388         388        1      2
    ## 389         389        0      3
    ## 390         390        1      2
    ## 391         391        1      1
    ## 392         392        1      3
    ## 393         393        0      3
    ## 394         394        1      1
    ## 395         395        1      3
    ## 396         396        0      3
    ## 397         397        0      3
    ## 398         398        0      2
    ## 399         399        0      2
    ## 400         400        1      2
    ## 401         401        1      3
    ## 402         402        0      3
    ## 403         403        0      3
    ## 404         404        0      3
    ## 405         405        0      3
    ## 406         406        0      2
    ## 407         407        0      3
    ## 408         408        1      2
    ## 409         409        0      3
    ## 410         410        0      3
    ## 411         411        0      3
    ## 412         412        0      3
    ## 413         413        1      1
    ## 414         414        0      2
    ## 415         415        1      3
    ## 416         416        0      3
    ## 417         417        1      2
    ## 418         418        1      2
    ## 419         419        0      2
    ## 420         420        0      3
    ## 421         421        0      3
    ## 422         422        0      3
    ## 423         423        0      3
    ## 424         424        0      3
    ## 425         425        0      3
    ## 426         426        0      3
    ## 427         427        1      2
    ## 428         428        1      2
    ## 429         429        0      3
    ## 430         430        1      3
    ## 431         431        1      1
    ## 432         432        1      3
    ## 433         433        1      2
    ## 434         434        0      3
    ## 435         435        0      1
    ## 436         436        1      1
    ## 437         437        0      3
    ## 438         438        1      2
    ## 439         439        0      1
    ## 440         440        0      2
    ## 441         441        1      2
    ## 442         442        0      3
    ## 443         443        0      3
    ## 444         444        1      2
    ## 445         445        1      3
    ## 446         446        1      1
    ## 447         447        1      2
    ## 448         448        1      1
    ## 449         449        1      3
    ## 450         450        1      1
    ## 451         451        0      2
    ## 452         452        0      3
    ## 453         453        0      1
    ## 454         454        1      1
    ## 455         455        0      3
    ## 456         456        1      3
    ## 457         457        0      1
    ## 458         458        1      1
    ## 459         459        1      2
    ## 460         460        0      3
    ## 461         461        1      1
    ## 462         462        0      3
    ## 463         463        0      1
    ## 464         464        0      2
    ## 465         465        0      3
    ## 466         466        0      3
    ## 467         467        0      2
    ## 468         468        0      1
    ## 469         469        0      3
    ## 470         470        1      3
    ## 471         471        0      3
    ## 472         472        0      3
    ## 473         473        1      2
    ## 474         474        1      2
    ## 475         475        0      3
    ## 476         476        0      1
    ## 477         477        0      2
    ## 478         478        0      3
    ## 479         479        0      3
    ## 480         480        1      3
    ## 481         481        0      3
    ## 482         482        0      2
    ## 483         483        0      3
    ## 484         484        1      3
    ## 485         485        1      1
    ## 486         486        0      3
    ## 487         487        1      1
    ## 488         488        0      1
    ## 489         489        0      3
    ## 490         490        1      3
    ## 491         491        0      3
    ## 492         492        0      3
    ## 493         493        0      1
    ## 494         494        0      1
    ## 495         495        0      3
    ## 496         496        0      3
    ## 497         497        1      1
    ## 498         498        0      3
    ## 499         499        0      1
    ## 500         500        0      3
    ## 501         501        0      3
    ## 502         502        0      3
    ## 503         503        0      3
    ## 504         504        0      3
    ## 505         505        1      1
    ## 506         506        0      1
    ## 507         507        1      2
    ## 508         508        1      1
    ## 509         509        0      3
    ## 510         510        1      3
    ## 511         511        1      3
    ## 512         512        0      3
    ## 513         513        1      1
    ## 514         514        1      1
    ## 515         515        0      3
    ## 516         516        0      1
    ## 517         517        1      2
    ## 518         518        0      3
    ## 519         519        1      2
    ## 520         520        0      3
    ## 521         521        1      1
    ## 522         522        0      3
    ## 523         523        0      3
    ## 524         524        1      1
    ## 525         525        0      3
    ## 526         526        0      3
    ## 527         527        1      2
    ## 528         528        0      1
    ## 529         529        0      3
    ## 530         530        0      2
    ## 531         531        1      2
    ## 532         532        0      3
    ## 533         533        0      3
    ## 534         534        1      3
    ## 535         535        0      3
    ## 536         536        1      2
    ## 537         537        0      1
    ## 538         538        1      1
    ## 539         539        0      3
    ## 540         540        1      1
    ## 541         541        1      1
    ## 542         542        0      3
    ## 543         543        0      3
    ## 544         544        1      2
    ## 545         545        0      1
    ## 546         546        0      1
    ## 547         547        1      2
    ## 548         548        1      2
    ## 549         549        0      3
    ## 550         550        1      2
    ## 551         551        1      1
    ## 552         552        0      2
    ## 553         553        0      3
    ## 554         554        1      3
    ## 555         555        1      3
    ## 556         556        0      1
    ## 557         557        1      1
    ## 558         558        0      1
    ## 559         559        1      1
    ## 560         560        1      3
    ## 561         561        0      3
    ## 562         562        0      3
    ## 563         563        0      2
    ## 564         564        0      3
    ## 565         565        0      3
    ## 566         566        0      3
    ## 567         567        0      3
    ## 568         568        0      3
    ## 569         569        0      3
    ## 570         570        1      3
    ## 571         571        1      2
    ## 572         572        1      1
    ## 573         573        1      1
    ## 574         574        1      3
    ## 575         575        0      3
    ## 576         576        0      3
    ## 577         577        1      2
    ## 578         578        1      1
    ## 579         579        0      3
    ## 580         580        1      3
    ## 581         581        1      2
    ## 582         582        1      1
    ## 583         583        0      2
    ## 584         584        0      1
    ## 585         585        0      3
    ## 586         586        1      1
    ## 587         587        0      2
    ## 588         588        1      1
    ## 589         589        0      3
    ## 590         590        0      3
    ## 591         591        0      3
    ## 592         592        1      1
    ## 593         593        0      3
    ## 594         594        0      3
    ## 595         595        0      2
    ## 596         596        0      3
    ## 597         597        1      2
    ## 598         598        0      3
    ## 599         599        0      3
    ## 600         600        1      1
    ## 601         601        1      2
    ## 602         602        0      3
    ## 603         603        0      1
    ## 604         604        0      3
    ## 605         605        1      1
    ## 606         606        0      3
    ## 607         607        0      3
    ## 608         608        1      1
    ## 609         609        1      2
    ## 610         610        1      1
    ## 611         611        0      3
    ## 612         612        0      3
    ## 613         613        1      3
    ## 614         614        0      3
    ## 615         615        0      3
    ## 616         616        1      2
    ## 617         617        0      3
    ## 618         618        0      3
    ## 619         619        1      2
    ## 620         620        0      2
    ## 621         621        0      3
    ## 622         622        1      1
    ## 623         623        1      3
    ## 624         624        0      3
    ## 625         625        0      3
    ## 626         626        0      1
    ## 627         627        0      2
    ## 628         628        1      1
    ## 629         629        0      3
    ## 630         630        0      3
    ## 631         631        1      1
    ## 632         632        0      3
    ## 633         633        1      1
    ## 634         634        0      1
    ## 635         635        0      3
    ## 636         636        1      2
    ## 637         637        0      3
    ## 638         638        0      2
    ## 639         639        0      3
    ## 640         640        0      3
    ## 641         641        0      3
    ## 642         642        1      1
    ## 643         643        0      3
    ## 644         644        1      3
    ## 645         645        1      3
    ## 646         646        1      1
    ## 647         647        0      3
    ## 648         648        1      1
    ## 649         649        0      3
    ## 650         650        1      3
    ## 651         651        0      3
    ## 652         652        1      2
    ## 653         653        0      3
    ## 654         654        1      3
    ## 655         655        0      3
    ## 656         656        0      2
    ## 657         657        0      3
    ## 658         658        0      3
    ## 659         659        0      2
    ## 660         660        0      1
    ## 661         661        1      1
    ## 662         662        0      3
    ## 663         663        0      1
    ## 664         664        0      3
    ## 665         665        1      3
    ## 666         666        0      2
    ## 667         667        0      2
    ## 668         668        0      3
    ## 669         669        0      3
    ## 670         670        1      1
    ## 671         671        1      2
    ## 672         672        0      1
    ## 673         673        0      2
    ## 674         674        1      2
    ## 675         675        0      2
    ## 676         676        0      3
    ## 677         677        0      3
    ## 678         678        1      3
    ## 679         679        0      3
    ## 680         680        1      1
    ## 681         681        0      3
    ## 682         682        1      1
    ## 683         683        0      3
    ## 684         684        0      3
    ## 685         685        0      2
    ## 686         686        0      2
    ## 687         687        0      3
    ## 688         688        0      3
    ## 689         689        0      3
    ## 690         690        1      1
    ## 691         691        1      1
    ## 692         692        1      3
    ## 693         693        1      3
    ## 694         694        0      3
    ## 695         695        0      1
    ## 696         696        0      2
    ## 697         697        0      3
    ## 698         698        1      3
    ## 699         699        0      1
    ## 700         700        0      3
    ## 701         701        1      1
    ## 702         702        1      1
    ## 703         703        0      3
    ## 704         704        0      3
    ## 705         705        0      3
    ## 706         706        0      2
    ## 707         707        1      2
    ## 708         708        1      1
    ## 709         709        1      1
    ## 710         710        1      3
    ## 711         711        1      1
    ## 712         712        0      1
    ## 713         713        1      1
    ## 714         714        0      3
    ## 715         715        0      2
    ## 716         716        0      3
    ## 717         717        1      1
    ## 718         718        1      2
    ## 719         719        0      3
    ## 720         720        0      3
    ## 721         721        1      2
    ## 722         722        0      3
    ## 723         723        0      2
    ## 724         724        0      2
    ## 725         725        1      1
    ## 726         726        0      3
    ## 727         727        1      2
    ## 728         728        1      3
    ## 729         729        0      2
    ## 730         730        0      3
    ## 731         731        1      1
    ## 732         732        0      3
    ## 733         733        0      2
    ## 734         734        0      2
    ## 735         735        0      2
    ## 736         736        0      3
    ## 737         737        0      3
    ## 738         738        1      1
    ## 739         739        0      3
    ## 740         740        0      3
    ## 741         741        1      1
    ## 742         742        0      1
    ## 743         743        1      1
    ## 744         744        0      3
    ## 745         745        1      3
    ## 746         746        0      1
    ## 747         747        0      3
    ## 748         748        1      2
    ## 749         749        0      1
    ## 750         750        0      3
    ## 751         751        1      2
    ## 752         752        1      3
    ## 753         753        0      3
    ## 754         754        0      3
    ## 755         755        1      2
    ## 756         756        1      2
    ## 757         757        0      3
    ## 758         758        0      2
    ## 759         759        0      3
    ## 760         760        1      1
    ## 761         761        0      3
    ## 762         762        0      3
    ## 763         763        1      3
    ## 764         764        1      1
    ## 765         765        0      3
    ## 766         766        1      1
    ## 767         767        0      1
    ## 768         768        0      3
    ## 769         769        0      3
    ## 770         770        0      3
    ## 771         771        0      3
    ## 772         772        0      3
    ## 773         773        0      2
    ## 774         774        0      3
    ## 775         775        1      2
    ## 776         776        0      3
    ## 777         777        0      3
    ## 778         778        1      3
    ## 779         779        0      3
    ## 780         780        1      1
    ## 781         781        1      3
    ## 782         782        1      1
    ## 783         783        0      1
    ## 784         784        0      3
    ## 785         785        0      3
    ## 786         786        0      3
    ## 787         787        1      3
    ## 788         788        0      3
    ## 789         789        1      3
    ## 790         790        0      1
    ## 791         791        0      3
    ## 792         792        0      2
    ## 793         793        0      3
    ## 794         794        0      1
    ## 795         795        0      3
    ## 796         796        0      2
    ## 797         797        1      1
    ## 798         798        1      3
    ## 799         799        0      3
    ## 800         800        0      3
    ## 801         801        0      2
    ## 802         802        1      2
    ## 803         803        1      1
    ## 804         804        1      3
    ## 805         805        1      3
    ## 806         806        0      3
    ## 807         807        0      1
    ## 808         808        0      3
    ## 809         809        0      2
    ## 810         810        1      1
    ## 811         811        0      3
    ## 812         812        0      3
    ## 813         813        0      2
    ## 814         814        0      3
    ## 815         815        0      3
    ## 816         816        0      1
    ## 817         817        0      3
    ## 818         818        0      2
    ## 819         819        0      3
    ## 820         820        0      3
    ## 821         821        1      1
    ## 822         822        1      3
    ## 823         823        0      1
    ## 824         824        1      3
    ## 825         825        0      3
    ## 826         826        0      3
    ## 827         827        0      3
    ## 828         828        1      2
    ## 829         829        1      3
    ## 830         830        1      1
    ## 831         831        1      3
    ## 832         832        1      2
    ## 833         833        0      3
    ## 834         834        0      3
    ## 835         835        0      3
    ## 836         836        1      1
    ## 837         837        0      3
    ## 838         838        0      3
    ## 839         839        1      3
    ## 840         840        1      1
    ## 841         841        0      3
    ## 842         842        0      2
    ## 843         843        1      1
    ## 844         844        0      3
    ## 845         845        0      3
    ## 846         846        0      3
    ## 847         847        0      3
    ## 848         848        0      3
    ## 849         849        0      2
    ## 850         850        1      1
    ## 851         851        0      3
    ## 852         852        0      3
    ## 853         853        0      3
    ## 854         854        1      1
    ## 855         855        0      2
    ## 856         856        1      3
    ## 857         857        1      1
    ## 858         858        1      1
    ## 859         859        1      3
    ## 860         860        0      3
    ## 861         861        0      3
    ## 862         862        0      2
    ## 863         863        1      1
    ## 864         864        0      3
    ## 865         865        0      2
    ## 866         866        1      2
    ## 867         867        1      2
    ## 868         868        0      1
    ## 869         869        0      3
    ## 870         870        1      3
    ## 871         871        0      3
    ## 872         872        1      1
    ## 873         873        0      1
    ## 874         874        0      3
    ## 875         875        1      2
    ## 876         876        1      3
    ## 877         877        0      3
    ## 878         878        0      3
    ## 879         879        0      3
    ## 880         880        1      1
    ## 881         881        1      2
    ## 882         882        0      3
    ## 883         883        0      3
    ## 884         884        0      2
    ## 885         885        0      3
    ## 886         886        0      3
    ## 887         887        0      2
    ## 888         888        1      1
    ## 889         889        0      3
    ## 890         890        1      1
    ## 891         891        0      3
    ##                                                                                   Name
    ## 1                                                              Braund, Mr. Owen Harris
    ## 2                                  Cumings, Mrs. John Bradley (Florence Briggs Thayer)
    ## 3                                                               Heikkinen, Miss. Laina
    ## 4                                         Futrelle, Mrs. Jacques Heath (Lily May Peel)
    ## 5                                                             Allen, Mr. William Henry
    ## 6                                                                     Moran, Mr. James
    ## 7                                                              McCarthy, Mr. Timothy J
    ## 8                                                       Palsson, Master. Gosta Leonard
    ## 9                                    Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)
    ## 10                                                 Nasser, Mrs. Nicholas (Adele Achem)
    ## 11                                                     Sandstrom, Miss. Marguerite Rut
    ## 12                                                            Bonnell, Miss. Elizabeth
    ## 13                                                      Saundercock, Mr. William Henry
    ## 14                                                         Andersson, Mr. Anders Johan
    ## 15                                                Vestrom, Miss. Hulda Amanda Adolfina
    ## 16                                                     Hewlett, Mrs. (Mary D Kingcome)
    ## 17                                                                Rice, Master. Eugene
    ## 18                                                        Williams, Mr. Charles Eugene
    ## 19                             Vander Planke, Mrs. Julius (Emelia Maria Vandemoortele)
    ## 20                                                             Masselmani, Mrs. Fatima
    ## 21                                                                Fynney, Mr. Joseph J
    ## 22                                                               Beesley, Mr. Lawrence
    ## 23                                                         McGowan, Miss. Anna "Annie"
    ## 24                                                        Sloper, Mr. William Thompson
    ## 25                                                       Palsson, Miss. Torborg Danira
    ## 26                           Asplund, Mrs. Carl Oscar (Selma Augusta Emilia Johansson)
    ## 27                                                             Emir, Mr. Farred Chehab
    ## 28                                                      Fortune, Mr. Charles Alexander
    ## 29                                                       O'Dwyer, Miss. Ellen "Nellie"
    ## 30                                                                 Todoroff, Mr. Lalio
    ## 31                                                            Uruchurtu, Don. Manuel E
    ## 32                                      Spencer, Mrs. William Augustus (Marie Eugenie)
    ## 33                                                            Glynn, Miss. Mary Agatha
    ## 34                                                               Wheadon, Mr. Edward H
    ## 35                                                             Meyer, Mr. Edgar Joseph
    ## 36                                                      Holverson, Mr. Alexander Oskar
    ## 37                                                                    Mamee, Mr. Hanna
    ## 38                                                            Cann, Mr. Ernest Charles
    ## 39                                                  Vander Planke, Miss. Augusta Maria
    ## 40                                                         Nicola-Yarred, Miss. Jamila
    ## 41                                      Ahlin, Mrs. Johan (Johanna Persdotter Larsson)
    ## 42                            Turpin, Mrs. William John Robert (Dorothy Ann Wonnacott)
    ## 43                                                                 Kraeff, Mr. Theodor
    ## 44                                            Laroche, Miss. Simonne Marie Anne Andree
    ## 45                                                       Devaney, Miss. Margaret Delia
    ## 46                                                            Rogers, Mr. William John
    ## 47                                                                   Lennon, Mr. Denis
    ## 48                                                           O'Driscoll, Miss. Bridget
    ## 49                                                                 Samaan, Mr. Youssef
    ## 50                                       Arnold-Franchi, Mrs. Josef (Josefine Franchi)
    ## 51                                                          Panula, Master. Juha Niilo
    ## 52                                                        Nosworthy, Mr. Richard Cater
    ## 53                                            Harper, Mrs. Henry Sleeper (Myna Haxtun)
    ## 54                                  Faunthorpe, Mrs. Lizzie (Elizabeth Anne Wilkinson)
    ## 55                                                      Ostby, Mr. Engelhart Cornelius
    ## 56                                                                   Woolner, Mr. Hugh
    ## 57                                                                   Rugg, Miss. Emily
    ## 58                                                                 Novel, Mr. Mansouer
    ## 59                                                        West, Miss. Constance Mirium
    ## 60                                                  Goodwin, Master. William Frederick
    ## 61                                                               Sirayanian, Mr. Orsen
    ## 62                                                                 Icard, Miss. Amelie
    ## 63                                                         Harris, Mr. Henry Birkhardt
    ## 64                                                               Skoog, Master. Harald
    ## 65                                                               Stewart, Mr. Albert A
    ## 66                                                            Moubarek, Master. Gerios
    ## 67                                                        Nye, Mrs. (Elizabeth Ramell)
    ## 68                                                            Crease, Mr. Ernest James
    ## 69                                                     Andersson, Miss. Erna Alexandra
    ## 70                                                                   Kink, Mr. Vincenz
    ## 71                                                          Jenkin, Mr. Stephen Curnow
    ## 72                                                          Goodwin, Miss. Lillian Amy
    ## 73                                                                Hood, Mr. Ambrose Jr
    ## 74                                                         Chronopoulos, Mr. Apostolos
    ## 75                                                                       Bing, Mr. Lee
    ## 76                                                             Moen, Mr. Sigurd Hansen
    ## 77                                                                   Staneff, Mr. Ivan
    ## 78                                                            Moutal, Mr. Rahamin Haim
    ## 79                                                       Caldwell, Master. Alden Gates
    ## 80                                                            Dowdell, Miss. Elizabeth
    ## 81                                                                Waelens, Mr. Achille
    ## 82                                                         Sheerlinck, Mr. Jan Baptist
    ## 83                                                      McDermott, Miss. Brigdet Delia
    ## 84                                                             Carrau, Mr. Francisco M
    ## 85                                                                 Ilett, Miss. Bertha
    ## 86                             Backstrom, Mrs. Karl Alfred (Maria Mathilda Gustafsson)
    ## 87                                                              Ford, Mr. William Neal
    ## 88                                                       Slocovski, Mr. Selman Francis
    ## 89                                                          Fortune, Miss. Mabel Helen
    ## 90                                                              Celotti, Mr. Francesco
    ## 91                                                                Christmann, Mr. Emil
    ## 92                                                          Andreasson, Mr. Paul Edvin
    ## 93                                                         Chaffee, Mr. Herbert Fuller
    ## 94                                                             Dean, Mr. Bertram Frank
    ## 95                                                                   Coxon, Mr. Daniel
    ## 96                                                         Shorney, Mr. Charles Joseph
    ## 97                                                           Goldschmidt, Mr. George B
    ## 98                                                     Greenfield, Mr. William Bertram
    ## 99                                                Doling, Mrs. John T (Ada Julia Bone)
    ## 100                                                                  Kantor, Mr. Sinai
    ## 101                                                            Petranec, Miss. Matilda
    ## 102                                                   Petroff, Mr. Pastcho ("Pentcho")
    ## 103                                                          White, Mr. Richard Frasar
    ## 104                                                         Johansson, Mr. Gustaf Joel
    ## 105                                                     Gustafsson, Mr. Anders Vilhelm
    ## 106                                                              Mionoff, Mr. Stoytcho
    ## 107                                                   Salkjelsvik, Miss. Anna Kristine
    ## 108                                                             Moss, Mr. Albert Johan
    ## 109                                                                    Rekic, Mr. Tido
    ## 110                                                                Moran, Miss. Bertha
    ## 111                                                     Porter, Mr. Walter Chamberlain
    ## 112                                                               Zabour, Miss. Hileni
    ## 113                                                             Barton, Mr. David John
    ## 114                                                            Jussila, Miss. Katriina
    ## 115                                                              Attalah, Miss. Malake
    ## 116                                                              Pekoniemi, Mr. Edvard
    ## 117                                                               Connors, Mr. Patrick
    ## 118                                                    Turpin, Mr. William John Robert
    ## 119                                                           Baxter, Mr. Quigg Edmond
    ## 120                                                  Andersson, Miss. Ellis Anna Maria
    ## 121                                                        Hickman, Mr. Stanley George
    ## 122                                                         Moore, Mr. Leonard Charles
    ## 123                                                               Nasser, Mr. Nicholas
    ## 124                                                                Webber, Miss. Susan
    ## 125                                                        White, Mr. Percival Wayland
    ## 126                                                       Nicola-Yarred, Master. Elias
    ## 127                                                                McMahon, Mr. Martin
    ## 128                                                          Madsen, Mr. Fridtjof Arne
    ## 129                                                                  Peter, Miss. Anna
    ## 130                                                                 Ekstrom, Mr. Johan
    ## 131                                                               Drazenoic, Mr. Jozef
    ## 132                                                     Coelho, Mr. Domingos Fernandeo
    ## 133                                     Robins, Mrs. Alexander A (Grace Charity Laury)
    ## 134                                      Weisz, Mrs. Leopold (Mathilde Francoise Pede)
    ## 135                                                     Sobey, Mr. Samuel James Hayden
    ## 136                                                                 Richard, Mr. Emile
    ## 137                                                       Newsom, Miss. Helen Monypeny
    ## 138                                                        Futrelle, Mr. Jacques Heath
    ## 139                                                                Osen, Mr. Olaf Elon
    ## 140                                                                 Giglio, Mr. Victor
    ## 141                                                      Boulos, Mrs. Joseph (Sultana)
    ## 142                                                           Nysten, Miss. Anna Sofia
    ## 143                               Hakkarainen, Mrs. Pekka Pietari (Elin Matilda Dolck)
    ## 144                                                                Burke, Mr. Jeremiah
    ## 145                                                         Andrew, Mr. Edgardo Samuel
    ## 146                                                       Nicholls, Mr. Joseph Charles
    ## 147                                       Andersson, Mr. August Edvard ("Wennerstrom")
    ## 148                                                   Ford, Miss. Robina Maggie "Ruby"
    ## 149                                           Navratil, Mr. Michel ("Louis M Hoffman")
    ## 150                                                  Byles, Rev. Thomas Roussel Davids
    ## 151                                                         Bateman, Rev. Robert James
    ## 152                                                  Pears, Mrs. Thomas (Edith Wearne)
    ## 153                                                                   Meo, Mr. Alfonzo
    ## 154                                                    van Billiard, Mr. Austin Blyler
    ## 155                                                              Olsen, Mr. Ole Martin
    ## 156                                                        Williams, Mr. Charles Duane
    ## 157                                                   Gilnagh, Miss. Katherine "Katie"
    ## 158                                                                    Corn, Mr. Harry
    ## 159                                                                Smiljanic, Mr. Mile
    ## 160                                                         Sage, Master. Thomas Henry
    ## 161                                                           Cribb, Mr. John Hatfield
    ## 162                                 Watt, Mrs. James (Elizabeth "Bessie" Inglis Milne)
    ## 163                                                         Bengtsson, Mr. John Viktor
    ## 164                                                                    Calic, Mr. Jovo
    ## 165                                                       Panula, Master. Eino Viljami
    ## 166                                    Goldsmith, Master. Frank John William "Frankie"
    ## 167                                             Chibnall, Mrs. (Edith Martha Bowerman)
    ## 168                                    Skoog, Mrs. William (Anna Bernhardina Karlsson)
    ## 169                                                                Baumann, Mr. John D
    ## 170                                                                      Ling, Mr. Lee
    ## 171                                                          Van der hoef, Mr. Wyckoff
    ## 172                                                               Rice, Master. Arthur
    ## 173                                                       Johnson, Miss. Eleanor Ileen
    ## 174                                                          Sivola, Mr. Antti Wilhelm
    ## 175                                                            Smith, Mr. James Clinch
    ## 176                                                             Klasen, Mr. Klas Albin
    ## 177                                                      Lefebre, Master. Henry Forbes
    ## 178                                                         Isham, Miss. Ann Elizabeth
    ## 179                                                                 Hale, Mr. Reginald
    ## 180                                                                Leonard, Mr. Lionel
    ## 181                                                       Sage, Miss. Constance Gladys
    ## 182                                                                   Pernot, Mr. Rene
    ## 183                                              Asplund, Master. Clarence Gustaf Hugo
    ## 184                                                          Becker, Master. Richard F
    ## 185                                                Kink-Heilmann, Miss. Luise Gretchen
    ## 186                                                              Rood, Mr. Hugh Roscoe
    ## 187                                    O'Brien, Mrs. Thomas (Johanna "Hannah" Godfrey)
    ## 188                                      Romaine, Mr. Charles Hallace ("Mr C Rolmane")
    ## 189                                                                   Bourke, Mr. John
    ## 190                                                                Turcin, Mr. Stjepan
    ## 191                                                                Pinsky, Mrs. (Rosa)
    ## 192                                                              Carbines, Mr. William
    ## 193                                    Andersen-Jensen, Miss. Carla Christine Nielsine
    ## 194                                                         Navratil, Master. Michel M
    ## 195                                          Brown, Mrs. James Joseph (Margaret Tobin)
    ## 196                                                               Lurette, Miss. Elise
    ## 197                                                                Mernagh, Mr. Robert
    ## 198                                                   Olsen, Mr. Karl Siegwart Andreas
    ## 199                                                   Madigan, Miss. Margaret "Maggie"
    ## 200                                             Yrois, Miss. Henriette ("Mrs Harbeck")
    ## 201                                                     Vande Walle, Mr. Nestor Cyriel
    ## 202                                                                Sage, Mr. Frederick
    ## 203                                                         Johanson, Mr. Jakob Alfred
    ## 204                                                               Youseff, Mr. Gerious
    ## 205                                                           Cohen, Mr. Gurshon "Gus"
    ## 206                                                         Strom, Miss. Telma Matilda
    ## 207                                                         Backstrom, Mr. Karl Alfred
    ## 208                                                        Albimona, Mr. Nassef Cassem
    ## 209                                                          Carr, Miss. Helen "Ellen"
    ## 210                                                                   Blank, Mr. Henry
    ## 211                                                                     Ali, Mr. Ahmed
    ## 212                                                         Cameron, Miss. Clear Annie
    ## 213                                                             Perkin, Mr. John Henry
    ## 214                                                        Givard, Mr. Hans Kristensen
    ## 215                                                                Kiernan, Mr. Philip
    ## 216                                                            Newell, Miss. Madeleine
    ## 217                                                             Honkanen, Miss. Eliina
    ## 218                                                       Jacobsohn, Mr. Sidney Samuel
    ## 219                                                              Bazzani, Miss. Albina
    ## 220                                                                 Harris, Mr. Walter
    ## 221                                                     Sunderland, Mr. Victor Francis
    ## 222                                                               Bracken, Mr. James H
    ## 223                                                            Green, Mr. George Henry
    ## 224                                                               Nenkoff, Mr. Christo
    ## 225                                                       Hoyt, Mr. Frederick Maxfield
    ## 226                                                       Berglund, Mr. Karl Ivar Sven
    ## 227                                                          Mellors, Mr. William John
    ## 228                                                    Lovell, Mr. John Hall ("Henry")
    ## 229                                                          Fahlstrom, Mr. Arne Jonas
    ## 230                                                            Lefebre, Miss. Mathilde
    ## 231                                       Harris, Mrs. Henry Birkhardt (Irene Wallach)
    ## 232                                                           Larsson, Mr. Bengt Edvin
    ## 233                                                          Sjostedt, Mr. Ernst Adolf
    ## 234                                                     Asplund, Miss. Lillian Gertrud
    ## 235                                                  Leyson, Mr. Robert William Norman
    ## 236                                                       Harknett, Miss. Alice Phoebe
    ## 237                                                                  Hold, Mr. Stephen
    ## 238                                                   Collyer, Miss. Marjorie "Lottie"
    ## 239                                                    Pengelly, Mr. Frederick William
    ## 240                                                             Hunt, Mr. George Henry
    ## 241                                                              Zabour, Miss. Thamine
    ## 242                                                     Murphy, Miss. Katherine "Kate"
    ## 243                                                    Coleridge, Mr. Reginald Charles
    ## 244                                                      Maenpaa, Mr. Matti Alexanteri
    ## 245                                                               Attalah, Mr. Sleiman
    ## 246                                                        Minahan, Dr. William Edward
    ## 247                                              Lindahl, Miss. Agda Thorilda Viktoria
    ## 248                                                    Hamalainen, Mrs. William (Anna)
    ## 249                                                      Beckwith, Mr. Richard Leonard
    ## 250                                                      Carter, Rev. Ernest Courtenay
    ## 251                                                             Reed, Mr. James George
    ## 252                                         Strom, Mrs. Wilhelm (Elna Matilda Persson)
    ## 253                                                          Stead, Mr. William Thomas
    ## 254                                                           Lobb, Mr. William Arthur
    ## 255                                           Rosblom, Mrs. Viktor (Helena Wilhelmina)
    ## 256                                            Touma, Mrs. Darwis (Hanne Youssef Razi)
    ## 257                                                     Thorne, Mrs. Gertrude Maybelle
    ## 258                                                               Cherry, Miss. Gladys
    ## 259                                                                   Ward, Miss. Anna
    ## 260                                                        Parrish, Mrs. (Lutie Davis)
    ## 261                                                                  Smith, Mr. Thomas
    ## 262                                                  Asplund, Master. Edvin Rojj Felix
    ## 263                                                                  Taussig, Mr. Emil
    ## 264                                                              Harrison, Mr. William
    ## 265                                                                 Henry, Miss. Delia
    ## 266                                                                  Reeves, Mr. David
    ## 267                                                          Panula, Mr. Ernesti Arvid
    ## 268                                                           Persson, Mr. Ernst Ulrik
    ## 269                                      Graham, Mrs. William Thompson (Edith Junkins)
    ## 270                                                             Bissette, Miss. Amelia
    ## 271                                                              Cairns, Mr. Alexander
    ## 272                                                       Tornquist, Mr. William Henry
    ## 273                                          Mellinger, Mrs. (Elizabeth Anne Maidment)
    ## 274                                                              Natsch, Mr. Charles H
    ## 275                                                         Healy, Miss. Hanora "Nora"
    ## 276                                                  Andrews, Miss. Kornelia Theodosia
    ## 277                                                  Lindblom, Miss. Augusta Charlotta
    ## 278                                                        Parkes, Mr. Francis "Frank"
    ## 279                                                                 Rice, Master. Eric
    ## 280                                                   Abbott, Mrs. Stanton (Rosa Hunt)
    ## 281                                                                   Duane, Mr. Frank
    ## 282                                                   Olsson, Mr. Nils Johan Goransson
    ## 283                                                          de Pelsmaeker, Mr. Alfons
    ## 284                                                         Dorking, Mr. Edward Arthur
    ## 285                                                         Smith, Mr. Richard William
    ## 286                                                                Stankovic, Mr. Ivan
    ## 287                                                            de Mulder, Mr. Theodore
    ## 288                                                               Naidenoff, Mr. Penko
    ## 289                                                               Hosono, Mr. Masabumi
    ## 290                                                               Connolly, Miss. Kate
    ## 291                                                       Barber, Miss. Ellen "Nellie"
    ## 292                                            Bishop, Mrs. Dickinson H (Helen Walton)
    ## 293                                                             Levy, Mr. Rene Jacques
    ## 294                                                                Haas, Miss. Aloisia
    ## 295                                                                   Mineff, Mr. Ivan
    ## 296                                                                  Lewy, Mr. Ervin G
    ## 297                                                                 Hanna, Mr. Mansour
    ## 298                                                       Allison, Miss. Helen Loraine
    ## 299                                                              Saalfeld, Mr. Adolphe
    ## 300                                    Baxter, Mrs. James (Helene DeLaudeniere Chaput)
    ## 301                                           Kelly, Miss. Anna Katherine "Annie Kate"
    ## 302                                                                 McCoy, Mr. Bernard
    ## 303                                                    Johnson, Mr. William Cahoone Jr
    ## 304                                                                Keane, Miss. Nora A
    ## 305                                                  Williams, Mr. Howard Hugh "Harry"
    ## 306                                                     Allison, Master. Hudson Trevor
    ## 307                                                            Fleming, Miss. Margaret
    ## 308 Penasco y Castellana, Mrs. Victor de Satode (Maria Josefa Perez de Soto y Vallejo)
    ## 309                                                                Abelson, Mr. Samuel
    ## 310                                                     Francatelli, Miss. Laura Mabel
    ## 311                                                     Hays, Miss. Margaret Bechstein
    ## 312                                                         Ryerson, Miss. Emily Borie
    ## 313                                              Lahtinen, Mrs. William (Anna Sylfven)
    ## 314                                                             Hendekovic, Mr. Ignjac
    ## 315                                                                 Hart, Mr. Benjamin
    ## 316                                                    Nilsson, Miss. Helmina Josefina
    ## 317                                                Kantor, Mrs. Sinai (Miriam Sternin)
    ## 318                                                               Moraweck, Dr. Ernest
    ## 319                                                           Wick, Miss. Mary Natalie
    ## 320                           Spedden, Mrs. Frederic Oakley (Margaretta Corning Stone)
    ## 321                                                                 Dennis, Mr. Samuel
    ## 322                                                                   Danoff, Mr. Yoto
    ## 323                                                          Slayter, Miss. Hilda Mary
    ## 324                                Caldwell, Mrs. Albert Francis (Sylvia Mae Harbaugh)
    ## 325                                                           Sage, Mr. George John Jr
    ## 326                                                           Young, Miss. Marie Grice
    ## 327                                                          Nysveen, Mr. Johan Hansen
    ## 328                                                            Ball, Mrs. (Ada E Hall)
    ## 329                                     Goldsmith, Mrs. Frank John (Emily Alice Brown)
    ## 330                                                       Hippach, Miss. Jean Gertrude
    ## 331                                                                 McCoy, Miss. Agnes
    ## 332                                                                Partner, Mr. Austen
    ## 333                                                          Graham, Mr. George Edward
    ## 334                                                    Vander Planke, Mr. Leo Edmondus
    ## 335                                 Frauenthal, Mrs. Henry William (Clara Heinsheimer)
    ## 336                                                                 Denkoff, Mr. Mitto
    ## 337                                                          Pears, Mr. Thomas Clinton
    ## 338                                                    Burns, Miss. Elizabeth Margaret
    ## 339                                                              Dahl, Mr. Karl Edwart
    ## 340                                                       Blackwell, Mr. Stephen Weart
    ## 341                                                     Navratil, Master. Edmond Roger
    ## 342                                                     Fortune, Miss. Alice Elizabeth
    ## 343                                                         Collander, Mr. Erik Gustaf
    ## 344                                         Sedgwick, Mr. Charles Frederick Waddington
    ## 345                                                            Fox, Mr. Stanley Hubert
    ## 346                                                      Brown, Miss. Amelia "Mildred"
    ## 347                                                          Smith, Miss. Marion Elsie
    ## 348                                          Davison, Mrs. Thomas Henry (Mary E Finck)
    ## 349                                             Coutts, Master. William Loch "William"
    ## 350                                                                   Dimic, Mr. Jovan
    ## 351                                                             Odahl, Mr. Nils Martin
    ## 352                                             Williams-Lambert, Mr. Fletcher Fellows
    ## 353                                                                 Elias, Mr. Tannous
    ## 354                                                          Arnold-Franchi, Mr. Josef
    ## 355                                                                  Yousif, Mr. Wazli
    ## 356                                                        Vanden Steen, Mr. Leo Peter
    ## 357                                                        Bowerman, Miss. Elsie Edith
    ## 358                                                          Funk, Miss. Annie Clemmer
    ## 359                                                               McGovern, Miss. Mary
    ## 360                                                  Mockler, Miss. Helen Mary "Ellie"
    ## 361                                                                 Skoog, Mr. Wilhelm
    ## 362                                                          del Carlo, Mr. Sebastiano
    ## 363                                                    Barbara, Mrs. (Catherine David)
    ## 364                                                                    Asim, Mr. Adola
    ## 365                                                                O'Brien, Mr. Thomas
    ## 366                                                     Adahl, Mr. Mauritz Nils Martin
    ## 367                                   Warren, Mrs. Frank Manley (Anna Sophia Atkinson)
    ## 368                                                     Moussa, Mrs. (Mantoura Boulos)
    ## 369                                                                Jermyn, Miss. Annie
    ## 370                                                      Aubart, Mme. Leontine Pauline
    ## 371                                                        Harder, Mr. George Achilles
    ## 372                                                          Wiklund, Mr. Jakob Alfred
    ## 373                                                         Beavan, Mr. William Thomas
    ## 374                                                                Ringhini, Mr. Sante
    ## 375                                                         Palsson, Miss. Stina Viola
    ## 376                                              Meyer, Mrs. Edgar Joseph (Leila Saks)
    ## 377                                                    Landergren, Miss. Aurora Adelia
    ## 378                                                          Widener, Mr. Harry Elkins
    ## 379                                                                Betros, Mr. Tannous
    ## 380                                                        Gustafsson, Mr. Karl Gideon
    ## 381                                                              Bidois, Miss. Rosalie
    ## 382                                                        Nakid, Miss. Maria ("Mary")
    ## 383                                                                 Tikkanen, Mr. Juho
    ## 384                                Holverson, Mrs. Alexander Oskar (Mary Aline Towner)
    ## 385                                                             Plotcharsky, Mr. Vasil
    ## 386                                                          Davies, Mr. Charles Henry
    ## 387                                                    Goodwin, Master. Sidney Leonard
    ## 388                                                                   Buss, Miss. Kate
    ## 389                                                               Sadlier, Mr. Matthew
    ## 390                                                              Lehmann, Miss. Bertha
    ## 391                                                         Carter, Mr. William Ernest
    ## 392                                                             Jansson, Mr. Carl Olof
    ## 393                                                       Gustafsson, Mr. Johan Birger
    ## 394                                                             Newell, Miss. Marjorie
    ## 395                                Sandstrom, Mrs. Hjalmar (Agnes Charlotta Bengtsson)
    ## 396                                                                Johansson, Mr. Erik
    ## 397                                                                Olsson, Miss. Elina
    ## 398                                                            McKane, Mr. Peter David
    ## 399                                                                   Pain, Dr. Alfred
    ## 400                                                   Trout, Mrs. William H (Jessie L)
    ## 401                                                                 Niskanen, Mr. Juha
    ## 402                                                                    Adams, Mr. John
    ## 403                                                           Jussila, Miss. Mari Aina
    ## 404                                                     Hakkarainen, Mr. Pekka Pietari
    ## 405                                                            Oreskovic, Miss. Marija
    ## 406                                                                 Gale, Mr. Shadrach
    ## 407                                                   Widegren, Mr. Carl/Charles Peter
    ## 408                                                     Richards, Master. William Rowe
    ## 409                                                  Birkeland, Mr. Hans Martin Monsen
    ## 410                                                                 Lefebre, Miss. Ida
    ## 411                                                                 Sdycoff, Mr. Todor
    ## 412                                                                    Hart, Mr. Henry
    ## 413                                                             Minahan, Miss. Daisy E
    ## 414                                                     Cunningham, Mr. Alfred Fleming
    ## 415                                                          Sundman, Mr. Johan Julian
    ## 416                                            Meek, Mrs. Thomas (Annie Louise Rowley)
    ## 417                                    Drew, Mrs. James Vivian (Lulu Thorne Christian)
    ## 418                                                      Silven, Miss. Lyyli Karoliina
    ## 419                                                         Matthews, Mr. William John
    ## 420                                                          Van Impe, Miss. Catharina
    ## 421                                                             Gheorgheff, Mr. Stanio
    ## 422                                                                Charters, Mr. David
    ## 423                                                                 Zimmerman, Mr. Leo
    ## 424                             Danbom, Mrs. Ernst Gilbert (Anna Sigrid Maria Brogren)
    ## 425                                                        Rosblom, Mr. Viktor Richard
    ## 426                                                             Wiseman, Mr. Phillippe
    ## 427                                        Clarke, Mrs. Charles V (Ada Maria Winfield)
    ## 428                Phillips, Miss. Kate Florence ("Mrs Kate Louise Phillips Marshall")
    ## 429                                                                   Flynn, Mr. James
    ## 430                                                 Pickard, Mr. Berk (Berk Trembisky)
    ## 431                                          Bjornstrom-Steffansson, Mr. Mauritz Hakan
    ## 432                                  Thorneycroft, Mrs. Percival (Florence Kate White)
    ## 433                                Louch, Mrs. Charles Alexander (Alice Adelaide Slow)
    ## 434                                                         Kallio, Mr. Nikolai Erland
    ## 435                                                          Silvey, Mr. William Baird
    ## 436                                                          Carter, Miss. Lucile Polk
    ## 437                                               Ford, Miss. Doolina Margaret "Daisy"
    ## 438                                              Richards, Mrs. Sidney (Emily Hocking)
    ## 439                                                                  Fortune, Mr. Mark
    ## 440                                             Kvillner, Mr. Johan Henrik Johannesson
    ## 441                                        Hart, Mrs. Benjamin (Esther Ada Bloomfield)
    ## 442                                                                    Hampe, Mr. Leon
    ## 443                                                          Petterson, Mr. Johan Emil
    ## 444                                                          Reynaldo, Ms. Encarnacion
    ## 445                                                  Johannesen-Bratthammer, Mr. Bernt
    ## 446                                                          Dodge, Master. Washington
    ## 447                                                  Mellinger, Miss. Madeleine Violet
    ## 448                                                        Seward, Mr. Frederic Kimber
    ## 449                                                     Baclini, Miss. Marie Catherine
    ## 450                                                     Peuchen, Major. Arthur Godfrey
    ## 451                                                              West, Mr. Edwy Arthur
    ## 452                                                    Hagland, Mr. Ingvald Olai Olsen
    ## 453                                                    Foreman, Mr. Benjamin Laventall
    ## 454                                                           Goldenberg, Mr. Samuel L
    ## 455                                                                Peduzzi, Mr. Joseph
    ## 456                                                                 Jalsevac, Mr. Ivan
    ## 457                                                          Millet, Mr. Francis Davis
    ## 458                                                  Kenyon, Mrs. Frederick R (Marion)
    ## 459                                                                Toomey, Miss. Ellen
    ## 460                                                              O'Connor, Mr. Maurice
    ## 461                                                                Anderson, Mr. Harry
    ## 462                                                                Morley, Mr. William
    ## 463                                                                  Gee, Mr. Arthur H
    ## 464                                                       Milling, Mr. Jacob Christian
    ## 465                                                                 Maisner, Mr. Simon
    ## 466                                                    Goncalves, Mr. Manuel Estanslas
    ## 467                                                              Campbell, Mr. William
    ## 468                                                         Smart, Mr. John Montgomery
    ## 469                                                                 Scanlan, Mr. James
    ## 470                                                      Baclini, Miss. Helene Barbara
    ## 471                                                                  Keefe, Mr. Arthur
    ## 472                                                                    Cacic, Mr. Luka
    ## 473                                            West, Mrs. Edwy Arthur (Ada Mary Worth)
    ## 474                                       Jerwan, Mrs. Amin S (Marie Marthe Thuillard)
    ## 475                                                        Strandberg, Miss. Ida Sofia
    ## 476                                                        Clifford, Mr. George Quincy
    ## 477                                                            Renouf, Mr. Peter Henry
    ## 478                                                          Braund, Mr. Lewis Richard
    ## 479                                                          Karlsson, Mr. Nils August
    ## 480                                                           Hirvonen, Miss. Hildur E
    ## 481                                                     Goodwin, Master. Harold Victor
    ## 482                                                   Frost, Mr. Anthony Wood "Archie"
    ## 483                                                           Rouse, Mr. Richard Henry
    ## 484                                                             Turkula, Mrs. (Hedwig)
    ## 485                                                            Bishop, Mr. Dickinson H
    ## 486                                                             Lefebre, Miss. Jeannie
    ## 487                                    Hoyt, Mrs. Frederick Maxfield (Jane Anne Forby)
    ## 488                                                            Kent, Mr. Edward Austin
    ## 489                                                      Somerton, Mr. Francis William
    ## 490                                              Coutts, Master. Eden Leslie "Neville"
    ## 491                                               Hagland, Mr. Konrad Mathias Reiersen
    ## 492                                                                Windelov, Mr. Einar
    ## 493                                                         Molson, Mr. Harry Markland
    ## 494                                                            Artagaveytia, Mr. Ramon
    ## 495                                                         Stanley, Mr. Edward Roland
    ## 496                                                              Yousseff, Mr. Gerious
    ## 497                                                     Eustis, Miss. Elizabeth Mussey
    ## 498                                                    Shellard, Mr. Frederick William
    ## 499                                    Allison, Mrs. Hudson J C (Bessie Waldo Daniels)
    ## 500                                                                 Svensson, Mr. Olof
    ## 501                                                                   Calic, Mr. Petar
    ## 502                                                                Canavan, Miss. Mary
    ## 503                                                     O'Sullivan, Miss. Bridget Mary
    ## 504                                                     Laitinen, Miss. Kristina Sofia
    ## 505                                                              Maioni, Miss. Roberta
    ## 506                                         Penasco y Castellana, Mr. Victor de Satode
    ## 507                                      Quick, Mrs. Frederick Charles (Jane Richards)
    ## 508                                      Bradley, Mr. George ("George Arthur Brayton")
    ## 509                                                           Olsen, Mr. Henry Margido
    ## 510                                                                     Lang, Mr. Fang
    ## 511                                                           Daly, Mr. Eugene Patrick
    ## 512                                                                  Webber, Mr. James
    ## 513                                                          McGough, Mr. James Robert
    ## 514                                     Rothschild, Mrs. Martin (Elizabeth L. Barrett)
    ## 515                                                                  Coleff, Mr. Satio
    ## 516                                                       Walker, Mr. William Anderson
    ## 517                                                       Lemore, Mrs. (Amelia Milley)
    ## 518                                                                  Ryan, Mr. Patrick
    ## 519                               Angle, Mrs. William A (Florence "Mary" Agnes Hughes)
    ## 520                                                                Pavlovic, Mr. Stefo
    ## 521                                                              Perreault, Miss. Anne
    ## 522                                                                    Vovk, Mr. Janko
    ## 523                                                                 Lahoud, Mr. Sarkis
    ## 524                                    Hippach, Mrs. Louis Albert (Ida Sophia Fischer)
    ## 525                                                                  Kassem, Mr. Fared
    ## 526                                                                 Farrell, Mr. James
    ## 527                                                               Ridsdale, Miss. Lucy
    ## 528                                                                 Farthing, Mr. John
    ## 529                                                          Salonen, Mr. Johan Werner
    ## 530                                                        Hocking, Mr. Richard George
    ## 531                                                           Quick, Miss. Phyllis May
    ## 532                                                                  Toufik, Mr. Nakli
    ## 533                                                               Elias, Mr. Joseph Jr
    ## 534                                             Peter, Mrs. Catherine (Catherine Rizk)
    ## 535                                                                Cacic, Miss. Marija
    ## 536                                                             Hart, Miss. Eva Miriam
    ## 537                                                  Butt, Major. Archibald Willingham
    ## 538                                                                LeRoy, Miss. Bertha
    ## 539                                                           Risien, Mr. Samuel Beard
    ## 540                                                 Frolicher, Miss. Hedwig Margaritha
    ## 541                                                            Crosby, Miss. Harriet R
    ## 542                                               Andersson, Miss. Ingeborg Constanzia
    ## 543                                                  Andersson, Miss. Sigrid Elisabeth
    ## 544                                                                  Beane, Mr. Edward
    ## 545                                                         Douglas, Mr. Walter Donald
    ## 546                                                       Nicholson, Mr. Arthur Ernest
    ## 547                                                  Beane, Mrs. Edward (Ethel Clarke)
    ## 548                                                         Padro y Manent, Mr. Julian
    ## 549                                                          Goldsmith, Mr. Frank John
    ## 550                                                     Davies, Master. John Morgan Jr
    ## 551                                                        Thayer, Mr. John Borland Jr
    ## 552                                                        Sharp, Mr. Percival James R
    ## 553                                                               O'Brien, Mr. Timothy
    ## 554                                                  Leeni, Mr. Fahim ("Philip Zenni")
    ## 555                                                                 Ohman, Miss. Velin
    ## 556                                                                 Wright, Mr. George
    ## 557                  Duff Gordon, Lady. (Lucille Christiana Sutherland) ("Mrs Morgan")
    ## 558                                                                Robbins, Mr. Victor
    ## 559                                             Taussig, Mrs. Emil (Tillie Mandelbaum)
    ## 560                                       de Messemaeker, Mrs. Guillaume Joseph (Emma)
    ## 561                                                           Morrow, Mr. Thomas Rowan
    ## 562                                                                  Sivic, Mr. Husein
    ## 563                                                         Norman, Mr. Robert Douglas
    ## 564                                                                  Simmons, Mr. John
    ## 565                                                     Meanwell, Miss. (Marion Ogden)
    ## 566                                                               Davies, Mr. Alfred J
    ## 567                                                               Stoytcheff, Mr. Ilia
    ## 568                                        Palsson, Mrs. Nils (Alma Cornelia Berglund)
    ## 569                                                                Doharr, Mr. Tannous
    ## 570                                                                  Jonsson, Mr. Carl
    ## 571                                                                 Harris, Mr. George
    ## 572                                      Appleton, Mrs. Edward Dale (Charlotte Lamson)
    ## 573                                                   Flynn, Mr. John Irwin ("Irving")
    ## 574                                                                  Kelly, Miss. Mary
    ## 575                                                       Rush, Mr. Alfred George John
    ## 576                                                               Patchett, Mr. George
    ## 577                                                               Garside, Miss. Ethel
    ## 578                                          Silvey, Mrs. William Baird (Alice Munger)
    ## 579                                                   Caram, Mrs. Joseph (Maria Elias)
    ## 580                                                                Jussila, Mr. Eiriik
    ## 581                                                        Christy, Miss. Julie Rachel
    ## 582                               Thayer, Mrs. John Borland (Marian Longstreth Morris)
    ## 583                                                         Downton, Mr. William James
    ## 584                                                                Ross, Mr. John Hugo
    ## 585                                                                Paulner, Mr. Uscher
    ## 586                                                                Taussig, Miss. Ruth
    ## 587                                                            Jarvis, Mr. John Denzil
    ## 588                                                   Frolicher-Stehli, Mr. Maxmillian
    ## 589                                                              Gilinski, Mr. Eliezer
    ## 590                                                                Murdlin, Mr. Joseph
    ## 591                                                               Rintamaki, Mr. Matti
    ## 592                                    Stephenson, Mrs. Walter Bertram (Martha Eustis)
    ## 593                                                         Elsbury, Mr. William James
    ## 594                                                                 Bourke, Miss. Mary
    ## 595                                                            Chapman, Mr. John Henry
    ## 596                                                        Van Impe, Mr. Jean Baptiste
    ## 597                                                         Leitch, Miss. Jessie Wills
    ## 598                                                                Johnson, Mr. Alfred
    ## 599                                                                  Boulos, Mr. Hanna
    ## 600                                       Duff Gordon, Sir. Cosmo Edmund ("Mr Morgan")
    ## 601                                Jacobsohn, Mrs. Sidney Samuel (Amy Frances Christy)
    ## 602                                                               Slabenoff, Mr. Petco
    ## 603                                                          Harrington, Mr. Charles H
    ## 604                                                          Torber, Mr. Ernst William
    ## 605                                                    Homer, Mr. Harry ("Mr E Haven")
    ## 606                                                      Lindell, Mr. Edvard Bengtsson
    ## 607                                                                  Karaic, Mr. Milan
    ## 608                                                        Daniel, Mr. Robert Williams
    ## 609                              Laroche, Mrs. Joseph (Juliette Marie Louise Lafargue)
    ## 610                                                          Shutes, Miss. Elizabeth W
    ## 611                          Andersson, Mrs. Anders Johan (Alfrida Konstantia Brogren)
    ## 612                                                              Jardin, Mr. Jose Neto
    ## 613                                                        Murphy, Miss. Margaret Jane
    ## 614                                                                   Horgan, Mr. John
    ## 615                                                    Brocklebank, Mr. William Alfred
    ## 616                                                                Herman, Miss. Alice
    ## 617                                                          Danbom, Mr. Ernst Gilbert
    ## 618                                    Lobb, Mrs. William Arthur (Cordelia K Stanlick)
    ## 619                                                        Becker, Miss. Marion Louise
    ## 620                                                                Gavey, Mr. Lawrence
    ## 621                                                                Yasbeck, Mr. Antoni
    ## 622                                                       Kimball, Mr. Edwin Nelson Jr
    ## 623                                                                   Nakid, Mr. Sahid
    ## 624                                                        Hansen, Mr. Henry Damsgaard
    ## 625                                                        Bowen, Mr. David John "Dai"
    ## 626                                                              Sutton, Mr. Frederick
    ## 627                                                     Kirkland, Rev. Charles Leonard
    ## 628                                                      Longley, Miss. Gretchen Fiske
    ## 629                                                          Bostandyeff, Mr. Guentcho
    ## 630                                                           O'Connell, Mr. Patrick D
    ## 631                                               Barkworth, Mr. Algernon Henry Wilson
    ## 632                                                        Lundahl, Mr. Johan Svensson
    ## 633                                                          Stahelin-Maeglin, Dr. Max
    ## 634                                                      Parr, Mr. William Henry Marsh
    ## 635                                                                 Skoog, Miss. Mabel
    ## 636                                                                  Davis, Miss. Mary
    ## 637                                                         Leinonen, Mr. Antti Gustaf
    ## 638                                                                Collyer, Mr. Harvey
    ## 639                                             Panula, Mrs. Juha (Maria Emilia Ojala)
    ## 640                                                         Thorneycroft, Mr. Percival
    ## 641                                                             Jensen, Mr. Hans Peder
    ## 642                                                               Sagesser, Mlle. Emma
    ## 643                                                      Skoog, Miss. Margit Elizabeth
    ## 644                                                                    Foo, Mr. Choong
    ## 645                                                             Baclini, Miss. Eugenie
    ## 646                                                          Harper, Mr. Henry Sleeper
    ## 647                                                                  Cor, Mr. Liudevit
    ## 648                                                Simonius-Blumer, Col. Oberst Alfons
    ## 649                                                                 Willey, Mr. Edward
    ## 650                                                    Stanley, Miss. Amy Zillah Elsie
    ## 651                                                                  Mitkoff, Mr. Mito
    ## 652                                                                Doling, Miss. Elsie
    ## 653                                                     Kalvik, Mr. Johannes Halvorsen
    ## 654                                                      O'Leary, Miss. Hanora "Norah"
    ## 655                                                       Hegarty, Miss. Hanora "Nora"
    ## 656                                                          Hickman, Mr. Leonard Mark
    ## 657                                                              Radeff, Mr. Alexander
    ## 658                                                      Bourke, Mrs. John (Catherine)
    ## 659                                                       Eitemiller, Mr. George Floyd
    ## 660                                                         Newell, Mr. Arthur Webster
    ## 661                                                      Frauenthal, Dr. Henry William
    ## 662                                                                  Badt, Mr. Mohamed
    ## 663                                                         Colley, Mr. Edward Pomeroy
    ## 664                                                                   Coleff, Mr. Peju
    ## 665                                                        Lindqvist, Mr. Eino William
    ## 666                                                                 Hickman, Mr. Lewis
    ## 667                                                        Butler, Mr. Reginald Fenton
    ## 668                                                         Rommetvedt, Mr. Knud Paust
    ## 669                                                                    Cook, Mr. Jacob
    ## 670                                  Taylor, Mrs. Elmer Zebley (Juliet Cummins Wright)
    ## 671                      Brown, Mrs. Thomas William Solomon (Elizabeth Catherine Ford)
    ## 672                                                             Davidson, Mr. Thornton
    ## 673                                                        Mitchell, Mr. Henry Michael
    ## 674                                                              Wilhelms, Mr. Charles
    ## 675                                                         Watson, Mr. Ennis Hastings
    ## 676                                                     Edvardsson, Mr. Gustaf Hjalmar
    ## 677                                                      Sawyer, Mr. Frederick Charles
    ## 678                                                            Turja, Miss. Anna Sofia
    ## 679                                            Goodwin, Mrs. Frederick (Augusta Tyler)
    ## 680                                                 Cardeza, Mr. Thomas Drake Martinez
    ## 681                                                                Peters, Miss. Katie
    ## 682                                                                 Hassab, Mr. Hammad
    ## 683                                                        Olsvigen, Mr. Thor Anderson
    ## 684                                                        Goodwin, Mr. Charles Edward
    ## 685                                                  Brown, Mr. Thomas William Solomon
    ## 686                                             Laroche, Mr. Joseph Philippe Lemercier
    ## 687                                                           Panula, Mr. Jaako Arnold
    ## 688                                                                  Dakic, Mr. Branko
    ## 689                                                    Fischer, Mr. Eberhard Thelander
    ## 690                                                  Madill, Miss. Georgette Alexandra
    ## 691                                                            Dick, Mr. Albert Adrian
    ## 692                                                                 Karun, Miss. Manca
    ## 693                                                                       Lam, Mr. Ali
    ## 694                                                                   Saad, Mr. Khalil
    ## 695                                                                    Weir, Col. John
    ## 696                                                         Chapman, Mr. Charles Henry
    ## 697                                                                   Kelly, Mr. James
    ## 698                                                   Mullens, Miss. Katherine "Katie"
    ## 699                                                           Thayer, Mr. John Borland
    ## 700                                           Humblen, Mr. Adolf Mathias Nicolai Olsen
    ## 701                                  Astor, Mrs. John Jacob (Madeleine Talmadge Force)
    ## 702                                                   Silverthorne, Mr. Spencer Victor
    ## 703                                                              Barbara, Miss. Saiide
    ## 704                                                              Gallagher, Mr. Martin
    ## 705                                                            Hansen, Mr. Henrik Juul
    ## 706                                     Morley, Mr. Henry Samuel ("Mr Henry Marshall")
    ## 707                                                      Kelly, Mrs. Florence "Fannie"
    ## 708                                                  Calderhead, Mr. Edward Pennington
    ## 709                                                               Cleaver, Miss. Alice
    ## 710                                  Moubarek, Master. Halim Gonios ("William George")
    ## 711                                   Mayne, Mlle. Berthe Antonine ("Mrs de Villiers")
    ## 712                                                                 Klaber, Mr. Herman
    ## 713                                                           Taylor, Mr. Elmer Zebley
    ## 714                                                         Larsson, Mr. August Viktor
    ## 715                                                              Greenberg, Mr. Samuel
    ## 716                                         Soholt, Mr. Peter Andreas Lauritz Andersen
    ## 717                                                      Endres, Miss. Caroline Louise
    ## 718                                                Troutt, Miss. Edwina Celia "Winnie"
    ## 719                                                                McEvoy, Mr. Michael
    ## 720                                                       Johnson, Mr. Malkolm Joackim
    ## 721                                                  Harper, Miss. Annie Jessie "Nina"
    ## 722                                                          Jensen, Mr. Svend Lauritz
    ## 723                                                       Gillespie, Mr. William Henry
    ## 724                                                            Hodges, Mr. Henry Price
    ## 725                                                      Chambers, Mr. Norman Campbell
    ## 726                                                                Oreskovic, Mr. Luka
    ## 727                                        Renouf, Mrs. Peter Henry (Lillian Jefferys)
    ## 728                                                           Mannion, Miss. Margareth
    ## 729                                                    Bryhl, Mr. Kurt Arnold Gottfrid
    ## 730                                                      Ilmakangas, Miss. Pieta Sofia
    ## 731                                                      Allen, Miss. Elisabeth Walton
    ## 732                                                           Hassan, Mr. Houssein G N
    ## 733                                                               Knight, Mr. Robert J
    ## 734                                                         Berriman, Mr. William John
    ## 735                                                       Troupiansky, Mr. Moses Aaron
    ## 736                                                               Williams, Mr. Leslie
    ## 737                                            Ford, Mrs. Edward (Margaret Ann Watson)
    ## 738                                                             Lesurer, Mr. Gustave J
    ## 739                                                                 Ivanoff, Mr. Kanio
    ## 740                                                                 Nankoff, Mr. Minko
    ## 741                                                        Hawksford, Mr. Walter James
    ## 742                                                      Cavendish, Mr. Tyrell William
    ## 743                                              Ryerson, Miss. Susan Parker "Suzette"
    ## 744                                                                  McNamee, Mr. Neal
    ## 745                                                                 Stranden, Mr. Juho
    ## 746                                                       Crosby, Capt. Edward Gifford
    ## 747                                                        Abbott, Mr. Rossmore Edward
    ## 748                                                              Sinkkonen, Miss. Anna
    ## 749                                                          Marvin, Mr. Daniel Warner
    ## 750                                                            Connaghton, Mr. Michael
    ## 751                                                                  Wells, Miss. Joan
    ## 752                                                                Moor, Master. Meier
    ## 753                                                   Vande Velde, Mr. Johannes Joseph
    ## 754                                                                 Jonkoff, Mr. Lalio
    ## 755                                                   Herman, Mrs. Samuel (Jane Laver)
    ## 756                                                          Hamalainen, Master. Viljo
    ## 757                                                       Carlsson, Mr. August Sigfrid
    ## 758                                                           Bailey, Mr. Percy Andrew
    ## 759                                                       Theobald, Mr. Thomas Leonard
    ## 760                           Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)
    ## 761                                                                 Garfirth, Mr. John
    ## 762                                                     Nirva, Mr. Iisakki Antino Aijo
    ## 763                                                              Barah, Mr. Hanna Assi
    ## 764                                          Carter, Mrs. William Ernest (Lucile Polk)
    ## 765                                                             Eklund, Mr. Hans Linus
    ## 766                                               Hogeboom, Mrs. John C (Anna Andrews)
    ## 767                                                          Brewe, Dr. Arthur Jackson
    ## 768                                                                 Mangan, Miss. Mary
    ## 769                                                                Moran, Mr. Daniel J
    ## 770                                                   Gronnestad, Mr. Daniel Danielsen
    ## 771                                                             Lievens, Mr. Rene Aime
    ## 772                                                            Jensen, Mr. Niels Peder
    ## 773                                                                  Mack, Mrs. (Mary)
    ## 774                                                                    Elias, Mr. Dibo
    ## 775                                              Hocking, Mrs. Elizabeth (Eliza Needs)
    ## 776                                            Myhrman, Mr. Pehr Fabian Oliver Malkolm
    ## 777                                                                   Tobin, Mr. Roger
    ## 778                                                      Emanuel, Miss. Virginia Ethel
    ## 779                                                            Kilgannon, Mr. Thomas J
    ## 780                              Robert, Mrs. Edward Scott (Elisabeth Walton McMillan)
    ## 781                                                               Ayoub, Miss. Banoura
    ## 782                                          Dick, Mrs. Albert Adrian (Vera Gillespie)
    ## 783                                                             Long, Mr. Milton Clyde
    ## 784                                                             Johnston, Mr. Andrew G
    ## 785                                                                   Ali, Mr. William
    ## 786                                                 Harmer, Mr. Abraham (David Lishin)
    ## 787                                                          Sjoblom, Miss. Anna Sofia
    ## 788                                                          Rice, Master. George Hugh
    ## 789                                                         Dean, Master. Bertram Vere
    ## 790                                                           Guggenheim, Mr. Benjamin
    ## 791                                                           Keane, Mr. Andrew "Andy"
    ## 792                                                                Gaskell, Mr. Alfred
    ## 793                                                            Sage, Miss. Stella Anna
    ## 794                                                           Hoyt, Mr. William Fisher
    ## 795                                                              Dantcheff, Mr. Ristiu
    ## 796                                                                 Otter, Mr. Richard
    ## 797                                                        Leader, Dr. Alice (Farnham)
    ## 798                                                                   Osman, Mrs. Mara
    ## 799                                                       Ibrahim Shawah, Mr. Yousseff
    ## 800                               Van Impe, Mrs. Jean Baptiste (Rosalie Paula Govaert)
    ## 801                                                               Ponesell, Mr. Martin
    ## 802                                        Collyer, Mrs. Harvey (Charlotte Annie Tate)
    ## 803                                                Carter, Master. William Thornton II
    ## 804                                                    Thomas, Master. Assad Alexander
    ## 805                                                            Hedman, Mr. Oskar Arvid
    ## 806                                                          Johansson, Mr. Karl Johan
    ## 807                                                             Andrews, Mr. Thomas Jr
    ## 808                                                    Pettersson, Miss. Ellen Natalia
    ## 809                                                                  Meyer, Mr. August
    ## 810                                     Chambers, Mrs. Norman Campbell (Bertha Griggs)
    ## 811                                                             Alexander, Mr. William
    ## 812                                                                  Lester, Mr. James
    ## 813                                                          Slemen, Mr. Richard James
    ## 814                                                 Andersson, Miss. Ebba Iris Alfrida
    ## 815                                                         Tomlin, Mr. Ernest Portage
    ## 816                                                                   Fry, Mr. Richard
    ## 817                                                       Heininen, Miss. Wendla Maria
    ## 818                                                                 Mallet, Mr. Albert
    ## 819                                                   Holm, Mr. John Fredrik Alexander
    ## 820                                                       Skoog, Master. Karl Thorsten
    ## 821                                 Hays, Mrs. Charles Melville (Clara Jennings Gregg)
    ## 822                                                                  Lulic, Mr. Nikola
    ## 823                                                    Reuchlin, Jonkheer. John George
    ## 824                                                                 Moor, Mrs. (Beila)
    ## 825                                                       Panula, Master. Urho Abraham
    ## 826                                                                    Flynn, Mr. John
    ## 827                                                                       Lam, Mr. Len
    ## 828                                                              Mallet, Master. Andre
    ## 829                                                       McCormack, Mr. Thomas Joseph
    ## 830                                          Stone, Mrs. George Nelson (Martha Evelyn)
    ## 831                                            Yasbeck, Mrs. Antoni (Selini Alexander)
    ## 832                                                    Richards, Master. George Sibley
    ## 833                                                                     Saad, Mr. Amin
    ## 834                                                             Augustsson, Mr. Albert
    ## 835                                                             Allum, Mr. Owen George
    ## 836                                                        Compton, Miss. Sara Rebecca
    ## 837                                                                   Pasic, Mr. Jakob
    ## 838                                                                Sirota, Mr. Maurice
    ## 839                                                                    Chip, Mr. Chang
    ## 840                                                               Marechal, Mr. Pierre
    ## 841                                                        Alhomaki, Mr. Ilmari Rudolf
    ## 842                                                           Mudd, Mr. Thomas Charles
    ## 843                                                            Serepeca, Miss. Augusta
    ## 844                                                         Lemberopolous, Mr. Peter L
    ## 845                                                                Culumovic, Mr. Jeso
    ## 846                                                                Abbing, Mr. Anthony
    ## 847                                                           Sage, Mr. Douglas Bullen
    ## 848                                                                 Markoff, Mr. Marin
    ## 849                                                                  Harper, Rev. John
    ## 850                                       Goldenberg, Mrs. Samuel L (Edwiga Grabowska)
    ## 851                                            Andersson, Master. Sigvard Harald Elias
    ## 852                                                                Svensson, Mr. Johan
    ## 853                                                            Boulos, Miss. Nourelain
    ## 854                                                          Lines, Miss. Mary Conover
    ## 855                                      Carter, Mrs. Ernest Courtenay (Lilian Hughes)
    ## 856                                                         Aks, Mrs. Sam (Leah Rosen)
    ## 857                                         Wick, Mrs. George Dennick (Mary Hitchcock)
    ## 858                                                              Daly, Mr. Peter Denis
    ## 859                                              Baclini, Mrs. Solomon (Latifa Qurban)
    ## 860                                                                   Razi, Mr. Raihed
    ## 861                                                            Hansen, Mr. Claus Peter
    ## 862                                                        Giles, Mr. Frederick Edward
    ## 863                                Swift, Mrs. Frederick Joel (Margaret Welles Barron)
    ## 864                                                  Sage, Miss. Dorothy Edith "Dolly"
    ## 865                                                             Gill, Mr. John William
    ## 866                                                           Bystrom, Mrs. (Karolina)
    ## 867                                                       Duran y More, Miss. Asuncion
    ## 868                                               Roebling, Mr. Washington Augustus II
    ## 869                                                        van Melkebeke, Mr. Philemon
    ## 870                                                    Johnson, Master. Harold Theodor
    ## 871                                                                  Balkic, Mr. Cerin
    ## 872                                   Beckwith, Mrs. Richard Leonard (Sallie Monypeny)
    ## 873                                                           Carlsson, Mr. Frans Olof
    ## 874                                                        Vander Cruyssen, Mr. Victor
    ## 875                                              Abelson, Mrs. Samuel (Hannah Wizosky)
    ## 876                                                   Najib, Miss. Adele Kiamie "Jane"
    ## 877                                                      Gustafsson, Mr. Alfred Ossian
    ## 878                                                               Petroff, Mr. Nedelio
    ## 879                                                                 Laleff, Mr. Kristo
    ## 880                                      Potter, Mrs. Thomas Jr (Lily Alexenia Wilson)
    ## 881                                       Shelley, Mrs. William (Imanita Parrish Hall)
    ## 882                                                                 Markun, Mr. Johann
    ## 883                                                       Dahlberg, Miss. Gerda Ulrika
    ## 884                                                      Banfield, Mr. Frederick James
    ## 885                                                             Sutehall, Mr. Henry Jr
    ## 886                                               Rice, Mrs. William (Margaret Norton)
    ## 887                                                              Montvila, Rev. Juozas
    ## 888                                                       Graham, Miss. Margaret Edith
    ## 889                                           Johnston, Miss. Catherine Helen "Carrie"
    ## 890                                                              Behr, Mr. Karl Howell
    ## 891                                                                Dooley, Mr. Patrick
    ##        Sex      Age SibSp Parch             Ticket     Fare
    ## 1     male 22.00000     1     0          A/5 21171   7.2500
    ## 2   female 38.00000     1     0           PC 17599  71.2833
    ## 3   female 26.00000     0     0   STON/O2. 3101282   7.9250
    ## 4   female 35.00000     1     0             113803  53.1000
    ## 5     male 35.00000     0     0             373450   8.0500
    ## 6     male 29.69912     0     0             330877   8.4583
    ## 7     male 54.00000     0     0              17463  51.8625
    ## 8     male  2.00000     3     1             349909  21.0750
    ## 9   female 27.00000     0     2             347742  11.1333
    ## 10  female 14.00000     1     0             237736  30.0708
    ## 11  female  4.00000     1     1            PP 9549  16.7000
    ## 12  female 58.00000     0     0             113783  26.5500
    ## 13    male 20.00000     0     0          A/5. 2151   8.0500
    ## 14    male 39.00000     1     5             347082  31.2750
    ## 15  female 14.00000     0     0             350406   7.8542
    ## 16  female 55.00000     0     0             248706  16.0000
    ## 17    male  2.00000     4     1             382652  29.1250
    ## 18    male 29.69912     0     0             244373  13.0000
    ## 19  female 31.00000     1     0             345763  18.0000
    ## 20  female 29.69912     0     0               2649   7.2250
    ## 21    male 35.00000     0     0             239865  26.0000
    ## 22    male 34.00000     0     0             248698  13.0000
    ## 23  female 15.00000     0     0             330923   8.0292
    ## 24    male 28.00000     0     0             113788  35.5000
    ## 25  female  8.00000     3     1             349909  21.0750
    ## 26  female 38.00000     1     5             347077  31.3875
    ## 27    male 29.69912     0     0               2631   7.2250
    ## 28    male 19.00000     3     2              19950 263.0000
    ## 29  female 29.69912     0     0             330959   7.8792
    ## 30    male 29.69912     0     0             349216   7.8958
    ## 31    male 40.00000     0     0           PC 17601  27.7208
    ## 32  female 29.69912     1     0           PC 17569 146.5208
    ## 33  female 29.69912     0     0             335677   7.7500
    ## 34    male 66.00000     0     0         C.A. 24579  10.5000
    ## 35    male 28.00000     1     0           PC 17604  82.1708
    ## 36    male 42.00000     1     0             113789  52.0000
    ## 37    male 29.69912     0     0               2677   7.2292
    ## 38    male 21.00000     0     0         A./5. 2152   8.0500
    ## 39  female 18.00000     2     0             345764  18.0000
    ## 40  female 14.00000     1     0               2651  11.2417
    ## 41  female 40.00000     1     0               7546   9.4750
    ## 42  female 27.00000     1     0              11668  21.0000
    ## 43    male 29.69912     0     0             349253   7.8958
    ## 44  female  3.00000     1     2      SC/Paris 2123  41.5792
    ## 45  female 19.00000     0     0             330958   7.8792
    ## 46    male 29.69912     0     0    S.C./A.4. 23567   8.0500
    ## 47    male 29.69912     1     0             370371  15.5000
    ## 48  female 29.69912     0     0              14311   7.7500
    ## 49    male 29.69912     2     0               2662  21.6792
    ## 50  female 18.00000     1     0             349237  17.8000
    ## 51    male  7.00000     4     1            3101295  39.6875
    ## 52    male 21.00000     0     0         A/4. 39886   7.8000
    ## 53  female 49.00000     1     0           PC 17572  76.7292
    ## 54  female 29.00000     1     0               2926  26.0000
    ## 55    male 65.00000     0     1             113509  61.9792
    ## 56    male 29.69912     0     0              19947  35.5000
    ## 57  female 21.00000     0     0         C.A. 31026  10.5000
    ## 58    male 28.50000     0     0               2697   7.2292
    ## 59  female  5.00000     1     2         C.A. 34651  27.7500
    ## 60    male 11.00000     5     2            CA 2144  46.9000
    ## 61    male 22.00000     0     0               2669   7.2292
    ## 62  female 38.00000     0     0             113572  80.0000
    ## 63    male 45.00000     1     0              36973  83.4750
    ## 64    male  4.00000     3     2             347088  27.9000
    ## 65    male 29.69912     0     0           PC 17605  27.7208
    ## 66    male 29.69912     1     1               2661  15.2458
    ## 67  female 29.00000     0     0         C.A. 29395  10.5000
    ## 68    male 19.00000     0     0          S.P. 3464   8.1583
    ## 69  female 17.00000     4     2            3101281   7.9250
    ## 70    male 26.00000     2     0             315151   8.6625
    ## 71    male 32.00000     0     0         C.A. 33111  10.5000
    ## 72  female 16.00000     5     2            CA 2144  46.9000
    ## 73    male 21.00000     0     0       S.O.C. 14879  73.5000
    ## 74    male 26.00000     1     0               2680  14.4542
    ## 75    male 32.00000     0     0               1601  56.4958
    ## 76    male 25.00000     0     0             348123   7.6500
    ## 77    male 29.69912     0     0             349208   7.8958
    ## 78    male 29.69912     0     0             374746   8.0500
    ## 79    male  0.83000     0     2             248738  29.0000
    ## 80  female 30.00000     0     0             364516  12.4750
    ## 81    male 22.00000     0     0             345767   9.0000
    ## 82    male 29.00000     0     0             345779   9.5000
    ## 83  female 29.69912     0     0             330932   7.7875
    ## 84    male 28.00000     0     0             113059  47.1000
    ## 85  female 17.00000     0     0         SO/C 14885  10.5000
    ## 86  female 33.00000     3     0            3101278  15.8500
    ## 87    male 16.00000     1     3         W./C. 6608  34.3750
    ## 88    male 29.69912     0     0    SOTON/OQ 392086   8.0500
    ## 89  female 23.00000     3     2              19950 263.0000
    ## 90    male 24.00000     0     0             343275   8.0500
    ## 91    male 29.00000     0     0             343276   8.0500
    ## 92    male 20.00000     0     0             347466   7.8542
    ## 93    male 46.00000     1     0        W.E.P. 5734  61.1750
    ## 94    male 26.00000     1     2          C.A. 2315  20.5750
    ## 95    male 59.00000     0     0             364500   7.2500
    ## 96    male 29.69912     0     0             374910   8.0500
    ## 97    male 71.00000     0     0           PC 17754  34.6542
    ## 98    male 23.00000     0     1           PC 17759  63.3583
    ## 99  female 34.00000     0     1             231919  23.0000
    ## 100   male 34.00000     1     0             244367  26.0000
    ## 101 female 28.00000     0     0             349245   7.8958
    ## 102   male 29.69912     0     0             349215   7.8958
    ## 103   male 21.00000     0     1              35281  77.2875
    ## 104   male 33.00000     0     0               7540   8.6542
    ## 105   male 37.00000     2     0            3101276   7.9250
    ## 106   male 28.00000     0     0             349207   7.8958
    ## 107 female 21.00000     0     0             343120   7.6500
    ## 108   male 29.69912     0     0             312991   7.7750
    ## 109   male 38.00000     0     0             349249   7.8958
    ## 110 female 29.69912     1     0             371110  24.1500
    ## 111   male 47.00000     0     0             110465  52.0000
    ## 112 female 14.50000     1     0               2665  14.4542
    ## 113   male 22.00000     0     0             324669   8.0500
    ## 114 female 20.00000     1     0               4136   9.8250
    ## 115 female 17.00000     0     0               2627  14.4583
    ## 116   male 21.00000     0     0  STON/O 2. 3101294   7.9250
    ## 117   male 70.50000     0     0             370369   7.7500
    ## 118   male 29.00000     1     0              11668  21.0000
    ## 119   male 24.00000     0     1           PC 17558 247.5208
    ## 120 female  2.00000     4     2             347082  31.2750
    ## 121   male 21.00000     2     0       S.O.C. 14879  73.5000
    ## 122   male 29.69912     0     0          A4. 54510   8.0500
    ## 123   male 32.50000     1     0             237736  30.0708
    ## 124 female 32.50000     0     0              27267  13.0000
    ## 125   male 54.00000     0     1              35281  77.2875
    ## 126   male 12.00000     1     0               2651  11.2417
    ## 127   male 29.69912     0     0             370372   7.7500
    ## 128   male 24.00000     0     0            C 17369   7.1417
    ## 129 female 29.69912     1     1               2668  22.3583
    ## 130   male 45.00000     0     0             347061   6.9750
    ## 131   male 33.00000     0     0             349241   7.8958
    ## 132   male 20.00000     0     0 SOTON/O.Q. 3101307   7.0500
    ## 133 female 47.00000     1     0          A/5. 3337  14.5000
    ## 134 female 29.00000     1     0             228414  26.0000
    ## 135   male 25.00000     0     0         C.A. 29178  13.0000
    ## 136   male 23.00000     0     0      SC/PARIS 2133  15.0458
    ## 137 female 19.00000     0     2              11752  26.2833
    ## 138   male 37.00000     1     0             113803  53.1000
    ## 139   male 16.00000     0     0               7534   9.2167
    ## 140   male 24.00000     0     0           PC 17593  79.2000
    ## 141 female 29.69912     0     2               2678  15.2458
    ## 142 female 22.00000     0     0             347081   7.7500
    ## 143 female 24.00000     1     0   STON/O2. 3101279  15.8500
    ## 144   male 19.00000     0     0             365222   6.7500
    ## 145   male 18.00000     0     0             231945  11.5000
    ## 146   male 19.00000     1     1         C.A. 33112  36.7500
    ## 147   male 27.00000     0     0             350043   7.7958
    ## 148 female  9.00000     2     2         W./C. 6608  34.3750
    ## 149   male 36.50000     0     2             230080  26.0000
    ## 150   male 42.00000     0     0             244310  13.0000
    ## 151   male 51.00000     0     0        S.O.P. 1166  12.5250
    ## 152 female 22.00000     1     0             113776  66.6000
    ## 153   male 55.50000     0     0         A.5. 11206   8.0500
    ## 154   male 40.50000     0     2           A/5. 851  14.5000
    ## 155   male 29.69912     0     0          Fa 265302   7.3125
    ## 156   male 51.00000     0     1           PC 17597  61.3792
    ## 157 female 16.00000     0     0              35851   7.7333
    ## 158   male 30.00000     0     0    SOTON/OQ 392090   8.0500
    ## 159   male 29.69912     0     0             315037   8.6625
    ## 160   male 29.69912     8     2           CA. 2343  69.5500
    ## 161   male 44.00000     0     1             371362  16.1000
    ## 162 female 40.00000     0     0         C.A. 33595  15.7500
    ## 163   male 26.00000     0     0             347068   7.7750
    ## 164   male 17.00000     0     0             315093   8.6625
    ## 165   male  1.00000     4     1            3101295  39.6875
    ## 166   male  9.00000     0     2             363291  20.5250
    ## 167 female 29.69912     0     1             113505  55.0000
    ## 168 female 45.00000     1     4             347088  27.9000
    ## 169   male 29.69912     0     0           PC 17318  25.9250
    ## 170   male 28.00000     0     0               1601  56.4958
    ## 171   male 61.00000     0     0             111240  33.5000
    ## 172   male  4.00000     4     1             382652  29.1250
    ## 173 female  1.00000     1     1             347742  11.1333
    ## 174   male 21.00000     0     0  STON/O 2. 3101280   7.9250
    ## 175   male 56.00000     0     0              17764  30.6958
    ## 176   male 18.00000     1     1             350404   7.8542
    ## 177   male 29.69912     3     1               4133  25.4667
    ## 178 female 50.00000     0     0           PC 17595  28.7125
    ## 179   male 30.00000     0     0             250653  13.0000
    ## 180   male 36.00000     0     0               LINE   0.0000
    ## 181 female 29.69912     8     2           CA. 2343  69.5500
    ## 182   male 29.69912     0     0      SC/PARIS 2131  15.0500
    ## 183   male  9.00000     4     2             347077  31.3875
    ## 184   male  1.00000     2     1             230136  39.0000
    ## 185 female  4.00000     0     2             315153  22.0250
    ## 186   male 29.69912     0     0             113767  50.0000
    ## 187 female 29.69912     1     0             370365  15.5000
    ## 188   male 45.00000     0     0             111428  26.5500
    ## 189   male 40.00000     1     1             364849  15.5000
    ## 190   male 36.00000     0     0             349247   7.8958
    ## 191 female 32.00000     0     0             234604  13.0000
    ## 192   male 19.00000     0     0              28424  13.0000
    ## 193 female 19.00000     1     0             350046   7.8542
    ## 194   male  3.00000     1     1             230080  26.0000
    ## 195 female 44.00000     0     0           PC 17610  27.7208
    ## 196 female 58.00000     0     0           PC 17569 146.5208
    ## 197   male 29.69912     0     0             368703   7.7500
    ## 198   male 42.00000     0     1               4579   8.4042
    ## 199 female 29.69912     0     0             370370   7.7500
    ## 200 female 24.00000     0     0             248747  13.0000
    ## 201   male 28.00000     0     0             345770   9.5000
    ## 202   male 29.69912     8     2           CA. 2343  69.5500
    ## 203   male 34.00000     0     0            3101264   6.4958
    ## 204   male 45.50000     0     0               2628   7.2250
    ## 205   male 18.00000     0     0           A/5 3540   8.0500
    ## 206 female  2.00000     0     1             347054  10.4625
    ## 207   male 32.00000     1     0            3101278  15.8500
    ## 208   male 26.00000     0     0               2699  18.7875
    ## 209 female 16.00000     0     0             367231   7.7500
    ## 210   male 40.00000     0     0             112277  31.0000
    ## 211   male 24.00000     0     0 SOTON/O.Q. 3101311   7.0500
    ## 212 female 35.00000     0     0       F.C.C. 13528  21.0000
    ## 213   male 22.00000     0     0          A/5 21174   7.2500
    ## 214   male 30.00000     0     0             250646  13.0000
    ## 215   male 29.69912     1     0             367229   7.7500
    ## 216 female 31.00000     1     0              35273 113.2750
    ## 217 female 27.00000     0     0   STON/O2. 3101283   7.9250
    ## 218   male 42.00000     1     0             243847  27.0000
    ## 219 female 32.00000     0     0              11813  76.2917
    ## 220   male 30.00000     0     0          W/C 14208  10.5000
    ## 221   male 16.00000     0     0    SOTON/OQ 392089   8.0500
    ## 222   male 27.00000     0     0             220367  13.0000
    ## 223   male 51.00000     0     0              21440   8.0500
    ## 224   male 29.69912     0     0             349234   7.8958
    ## 225   male 38.00000     1     0              19943  90.0000
    ## 226   male 22.00000     0     0            PP 4348   9.3500
    ## 227   male 19.00000     0     0          SW/PP 751  10.5000
    ## 228   male 20.50000     0     0          A/5 21173   7.2500
    ## 229   male 18.00000     0     0             236171  13.0000
    ## 230 female 29.69912     3     1               4133  25.4667
    ## 231 female 35.00000     1     0              36973  83.4750
    ## 232   male 29.00000     0     0             347067   7.7750
    ## 233   male 59.00000     0     0             237442  13.5000
    ## 234 female  5.00000     4     2             347077  31.3875
    ## 235   male 24.00000     0     0         C.A. 29566  10.5000
    ## 236 female 29.69912     0     0         W./C. 6609   7.5500
    ## 237   male 44.00000     1     0              26707  26.0000
    ## 238 female  8.00000     0     2         C.A. 31921  26.2500
    ## 239   male 19.00000     0     0              28665  10.5000
    ## 240   male 33.00000     0     0         SCO/W 1585  12.2750
    ## 241 female 29.69912     1     0               2665  14.4542
    ## 242 female 29.69912     1     0             367230  15.5000
    ## 243   male 29.00000     0     0        W./C. 14263  10.5000
    ## 244   male 22.00000     0     0  STON/O 2. 3101275   7.1250
    ## 245   male 30.00000     0     0               2694   7.2250
    ## 246   male 44.00000     2     0              19928  90.0000
    ## 247 female 25.00000     0     0             347071   7.7750
    ## 248 female 24.00000     0     2             250649  14.5000
    ## 249   male 37.00000     1     1              11751  52.5542
    ## 250   male 54.00000     1     0             244252  26.0000
    ## 251   male 29.69912     0     0             362316   7.2500
    ## 252 female 29.00000     1     1             347054  10.4625
    ## 253   male 62.00000     0     0             113514  26.5500
    ## 254   male 30.00000     1     0          A/5. 3336  16.1000
    ## 255 female 41.00000     0     2             370129  20.2125
    ## 256 female 29.00000     0     2               2650  15.2458
    ## 257 female 29.69912     0     0           PC 17585  79.2000
    ## 258 female 30.00000     0     0             110152  86.5000
    ## 259 female 35.00000     0     0           PC 17755 512.3292
    ## 260 female 50.00000     0     1             230433  26.0000
    ## 261   male 29.69912     0     0             384461   7.7500
    ## 262   male  3.00000     4     2             347077  31.3875
    ## 263   male 52.00000     1     1             110413  79.6500
    ## 264   male 40.00000     0     0             112059   0.0000
    ## 265 female 29.69912     0     0             382649   7.7500
    ## 266   male 36.00000     0     0         C.A. 17248  10.5000
    ## 267   male 16.00000     4     1            3101295  39.6875
    ## 268   male 25.00000     1     0             347083   7.7750
    ## 269 female 58.00000     0     1           PC 17582 153.4625
    ## 270 female 35.00000     0     0           PC 17760 135.6333
    ## 271   male 29.69912     0     0             113798  31.0000
    ## 272   male 25.00000     0     0               LINE   0.0000
    ## 273 female 41.00000     0     1             250644  19.5000
    ## 274   male 37.00000     0     1           PC 17596  29.7000
    ## 275 female 29.69912     0     0             370375   7.7500
    ## 276 female 63.00000     1     0              13502  77.9583
    ## 277 female 45.00000     0     0             347073   7.7500
    ## 278   male 29.69912     0     0             239853   0.0000
    ## 279   male  7.00000     4     1             382652  29.1250
    ## 280 female 35.00000     1     1          C.A. 2673  20.2500
    ## 281   male 65.00000     0     0             336439   7.7500
    ## 282   male 28.00000     0     0             347464   7.8542
    ## 283   male 16.00000     0     0             345778   9.5000
    ## 284   male 19.00000     0     0         A/5. 10482   8.0500
    ## 285   male 29.69912     0     0             113056  26.0000
    ## 286   male 33.00000     0     0             349239   8.6625
    ## 287   male 30.00000     0     0             345774   9.5000
    ## 288   male 22.00000     0     0             349206   7.8958
    ## 289   male 42.00000     0     0             237798  13.0000
    ## 290 female 22.00000     0     0             370373   7.7500
    ## 291 female 26.00000     0     0              19877  78.8500
    ## 292 female 19.00000     1     0              11967  91.0792
    ## 293   male 36.00000     0     0      SC/Paris 2163  12.8750
    ## 294 female 24.00000     0     0             349236   8.8500
    ## 295   male 24.00000     0     0             349233   7.8958
    ## 296   male 29.69912     0     0           PC 17612  27.7208
    ## 297   male 23.50000     0     0               2693   7.2292
    ## 298 female  2.00000     1     2             113781 151.5500
    ## 299   male 29.69912     0     0              19988  30.5000
    ## 300 female 50.00000     0     1           PC 17558 247.5208
    ## 301 female 29.69912     0     0               9234   7.7500
    ## 302   male 29.69912     2     0             367226  23.2500
    ## 303   male 19.00000     0     0               LINE   0.0000
    ## 304 female 29.69912     0     0             226593  12.3500
    ## 305   male 29.69912     0     0           A/5 2466   8.0500
    ## 306   male  0.92000     1     2             113781 151.5500
    ## 307 female 29.69912     0     0              17421 110.8833
    ## 308 female 17.00000     1     0           PC 17758 108.9000
    ## 309   male 30.00000     1     0          P/PP 3381  24.0000
    ## 310 female 30.00000     0     0           PC 17485  56.9292
    ## 311 female 24.00000     0     0              11767  83.1583
    ## 312 female 18.00000     2     2           PC 17608 262.3750
    ## 313 female 26.00000     1     1             250651  26.0000
    ## 314   male 28.00000     0     0             349243   7.8958
    ## 315   male 43.00000     1     1       F.C.C. 13529  26.2500
    ## 316 female 26.00000     0     0             347470   7.8542
    ## 317 female 24.00000     1     0             244367  26.0000
    ## 318   male 54.00000     0     0              29011  14.0000
    ## 319 female 31.00000     0     2              36928 164.8667
    ## 320 female 40.00000     1     1              16966 134.5000
    ## 321   male 22.00000     0     0          A/5 21172   7.2500
    ## 322   male 27.00000     0     0             349219   7.8958
    ## 323 female 30.00000     0     0             234818  12.3500
    ## 324 female 22.00000     1     1             248738  29.0000
    ## 325   male 29.69912     8     2           CA. 2343  69.5500
    ## 326 female 36.00000     0     0           PC 17760 135.6333
    ## 327   male 61.00000     0     0             345364   6.2375
    ## 328 female 36.00000     0     0              28551  13.0000
    ## 329 female 31.00000     1     1             363291  20.5250
    ## 330 female 16.00000     0     1             111361  57.9792
    ## 331 female 29.69912     2     0             367226  23.2500
    ## 332   male 45.50000     0     0             113043  28.5000
    ## 333   male 38.00000     0     1           PC 17582 153.4625
    ## 334   male 16.00000     2     0             345764  18.0000
    ## 335 female 29.69912     1     0           PC 17611 133.6500
    ## 336   male 29.69912     0     0             349225   7.8958
    ## 337   male 29.00000     1     0             113776  66.6000
    ## 338 female 41.00000     0     0              16966 134.5000
    ## 339   male 45.00000     0     0               7598   8.0500
    ## 340   male 45.00000     0     0             113784  35.5000
    ## 341   male  2.00000     1     1             230080  26.0000
    ## 342 female 24.00000     3     2              19950 263.0000
    ## 343   male 28.00000     0     0             248740  13.0000
    ## 344   male 25.00000     0     0             244361  13.0000
    ## 345   male 36.00000     0     0             229236  13.0000
    ## 346 female 24.00000     0     0             248733  13.0000
    ## 347 female 40.00000     0     0              31418  13.0000
    ## 348 female 29.69912     1     0             386525  16.1000
    ## 349   male  3.00000     1     1         C.A. 37671  15.9000
    ## 350   male 42.00000     0     0             315088   8.6625
    ## 351   male 23.00000     0     0               7267   9.2250
    ## 352   male 29.69912     0     0             113510  35.0000
    ## 353   male 15.00000     1     1               2695   7.2292
    ## 354   male 25.00000     1     0             349237  17.8000
    ## 355   male 29.69912     0     0               2647   7.2250
    ## 356   male 28.00000     0     0             345783   9.5000
    ## 357 female 22.00000     0     1             113505  55.0000
    ## 358 female 38.00000     0     0             237671  13.0000
    ## 359 female 29.69912     0     0             330931   7.8792
    ## 360 female 29.69912     0     0             330980   7.8792
    ## 361   male 40.00000     1     4             347088  27.9000
    ## 362   male 29.00000     1     0      SC/PARIS 2167  27.7208
    ## 363 female 45.00000     0     1               2691  14.4542
    ## 364   male 35.00000     0     0 SOTON/O.Q. 3101310   7.0500
    ## 365   male 29.69912     1     0             370365  15.5000
    ## 366   male 30.00000     0     0             C 7076   7.2500
    ## 367 female 60.00000     1     0             110813  75.2500
    ## 368 female 29.69912     0     0               2626   7.2292
    ## 369 female 29.69912     0     0              14313   7.7500
    ## 370 female 24.00000     0     0           PC 17477  69.3000
    ## 371   male 25.00000     1     0              11765  55.4417
    ## 372   male 18.00000     1     0            3101267   6.4958
    ## 373   male 19.00000     0     0             323951   8.0500
    ## 374   male 22.00000     0     0           PC 17760 135.6333
    ## 375 female  3.00000     3     1             349909  21.0750
    ## 376 female 29.69912     1     0           PC 17604  82.1708
    ## 377 female 22.00000     0     0             C 7077   7.2500
    ## 378   male 27.00000     0     2             113503 211.5000
    ## 379   male 20.00000     0     0               2648   4.0125
    ## 380   male 19.00000     0     0             347069   7.7750
    ## 381 female 42.00000     0     0           PC 17757 227.5250
    ## 382 female  1.00000     0     2               2653  15.7417
    ## 383   male 32.00000     0     0  STON/O 2. 3101293   7.9250
    ## 384 female 35.00000     1     0             113789  52.0000
    ## 385   male 29.69912     0     0             349227   7.8958
    ## 386   male 18.00000     0     0       S.O.C. 14879  73.5000
    ## 387   male  1.00000     5     2            CA 2144  46.9000
    ## 388 female 36.00000     0     0              27849  13.0000
    ## 389   male 29.69912     0     0             367655   7.7292
    ## 390 female 17.00000     0     0            SC 1748  12.0000
    ## 391   male 36.00000     1     2             113760 120.0000
    ## 392   male 21.00000     0     0             350034   7.7958
    ## 393   male 28.00000     2     0            3101277   7.9250
    ## 394 female 23.00000     1     0              35273 113.2750
    ## 395 female 24.00000     0     2            PP 9549  16.7000
    ## 396   male 22.00000     0     0             350052   7.7958
    ## 397 female 31.00000     0     0             350407   7.8542
    ## 398   male 46.00000     0     0              28403  26.0000
    ## 399   male 23.00000     0     0             244278  10.5000
    ## 400 female 28.00000     0     0             240929  12.6500
    ## 401   male 39.00000     0     0  STON/O 2. 3101289   7.9250
    ## 402   male 26.00000     0     0             341826   8.0500
    ## 403 female 21.00000     1     0               4137   9.8250
    ## 404   male 28.00000     1     0   STON/O2. 3101279  15.8500
    ## 405 female 20.00000     0     0             315096   8.6625
    ## 406   male 34.00000     1     0              28664  21.0000
    ## 407   male 51.00000     0     0             347064   7.7500
    ## 408   male  3.00000     1     1              29106  18.7500
    ## 409   male 21.00000     0     0             312992   7.7750
    ## 410 female 29.69912     3     1               4133  25.4667
    ## 411   male 29.69912     0     0             349222   7.8958
    ## 412   male 29.69912     0     0             394140   6.8583
    ## 413 female 33.00000     1     0              19928  90.0000
    ## 414   male 29.69912     0     0             239853   0.0000
    ## 415   male 44.00000     0     0  STON/O 2. 3101269   7.9250
    ## 416 female 29.69912     0     0             343095   8.0500
    ## 417 female 34.00000     1     1              28220  32.5000
    ## 418 female 18.00000     0     2             250652  13.0000
    ## 419   male 30.00000     0     0              28228  13.0000
    ## 420 female 10.00000     0     2             345773  24.1500
    ## 421   male 29.69912     0     0             349254   7.8958
    ## 422   male 21.00000     0     0         A/5. 13032   7.7333
    ## 423   male 29.00000     0     0             315082   7.8750
    ## 424 female 28.00000     1     1             347080  14.4000
    ## 425   male 18.00000     1     1             370129  20.2125
    ## 426   male 29.69912     0     0         A/4. 34244   7.2500
    ## 427 female 28.00000     1     0               2003  26.0000
    ## 428 female 19.00000     0     0             250655  26.0000
    ## 429   male 29.69912     0     0             364851   7.7500
    ## 430   male 32.00000     0     0  SOTON/O.Q. 392078   8.0500
    ## 431   male 28.00000     0     0             110564  26.5500
    ## 432 female 29.69912     1     0             376564  16.1000
    ## 433 female 42.00000     1     0         SC/AH 3085  26.0000
    ## 434   male 17.00000     0     0  STON/O 2. 3101274   7.1250
    ## 435   male 50.00000     1     0              13507  55.9000
    ## 436 female 14.00000     1     2             113760 120.0000
    ## 437 female 21.00000     2     2         W./C. 6608  34.3750
    ## 438 female 24.00000     2     3              29106  18.7500
    ## 439   male 64.00000     1     4              19950 263.0000
    ## 440   male 31.00000     0     0         C.A. 18723  10.5000
    ## 441 female 45.00000     1     1       F.C.C. 13529  26.2500
    ## 442   male 20.00000     0     0             345769   9.5000
    ## 443   male 25.00000     1     0             347076   7.7750
    ## 444 female 28.00000     0     0             230434  13.0000
    ## 445   male 29.69912     0     0              65306   8.1125
    ## 446   male  4.00000     0     2              33638  81.8583
    ## 447 female 13.00000     0     1             250644  19.5000
    ## 448   male 34.00000     0     0             113794  26.5500
    ## 449 female  5.00000     2     1               2666  19.2583
    ## 450   male 52.00000     0     0             113786  30.5000
    ## 451   male 36.00000     1     2         C.A. 34651  27.7500
    ## 452   male 29.69912     1     0              65303  19.9667
    ## 453   male 30.00000     0     0             113051  27.7500
    ## 454   male 49.00000     1     0              17453  89.1042
    ## 455   male 29.69912     0     0           A/5 2817   8.0500
    ## 456   male 29.00000     0     0             349240   7.8958
    ## 457   male 65.00000     0     0              13509  26.5500
    ## 458 female 29.69912     1     0              17464  51.8625
    ## 459 female 50.00000     0     0       F.C.C. 13531  10.5000
    ## 460   male 29.69912     0     0             371060   7.7500
    ## 461   male 48.00000     0     0              19952  26.5500
    ## 462   male 34.00000     0     0             364506   8.0500
    ## 463   male 47.00000     0     0             111320  38.5000
    ## 464   male 48.00000     0     0             234360  13.0000
    ## 465   male 29.69912     0     0           A/S 2816   8.0500
    ## 466   male 38.00000     0     0 SOTON/O.Q. 3101306   7.0500
    ## 467   male 29.69912     0     0             239853   0.0000
    ## 468   male 56.00000     0     0             113792  26.5500
    ## 469   male 29.69912     0     0              36209   7.7250
    ## 470 female  0.75000     2     1               2666  19.2583
    ## 471   male 29.69912     0     0             323592   7.2500
    ## 472   male 38.00000     0     0             315089   8.6625
    ## 473 female 33.00000     1     2         C.A. 34651  27.7500
    ## 474 female 23.00000     0     0    SC/AH Basle 541  13.7917
    ## 475 female 22.00000     0     0               7553   9.8375
    ## 476   male 29.69912     0     0             110465  52.0000
    ## 477   male 34.00000     1     0              31027  21.0000
    ## 478   male 29.00000     1     0               3460   7.0458
    ## 479   male 22.00000     0     0             350060   7.5208
    ## 480 female  2.00000     0     1            3101298  12.2875
    ## 481   male  9.00000     5     2            CA 2144  46.9000
    ## 482   male 29.69912     0     0             239854   0.0000
    ## 483   male 50.00000     0     0           A/5 3594   8.0500
    ## 484 female 63.00000     0     0               4134   9.5875
    ## 485   male 25.00000     1     0              11967  91.0792
    ## 486 female 29.69912     3     1               4133  25.4667
    ## 487 female 35.00000     1     0              19943  90.0000
    ## 488   male 58.00000     0     0              11771  29.7000
    ## 489   male 30.00000     0     0         A.5. 18509   8.0500
    ## 490   male  9.00000     1     1         C.A. 37671  15.9000
    ## 491   male 29.69912     1     0              65304  19.9667
    ## 492   male 21.00000     0     0   SOTON/OQ 3101317   7.2500
    ## 493   male 55.00000     0     0             113787  30.5000
    ## 494   male 71.00000     0     0           PC 17609  49.5042
    ## 495   male 21.00000     0     0          A/4 45380   8.0500
    ## 496   male 29.69912     0     0               2627  14.4583
    ## 497 female 54.00000     1     0              36947  78.2667
    ## 498   male 29.69912     0     0          C.A. 6212  15.1000
    ## 499 female 25.00000     1     2             113781 151.5500
    ## 500   male 24.00000     0     0             350035   7.7958
    ## 501   male 17.00000     0     0             315086   8.6625
    ## 502 female 21.00000     0     0             364846   7.7500
    ## 503 female 29.69912     0     0             330909   7.6292
    ## 504 female 37.00000     0     0               4135   9.5875
    ## 505 female 16.00000     0     0             110152  86.5000
    ## 506   male 18.00000     1     0           PC 17758 108.9000
    ## 507 female 33.00000     0     2              26360  26.0000
    ## 508   male 29.69912     0     0             111427  26.5500
    ## 509   male 28.00000     0     0             C 4001  22.5250
    ## 510   male 26.00000     0     0               1601  56.4958
    ## 511   male 29.00000     0     0             382651   7.7500
    ## 512   male 29.69912     0     0   SOTON/OQ 3101316   8.0500
    ## 513   male 36.00000     0     0           PC 17473  26.2875
    ## 514 female 54.00000     1     0           PC 17603  59.4000
    ## 515   male 24.00000     0     0             349209   7.4958
    ## 516   male 47.00000     0     0              36967  34.0208
    ## 517 female 34.00000     0     0         C.A. 34260  10.5000
    ## 518   male 29.69912     0     0             371110  24.1500
    ## 519 female 36.00000     1     0             226875  26.0000
    ## 520   male 32.00000     0     0             349242   7.8958
    ## 521 female 30.00000     0     0              12749  93.5000
    ## 522   male 22.00000     0     0             349252   7.8958
    ## 523   male 29.69912     0     0               2624   7.2250
    ## 524 female 44.00000     0     1             111361  57.9792
    ## 525   male 29.69912     0     0               2700   7.2292
    ## 526   male 40.50000     0     0             367232   7.7500
    ## 527 female 50.00000     0     0        W./C. 14258  10.5000
    ## 528   male 29.69912     0     0           PC 17483 221.7792
    ## 529   male 39.00000     0     0            3101296   7.9250
    ## 530   male 23.00000     2     1              29104  11.5000
    ## 531 female  2.00000     1     1              26360  26.0000
    ## 532   male 29.69912     0     0               2641   7.2292
    ## 533   male 17.00000     1     1               2690   7.2292
    ## 534 female 29.69912     0     2               2668  22.3583
    ## 535 female 30.00000     0     0             315084   8.6625
    ## 536 female  7.00000     0     2       F.C.C. 13529  26.2500
    ## 537   male 45.00000     0     0             113050  26.5500
    ## 538 female 30.00000     0     0           PC 17761 106.4250
    ## 539   male 29.69912     0     0             364498  14.5000
    ## 540 female 22.00000     0     2              13568  49.5000
    ## 541 female 36.00000     0     2          WE/P 5735  71.0000
    ## 542 female  9.00000     4     2             347082  31.2750
    ## 543 female 11.00000     4     2             347082  31.2750
    ## 544   male 32.00000     1     0               2908  26.0000
    ## 545   male 50.00000     1     0           PC 17761 106.4250
    ## 546   male 64.00000     0     0                693  26.0000
    ## 547 female 19.00000     1     0               2908  26.0000
    ## 548   male 29.69912     0     0      SC/PARIS 2146  13.8625
    ## 549   male 33.00000     1     1             363291  20.5250
    ## 550   male  8.00000     1     1         C.A. 33112  36.7500
    ## 551   male 17.00000     0     2              17421 110.8833
    ## 552   male 27.00000     0     0             244358  26.0000
    ## 553   male 29.69912     0     0             330979   7.8292
    ## 554   male 22.00000     0     0               2620   7.2250
    ## 555 female 22.00000     0     0             347085   7.7750
    ## 556   male 62.00000     0     0             113807  26.5500
    ## 557 female 48.00000     1     0              11755  39.6000
    ## 558   male 29.69912     0     0           PC 17757 227.5250
    ## 559 female 39.00000     1     1             110413  79.6500
    ## 560 female 36.00000     1     0             345572  17.4000
    ## 561   male 29.69912     0     0             372622   7.7500
    ## 562   male 40.00000     0     0             349251   7.8958
    ## 563   male 28.00000     0     0             218629  13.5000
    ## 564   male 29.69912     0     0    SOTON/OQ 392082   8.0500
    ## 565 female 29.69912     0     0  SOTON/O.Q. 392087   8.0500
    ## 566   male 24.00000     2     0          A/4 48871  24.1500
    ## 567   male 19.00000     0     0             349205   7.8958
    ## 568 female 29.00000     0     4             349909  21.0750
    ## 569   male 29.69912     0     0               2686   7.2292
    ## 570   male 32.00000     0     0             350417   7.8542
    ## 571   male 62.00000     0     0        S.W./PP 752  10.5000
    ## 572 female 53.00000     2     0              11769  51.4792
    ## 573   male 36.00000     0     0           PC 17474  26.3875
    ## 574 female 29.69912     0     0              14312   7.7500
    ## 575   male 16.00000     0     0         A/4. 20589   8.0500
    ## 576   male 19.00000     0     0             358585  14.5000
    ## 577 female 34.00000     0     0             243880  13.0000
    ## 578 female 39.00000     1     0              13507  55.9000
    ## 579 female 29.69912     1     0               2689  14.4583
    ## 580   male 32.00000     0     0  STON/O 2. 3101286   7.9250
    ## 581 female 25.00000     1     1             237789  30.0000
    ## 582 female 39.00000     1     1              17421 110.8833
    ## 583   male 54.00000     0     0              28403  26.0000
    ## 584   male 36.00000     0     0              13049  40.1250
    ## 585   male 29.69912     0     0               3411   8.7125
    ## 586 female 18.00000     0     2             110413  79.6500
    ## 587   male 47.00000     0     0             237565  15.0000
    ## 588   male 60.00000     1     1              13567  79.2000
    ## 589   male 22.00000     0     0              14973   8.0500
    ## 590   male 29.69912     0     0         A./5. 3235   8.0500
    ## 591   male 35.00000     0     0  STON/O 2. 3101273   7.1250
    ## 592 female 52.00000     1     0              36947  78.2667
    ## 593   male 47.00000     0     0           A/5 3902   7.2500
    ## 594 female 29.69912     0     2             364848   7.7500
    ## 595   male 37.00000     1     0        SC/AH 29037  26.0000
    ## 596   male 36.00000     1     1             345773  24.1500
    ## 597 female 29.69912     0     0             248727  33.0000
    ## 598   male 49.00000     0     0               LINE   0.0000
    ## 599   male 29.69912     0     0               2664   7.2250
    ## 600   male 49.00000     1     0           PC 17485  56.9292
    ## 601 female 24.00000     2     1             243847  27.0000
    ## 602   male 29.69912     0     0             349214   7.8958
    ## 603   male 29.69912     0     0             113796  42.4000
    ## 604   male 44.00000     0     0             364511   8.0500
    ## 605   male 35.00000     0     0             111426  26.5500
    ## 606   male 36.00000     1     0             349910  15.5500
    ## 607   male 30.00000     0     0             349246   7.8958
    ## 608   male 27.00000     0     0             113804  30.5000
    ## 609 female 22.00000     1     2      SC/Paris 2123  41.5792
    ## 610 female 40.00000     0     0           PC 17582 153.4625
    ## 611 female 39.00000     1     5             347082  31.2750
    ## 612   male 29.69912     0     0 SOTON/O.Q. 3101305   7.0500
    ## 613 female 29.69912     1     0             367230  15.5000
    ## 614   male 29.69912     0     0             370377   7.7500
    ## 615   male 35.00000     0     0             364512   8.0500
    ## 616 female 24.00000     1     2             220845  65.0000
    ## 617   male 34.00000     1     1             347080  14.4000
    ## 618 female 26.00000     1     0          A/5. 3336  16.1000
    ## 619 female  4.00000     2     1             230136  39.0000
    ## 620   male 26.00000     0     0              31028  10.5000
    ## 621   male 27.00000     1     0               2659  14.4542
    ## 622   male 42.00000     1     0              11753  52.5542
    ## 623   male 20.00000     1     1               2653  15.7417
    ## 624   male 21.00000     0     0             350029   7.8542
    ## 625   male 21.00000     0     0              54636  16.1000
    ## 626   male 61.00000     0     0              36963  32.3208
    ## 627   male 57.00000     0     0             219533  12.3500
    ## 628 female 21.00000     0     0              13502  77.9583
    ## 629   male 26.00000     0     0             349224   7.8958
    ## 630   male 29.69912     0     0             334912   7.7333
    ## 631   male 80.00000     0     0              27042  30.0000
    ## 632   male 51.00000     0     0             347743   7.0542
    ## 633   male 32.00000     0     0              13214  30.5000
    ## 634   male 29.69912     0     0             112052   0.0000
    ## 635 female  9.00000     3     2             347088  27.9000
    ## 636 female 28.00000     0     0             237668  13.0000
    ## 637   male 32.00000     0     0  STON/O 2. 3101292   7.9250
    ## 638   male 31.00000     1     1         C.A. 31921  26.2500
    ## 639 female 41.00000     0     5            3101295  39.6875
    ## 640   male 29.69912     1     0             376564  16.1000
    ## 641   male 20.00000     0     0             350050   7.8542
    ## 642 female 24.00000     0     0           PC 17477  69.3000
    ## 643 female  2.00000     3     2             347088  27.9000
    ## 644   male 29.69912     0     0               1601  56.4958
    ## 645 female  0.75000     2     1               2666  19.2583
    ## 646   male 48.00000     1     0           PC 17572  76.7292
    ## 647   male 19.00000     0     0             349231   7.8958
    ## 648   male 56.00000     0     0              13213  35.5000
    ## 649   male 29.69912     0     0      S.O./P.P. 751   7.5500
    ## 650 female 23.00000     0     0           CA. 2314   7.5500
    ## 651   male 29.69912     0     0             349221   7.8958
    ## 652 female 18.00000     0     1             231919  23.0000
    ## 653   male 21.00000     0     0               8475   8.4333
    ## 654 female 29.69912     0     0             330919   7.8292
    ## 655 female 18.00000     0     0             365226   6.7500
    ## 656   male 24.00000     2     0       S.O.C. 14879  73.5000
    ## 657   male 29.69912     0     0             349223   7.8958
    ## 658 female 32.00000     1     1             364849  15.5000
    ## 659   male 23.00000     0     0              29751  13.0000
    ## 660   male 58.00000     0     2              35273 113.2750
    ## 661   male 50.00000     2     0           PC 17611 133.6500
    ## 662   male 40.00000     0     0               2623   7.2250
    ## 663   male 47.00000     0     0               5727  25.5875
    ## 664   male 36.00000     0     0             349210   7.4958
    ## 665   male 20.00000     1     0  STON/O 2. 3101285   7.9250
    ## 666   male 32.00000     2     0       S.O.C. 14879  73.5000
    ## 667   male 25.00000     0     0             234686  13.0000
    ## 668   male 29.69912     0     0             312993   7.7750
    ## 669   male 43.00000     0     0           A/5 3536   8.0500
    ## 670 female 29.69912     1     0              19996  52.0000
    ## 671 female 40.00000     1     1              29750  39.0000
    ## 672   male 31.00000     1     0         F.C. 12750  52.0000
    ## 673   male 70.00000     0     0         C.A. 24580  10.5000
    ## 674   male 31.00000     0     0             244270  13.0000
    ## 675   male 29.69912     0     0             239856   0.0000
    ## 676   male 18.00000     0     0             349912   7.7750
    ## 677   male 24.50000     0     0             342826   8.0500
    ## 678 female 18.00000     0     0               4138   9.8417
    ## 679 female 43.00000     1     6            CA 2144  46.9000
    ## 680   male 36.00000     0     1           PC 17755 512.3292
    ## 681 female 29.69912     0     0             330935   8.1375
    ## 682   male 27.00000     0     0           PC 17572  76.7292
    ## 683   male 20.00000     0     0               6563   9.2250
    ## 684   male 14.00000     5     2            CA 2144  46.9000
    ## 685   male 60.00000     1     1              29750  39.0000
    ## 686   male 25.00000     1     2      SC/Paris 2123  41.5792
    ## 687   male 14.00000     4     1            3101295  39.6875
    ## 688   male 19.00000     0     0             349228  10.1708
    ## 689   male 18.00000     0     0             350036   7.7958
    ## 690 female 15.00000     0     1              24160 211.3375
    ## 691   male 31.00000     1     0              17474  57.0000
    ## 692 female  4.00000     0     1             349256  13.4167
    ## 693   male 29.69912     0     0               1601  56.4958
    ## 694   male 25.00000     0     0               2672   7.2250
    ## 695   male 60.00000     0     0             113800  26.5500
    ## 696   male 52.00000     0     0             248731  13.5000
    ## 697   male 44.00000     0     0             363592   8.0500
    ## 698 female 29.69912     0     0              35852   7.7333
    ## 699   male 49.00000     1     1              17421 110.8833
    ## 700   male 42.00000     0     0             348121   7.6500
    ## 701 female 18.00000     1     0           PC 17757 227.5250
    ## 702   male 35.00000     0     0           PC 17475  26.2875
    ## 703 female 18.00000     0     1               2691  14.4542
    ## 704   male 25.00000     0     0              36864   7.7417
    ## 705   male 26.00000     1     0             350025   7.8542
    ## 706   male 39.00000     0     0             250655  26.0000
    ## 707 female 45.00000     0     0             223596  13.5000
    ## 708   male 42.00000     0     0           PC 17476  26.2875
    ## 709 female 22.00000     0     0             113781 151.5500
    ## 710   male 29.69912     1     1               2661  15.2458
    ## 711 female 24.00000     0     0           PC 17482  49.5042
    ## 712   male 29.69912     0     0             113028  26.5500
    ## 713   male 48.00000     1     0              19996  52.0000
    ## 714   male 29.00000     0     0               7545   9.4833
    ## 715   male 52.00000     0     0             250647  13.0000
    ## 716   male 19.00000     0     0             348124   7.6500
    ## 717 female 38.00000     0     0           PC 17757 227.5250
    ## 718 female 27.00000     0     0              34218  10.5000
    ## 719   male 29.69912     0     0              36568  15.5000
    ## 720   male 33.00000     0     0             347062   7.7750
    ## 721 female  6.00000     0     1             248727  33.0000
    ## 722   male 17.00000     1     0             350048   7.0542
    ## 723   male 34.00000     0     0              12233  13.0000
    ## 724   male 50.00000     0     0             250643  13.0000
    ## 725   male 27.00000     1     0             113806  53.1000
    ## 726   male 20.00000     0     0             315094   8.6625
    ## 727 female 30.00000     3     0              31027  21.0000
    ## 728 female 29.69912     0     0              36866   7.7375
    ## 729   male 25.00000     1     0             236853  26.0000
    ## 730 female 25.00000     1     0   STON/O2. 3101271   7.9250
    ## 731 female 29.00000     0     0              24160 211.3375
    ## 732   male 11.00000     0     0               2699  18.7875
    ## 733   male 29.69912     0     0             239855   0.0000
    ## 734   male 23.00000     0     0              28425  13.0000
    ## 735   male 23.00000     0     0             233639  13.0000
    ## 736   male 28.50000     0     0              54636  16.1000
    ## 737 female 48.00000     1     3         W./C. 6608  34.3750
    ## 738   male 35.00000     0     0           PC 17755 512.3292
    ## 739   male 29.69912     0     0             349201   7.8958
    ## 740   male 29.69912     0     0             349218   7.8958
    ## 741   male 29.69912     0     0              16988  30.0000
    ## 742   male 36.00000     1     0              19877  78.8500
    ## 743 female 21.00000     2     2           PC 17608 262.3750
    ## 744   male 24.00000     1     0             376566  16.1000
    ## 745   male 31.00000     0     0  STON/O 2. 3101288   7.9250
    ## 746   male 70.00000     1     1          WE/P 5735  71.0000
    ## 747   male 16.00000     1     1          C.A. 2673  20.2500
    ## 748 female 30.00000     0     0             250648  13.0000
    ## 749   male 19.00000     1     0             113773  53.1000
    ## 750   male 31.00000     0     0             335097   7.7500
    ## 751 female  4.00000     1     1              29103  23.0000
    ## 752   male  6.00000     0     1             392096  12.4750
    ## 753   male 33.00000     0     0             345780   9.5000
    ## 754   male 23.00000     0     0             349204   7.8958
    ## 755 female 48.00000     1     2             220845  65.0000
    ## 756   male  0.67000     1     1             250649  14.5000
    ## 757   male 28.00000     0     0             350042   7.7958
    ## 758   male 18.00000     0     0              29108  11.5000
    ## 759   male 34.00000     0     0             363294   8.0500
    ## 760 female 33.00000     0     0             110152  86.5000
    ## 761   male 29.69912     0     0             358585  14.5000
    ## 762   male 41.00000     0     0   SOTON/O2 3101272   7.1250
    ## 763   male 20.00000     0     0               2663   7.2292
    ## 764 female 36.00000     1     2             113760 120.0000
    ## 765   male 16.00000     0     0             347074   7.7750
    ## 766 female 51.00000     1     0              13502  77.9583
    ## 767   male 29.69912     0     0             112379  39.6000
    ## 768 female 30.50000     0     0             364850   7.7500
    ## 769   male 29.69912     1     0             371110  24.1500
    ## 770   male 32.00000     0     0               8471   8.3625
    ## 771   male 24.00000     0     0             345781   9.5000
    ## 772   male 48.00000     0     0             350047   7.8542
    ## 773 female 57.00000     0     0        S.O./P.P. 3  10.5000
    ## 774   male 29.69912     0     0               2674   7.2250
    ## 775 female 54.00000     1     3              29105  23.0000
    ## 776   male 18.00000     0     0             347078   7.7500
    ## 777   male 29.69912     0     0             383121   7.7500
    ## 778 female  5.00000     0     0             364516  12.4750
    ## 779   male 29.69912     0     0              36865   7.7375
    ## 780 female 43.00000     0     1              24160 211.3375
    ## 781 female 13.00000     0     0               2687   7.2292
    ## 782 female 17.00000     1     0              17474  57.0000
    ## 783   male 29.00000     0     0             113501  30.0000
    ## 784   male 29.69912     1     2         W./C. 6607  23.4500
    ## 785   male 25.00000     0     0 SOTON/O.Q. 3101312   7.0500
    ## 786   male 25.00000     0     0             374887   7.2500
    ## 787 female 18.00000     0     0            3101265   7.4958
    ## 788   male  8.00000     4     1             382652  29.1250
    ## 789   male  1.00000     1     2          C.A. 2315  20.5750
    ## 790   male 46.00000     0     0           PC 17593  79.2000
    ## 791   male 29.69912     0     0              12460   7.7500
    ## 792   male 16.00000     0     0             239865  26.0000
    ## 793 female 29.69912     8     2           CA. 2343  69.5500
    ## 794   male 29.69912     0     0           PC 17600  30.6958
    ## 795   male 25.00000     0     0             349203   7.8958
    ## 796   male 39.00000     0     0              28213  13.0000
    ## 797 female 49.00000     0     0              17465  25.9292
    ## 798 female 31.00000     0     0             349244   8.6833
    ## 799   male 30.00000     0     0               2685   7.2292
    ## 800 female 30.00000     1     1             345773  24.1500
    ## 801   male 34.00000     0     0             250647  13.0000
    ## 802 female 31.00000     1     1         C.A. 31921  26.2500
    ## 803   male 11.00000     1     2             113760 120.0000
    ## 804   male  0.42000     0     1               2625   8.5167
    ## 805   male 27.00000     0     0             347089   6.9750
    ## 806   male 31.00000     0     0             347063   7.7750
    ## 807   male 39.00000     0     0             112050   0.0000
    ## 808 female 18.00000     0     0             347087   7.7750
    ## 809   male 39.00000     0     0             248723  13.0000
    ## 810 female 33.00000     1     0             113806  53.1000
    ## 811   male 26.00000     0     0               3474   7.8875
    ## 812   male 39.00000     0     0          A/4 48871  24.1500
    ## 813   male 35.00000     0     0              28206  10.5000
    ## 814 female  6.00000     4     2             347082  31.2750
    ## 815   male 30.50000     0     0             364499   8.0500
    ## 816   male 29.69912     0     0             112058   0.0000
    ## 817 female 23.00000     0     0   STON/O2. 3101290   7.9250
    ## 818   male 31.00000     1     1    S.C./PARIS 2079  37.0042
    ## 819   male 43.00000     0     0             C 7075   6.4500
    ## 820   male 10.00000     3     2             347088  27.9000
    ## 821 female 52.00000     1     1              12749  93.5000
    ## 822   male 27.00000     0     0             315098   8.6625
    ## 823   male 38.00000     0     0              19972   0.0000
    ## 824 female 27.00000     0     1             392096  12.4750
    ## 825   male  2.00000     4     1            3101295  39.6875
    ## 826   male 29.69912     0     0             368323   6.9500
    ## 827   male 29.69912     0     0               1601  56.4958
    ## 828   male  1.00000     0     2    S.C./PARIS 2079  37.0042
    ## 829   male 29.69912     0     0             367228   7.7500
    ## 830 female 62.00000     0     0             113572  80.0000
    ## 831 female 15.00000     1     0               2659  14.4542
    ## 832   male  0.83000     1     1              29106  18.7500
    ## 833   male 29.69912     0     0               2671   7.2292
    ## 834   male 23.00000     0     0             347468   7.8542
    ## 835   male 18.00000     0     0               2223   8.3000
    ## 836 female 39.00000     1     1           PC 17756  83.1583
    ## 837   male 21.00000     0     0             315097   8.6625
    ## 838   male 29.69912     0     0             392092   8.0500
    ## 839   male 32.00000     0     0               1601  56.4958
    ## 840   male 29.69912     0     0              11774  29.7000
    ## 841   male 20.00000     0     0   SOTON/O2 3101287   7.9250
    ## 842   male 16.00000     0     0        S.O./P.P. 3  10.5000
    ## 843 female 30.00000     0     0             113798  31.0000
    ## 844   male 34.50000     0     0               2683   6.4375
    ## 845   male 17.00000     0     0             315090   8.6625
    ## 846   male 42.00000     0     0          C.A. 5547   7.5500
    ## 847   male 29.69912     8     2           CA. 2343  69.5500
    ## 848   male 35.00000     0     0             349213   7.8958
    ## 849   male 28.00000     0     1             248727  33.0000
    ## 850 female 29.69912     1     0              17453  89.1042
    ## 851   male  4.00000     4     2             347082  31.2750
    ## 852   male 74.00000     0     0             347060   7.7750
    ## 853 female  9.00000     1     1               2678  15.2458
    ## 854 female 16.00000     0     1           PC 17592  39.4000
    ## 855 female 44.00000     1     0             244252  26.0000
    ## 856 female 18.00000     0     1             392091   9.3500
    ## 857 female 45.00000     1     1              36928 164.8667
    ## 858   male 51.00000     0     0             113055  26.5500
    ## 859 female 24.00000     0     3               2666  19.2583
    ## 860   male 29.69912     0     0               2629   7.2292
    ## 861   male 41.00000     2     0             350026  14.1083
    ## 862   male 21.00000     1     0              28134  11.5000
    ## 863 female 48.00000     0     0              17466  25.9292
    ## 864 female 29.69912     8     2           CA. 2343  69.5500
    ## 865   male 24.00000     0     0             233866  13.0000
    ## 866 female 42.00000     0     0             236852  13.0000
    ## 867 female 27.00000     1     0      SC/PARIS 2149  13.8583
    ## 868   male 31.00000     0     0           PC 17590  50.4958
    ## 869   male 29.69912     0     0             345777   9.5000
    ## 870   male  4.00000     1     1             347742  11.1333
    ## 871   male 26.00000     0     0             349248   7.8958
    ## 872 female 47.00000     1     1              11751  52.5542
    ## 873   male 33.00000     0     0                695   5.0000
    ## 874   male 47.00000     0     0             345765   9.0000
    ## 875 female 28.00000     1     0          P/PP 3381  24.0000
    ## 876 female 15.00000     0     0               2667   7.2250
    ## 877   male 20.00000     0     0               7534   9.8458
    ## 878   male 19.00000     0     0             349212   7.8958
    ## 879   male 29.69912     0     0             349217   7.8958
    ## 880 female 56.00000     0     1              11767  83.1583
    ## 881 female 25.00000     0     1             230433  26.0000
    ## 882   male 33.00000     0     0             349257   7.8958
    ## 883 female 22.00000     0     0               7552  10.5167
    ## 884   male 28.00000     0     0   C.A./SOTON 34068  10.5000
    ## 885   male 25.00000     0     0    SOTON/OQ 392076   7.0500
    ## 886 female 39.00000     0     5             382652  29.1250
    ## 887   male 27.00000     0     0             211536  13.0000
    ## 888 female 19.00000     0     0             112053  30.0000
    ## 889 female 29.69912     1     2         W./C. 6607  23.4500
    ## 890   male 26.00000     0     0             111369  30.0000
    ## 891   male 32.00000     0     0             370376   7.7500
    ##               Cabin Embarked
    ## 1              <NA>        S
    ## 2               C85        C
    ## 3              <NA>        S
    ## 4              C123        S
    ## 5              <NA>        S
    ## 6              <NA>        Q
    ## 7               E46        S
    ## 8              <NA>        S
    ## 9              <NA>        S
    ## 10             <NA>        C
    ## 11               G6        S
    ## 12             C103        S
    ## 13             <NA>        S
    ## 14             <NA>        S
    ## 15             <NA>        S
    ## 16             <NA>        S
    ## 17             <NA>        Q
    ## 18             <NA>        S
    ## 19             <NA>        S
    ## 20             <NA>        C
    ## 21             <NA>        S
    ## 22              D56        S
    ## 23             <NA>        Q
    ## 24               A6        S
    ## 25             <NA>        S
    ## 26             <NA>        S
    ## 27             <NA>        C
    ## 28      C23 C25 C27        S
    ## 29             <NA>        Q
    ## 30             <NA>        S
    ## 31             <NA>        C
    ## 32              B78        C
    ## 33             <NA>        Q
    ## 34             <NA>        S
    ## 35             <NA>        C
    ## 36             <NA>        S
    ## 37             <NA>        C
    ## 38             <NA>        S
    ## 39             <NA>        S
    ## 40             <NA>        C
    ## 41             <NA>        S
    ## 42             <NA>        S
    ## 43             <NA>        C
    ## 44             <NA>        C
    ## 45             <NA>        Q
    ## 46             <NA>        S
    ## 47             <NA>        Q
    ## 48             <NA>        Q
    ## 49             <NA>        C
    ## 50             <NA>        S
    ## 51             <NA>        S
    ## 52             <NA>        S
    ## 53              D33        C
    ## 54             <NA>        S
    ## 55              B30        C
    ## 56              C52        S
    ## 57             <NA>        S
    ## 58             <NA>        C
    ## 59             <NA>        S
    ## 60             <NA>        S
    ## 61             <NA>        C
    ## 62              B28     <NA>
    ## 63              C83        S
    ## 64             <NA>        S
    ## 65             <NA>        C
    ## 66             <NA>        C
    ## 67              F33        S
    ## 68             <NA>        S
    ## 69             <NA>        S
    ## 70             <NA>        S
    ## 71             <NA>        S
    ## 72             <NA>        S
    ## 73             <NA>        S
    ## 74             <NA>        C
    ## 75             <NA>        S
    ## 76            F G73        S
    ## 77             <NA>        S
    ## 78             <NA>        S
    ## 79             <NA>        S
    ## 80             <NA>        S
    ## 81             <NA>        S
    ## 82             <NA>        S
    ## 83             <NA>        Q
    ## 84             <NA>        S
    ## 85             <NA>        S
    ## 86             <NA>        S
    ## 87             <NA>        S
    ## 88             <NA>        S
    ## 89      C23 C25 C27        S
    ## 90             <NA>        S
    ## 91             <NA>        S
    ## 92             <NA>        S
    ## 93              E31        S
    ## 94             <NA>        S
    ## 95             <NA>        S
    ## 96             <NA>        S
    ## 97               A5        C
    ## 98          D10 D12        C
    ## 99             <NA>        S
    ## 100            <NA>        S
    ## 101            <NA>        S
    ## 102            <NA>        S
    ## 103             D26        S
    ## 104            <NA>        S
    ## 105            <NA>        S
    ## 106            <NA>        S
    ## 107            <NA>        S
    ## 108            <NA>        S
    ## 109            <NA>        S
    ## 110            <NA>        Q
    ## 111            C110        S
    ## 112            <NA>        C
    ## 113            <NA>        S
    ## 114            <NA>        S
    ## 115            <NA>        C
    ## 116            <NA>        S
    ## 117            <NA>        Q
    ## 118            <NA>        S
    ## 119         B58 B60        C
    ## 120            <NA>        S
    ## 121            <NA>        S
    ## 122            <NA>        S
    ## 123            <NA>        C
    ## 124            E101        S
    ## 125             D26        S
    ## 126            <NA>        C
    ## 127            <NA>        Q
    ## 128            <NA>        S
    ## 129           F E69        C
    ## 130            <NA>        S
    ## 131            <NA>        C
    ## 132            <NA>        S
    ## 133            <NA>        S
    ## 134            <NA>        S
    ## 135            <NA>        S
    ## 136            <NA>        C
    ## 137             D47        S
    ## 138            C123        S
    ## 139            <NA>        S
    ## 140             B86        C
    ## 141            <NA>        C
    ## 142            <NA>        S
    ## 143            <NA>        S
    ## 144            <NA>        Q
    ## 145            <NA>        S
    ## 146            <NA>        S
    ## 147            <NA>        S
    ## 148            <NA>        S
    ## 149              F2        S
    ## 150            <NA>        S
    ## 151            <NA>        S
    ## 152              C2        S
    ## 153            <NA>        S
    ## 154            <NA>        S
    ## 155            <NA>        S
    ## 156            <NA>        C
    ## 157            <NA>        Q
    ## 158            <NA>        S
    ## 159            <NA>        S
    ## 160            <NA>        S
    ## 161            <NA>        S
    ## 162            <NA>        S
    ## 163            <NA>        S
    ## 164            <NA>        S
    ## 165            <NA>        S
    ## 166            <NA>        S
    ## 167             E33        S
    ## 168            <NA>        S
    ## 169            <NA>        S
    ## 170            <NA>        S
    ## 171             B19        S
    ## 172            <NA>        Q
    ## 173            <NA>        S
    ## 174            <NA>        S
    ## 175              A7        C
    ## 176            <NA>        S
    ## 177            <NA>        S
    ## 178             C49        C
    ## 179            <NA>        S
    ## 180            <NA>        S
    ## 181            <NA>        S
    ## 182            <NA>        C
    ## 183            <NA>        S
    ## 184              F4        S
    ## 185            <NA>        S
    ## 186             A32        S
    ## 187            <NA>        Q
    ## 188            <NA>        S
    ## 189            <NA>        Q
    ## 190            <NA>        S
    ## 191            <NA>        S
    ## 192            <NA>        S
    ## 193            <NA>        S
    ## 194              F2        S
    ## 195              B4        C
    ## 196             B80        C
    ## 197            <NA>        Q
    ## 198            <NA>        S
    ## 199            <NA>        Q
    ## 200            <NA>        S
    ## 201            <NA>        S
    ## 202            <NA>        S
    ## 203            <NA>        S
    ## 204            <NA>        C
    ## 205            <NA>        S
    ## 206              G6        S
    ## 207            <NA>        S
    ## 208            <NA>        C
    ## 209            <NA>        Q
    ## 210             A31        C
    ## 211            <NA>        S
    ## 212            <NA>        S
    ## 213            <NA>        S
    ## 214            <NA>        S
    ## 215            <NA>        Q
    ## 216             D36        C
    ## 217            <NA>        S
    ## 218            <NA>        S
    ## 219             D15        C
    ## 220            <NA>        S
    ## 221            <NA>        S
    ## 222            <NA>        S
    ## 223            <NA>        S
    ## 224            <NA>        S
    ## 225             C93        S
    ## 226            <NA>        S
    ## 227            <NA>        S
    ## 228            <NA>        S
    ## 229            <NA>        S
    ## 230            <NA>        S
    ## 231             C83        S
    ## 232            <NA>        S
    ## 233            <NA>        S
    ## 234            <NA>        S
    ## 235            <NA>        S
    ## 236            <NA>        S
    ## 237            <NA>        S
    ## 238            <NA>        S
    ## 239            <NA>        S
    ## 240            <NA>        S
    ## 241            <NA>        C
    ## 242            <NA>        Q
    ## 243            <NA>        S
    ## 244            <NA>        S
    ## 245            <NA>        C
    ## 246             C78        Q
    ## 247            <NA>        S
    ## 248            <NA>        S
    ## 249             D35        S
    ## 250            <NA>        S
    ## 251            <NA>        S
    ## 252              G6        S
    ## 253             C87        S
    ## 254            <NA>        S
    ## 255            <NA>        S
    ## 256            <NA>        C
    ## 257            <NA>        C
    ## 258             B77        S
    ## 259            <NA>        C
    ## 260            <NA>        S
    ## 261            <NA>        Q
    ## 262            <NA>        S
    ## 263             E67        S
    ## 264             B94        S
    ## 265            <NA>        Q
    ## 266            <NA>        S
    ## 267            <NA>        S
    ## 268            <NA>        S
    ## 269            C125        S
    ## 270             C99        S
    ## 271            <NA>        S
    ## 272            <NA>        S
    ## 273            <NA>        S
    ## 274            C118        C
    ## 275            <NA>        Q
    ## 276              D7        S
    ## 277            <NA>        S
    ## 278            <NA>        S
    ## 279            <NA>        Q
    ## 280            <NA>        S
    ## 281            <NA>        Q
    ## 282            <NA>        S
    ## 283            <NA>        S
    ## 284            <NA>        S
    ## 285             A19        S
    ## 286            <NA>        C
    ## 287            <NA>        S
    ## 288            <NA>        S
    ## 289            <NA>        S
    ## 290            <NA>        Q
    ## 291            <NA>        S
    ## 292             B49        C
    ## 293               D        C
    ## 294            <NA>        S
    ## 295            <NA>        S
    ## 296            <NA>        C
    ## 297            <NA>        C
    ## 298         C22 C26        S
    ## 299            C106        S
    ## 300         B58 B60        C
    ## 301            <NA>        Q
    ## 302            <NA>        Q
    ## 303            <NA>        S
    ## 304            E101        Q
    ## 305            <NA>        S
    ## 306         C22 C26        S
    ## 307            <NA>        C
    ## 308             C65        C
    ## 309            <NA>        C
    ## 310             E36        C
    ## 311             C54        C
    ## 312 B57 B59 B63 B66        C
    ## 313            <NA>        S
    ## 314            <NA>        S
    ## 315            <NA>        S
    ## 316            <NA>        S
    ## 317            <NA>        S
    ## 318            <NA>        S
    ## 319              C7        S
    ## 320             E34        C
    ## 321            <NA>        S
    ## 322            <NA>        S
    ## 323            <NA>        Q
    ## 324            <NA>        S
    ## 325            <NA>        S
    ## 326             C32        C
    ## 327            <NA>        S
    ## 328               D        S
    ## 329            <NA>        S
    ## 330             B18        C
    ## 331            <NA>        Q
    ## 332            C124        S
    ## 333             C91        S
    ## 334            <NA>        S
    ## 335            <NA>        S
    ## 336            <NA>        S
    ## 337              C2        S
    ## 338             E40        C
    ## 339            <NA>        S
    ## 340               T        S
    ## 341              F2        S
    ## 342     C23 C25 C27        S
    ## 343            <NA>        S
    ## 344            <NA>        S
    ## 345            <NA>        S
    ## 346             F33        S
    ## 347            <NA>        S
    ## 348            <NA>        S
    ## 349            <NA>        S
    ## 350            <NA>        S
    ## 351            <NA>        S
    ## 352            C128        S
    ## 353            <NA>        C
    ## 354            <NA>        S
    ## 355            <NA>        C
    ## 356            <NA>        S
    ## 357             E33        S
    ## 358            <NA>        S
    ## 359            <NA>        Q
    ## 360            <NA>        Q
    ## 361            <NA>        S
    ## 362            <NA>        C
    ## 363            <NA>        C
    ## 364            <NA>        S
    ## 365            <NA>        Q
    ## 366            <NA>        S
    ## 367             D37        C
    ## 368            <NA>        C
    ## 369            <NA>        Q
    ## 370             B35        C
    ## 371             E50        C
    ## 372            <NA>        S
    ## 373            <NA>        S
    ## 374            <NA>        C
    ## 375            <NA>        S
    ## 376            <NA>        C
    ## 377            <NA>        S
    ## 378             C82        C
    ## 379            <NA>        C
    ## 380            <NA>        S
    ## 381            <NA>        C
    ## 382            <NA>        C
    ## 383            <NA>        S
    ## 384            <NA>        S
    ## 385            <NA>        S
    ## 386            <NA>        S
    ## 387            <NA>        S
    ## 388            <NA>        S
    ## 389            <NA>        Q
    ## 390            <NA>        C
    ## 391         B96 B98        S
    ## 392            <NA>        S
    ## 393            <NA>        S
    ## 394             D36        C
    ## 395              G6        S
    ## 396            <NA>        S
    ## 397            <NA>        S
    ## 398            <NA>        S
    ## 399            <NA>        S
    ## 400            <NA>        S
    ## 401            <NA>        S
    ## 402            <NA>        S
    ## 403            <NA>        S
    ## 404            <NA>        S
    ## 405            <NA>        S
    ## 406            <NA>        S
    ## 407            <NA>        S
    ## 408            <NA>        S
    ## 409            <NA>        S
    ## 410            <NA>        S
    ## 411            <NA>        S
    ## 412            <NA>        Q
    ## 413             C78        Q
    ## 414            <NA>        S
    ## 415            <NA>        S
    ## 416            <NA>        S
    ## 417            <NA>        S
    ## 418            <NA>        S
    ## 419            <NA>        S
    ## 420            <NA>        S
    ## 421            <NA>        C
    ## 422            <NA>        Q
    ## 423            <NA>        S
    ## 424            <NA>        S
    ## 425            <NA>        S
    ## 426            <NA>        S
    ## 427            <NA>        S
    ## 428            <NA>        S
    ## 429            <NA>        Q
    ## 430             E10        S
    ## 431             C52        S
    ## 432            <NA>        S
    ## 433            <NA>        S
    ## 434            <NA>        S
    ## 435             E44        S
    ## 436         B96 B98        S
    ## 437            <NA>        S
    ## 438            <NA>        S
    ## 439     C23 C25 C27        S
    ## 440            <NA>        S
    ## 441            <NA>        S
    ## 442            <NA>        S
    ## 443            <NA>        S
    ## 444            <NA>        S
    ## 445            <NA>        S
    ## 446             A34        S
    ## 447            <NA>        S
    ## 448            <NA>        S
    ## 449            <NA>        C
    ## 450            C104        S
    ## 451            <NA>        S
    ## 452            <NA>        S
    ## 453            C111        C
    ## 454             C92        C
    ## 455            <NA>        S
    ## 456            <NA>        C
    ## 457             E38        S
    ## 458             D21        S
    ## 459            <NA>        S
    ## 460            <NA>        Q
    ## 461             E12        S
    ## 462            <NA>        S
    ## 463             E63        S
    ## 464            <NA>        S
    ## 465            <NA>        S
    ## 466            <NA>        S
    ## 467            <NA>        S
    ## 468            <NA>        S
    ## 469            <NA>        Q
    ## 470            <NA>        C
    ## 471            <NA>        S
    ## 472            <NA>        S
    ## 473            <NA>        S
    ## 474               D        C
    ## 475            <NA>        S
    ## 476             A14        S
    ## 477            <NA>        S
    ## 478            <NA>        S
    ## 479            <NA>        S
    ## 480            <NA>        S
    ## 481            <NA>        S
    ## 482            <NA>        S
    ## 483            <NA>        S
    ## 484            <NA>        S
    ## 485             B49        C
    ## 486            <NA>        S
    ## 487             C93        S
    ## 488             B37        C
    ## 489            <NA>        S
    ## 490            <NA>        S
    ## 491            <NA>        S
    ## 492            <NA>        S
    ## 493             C30        S
    ## 494            <NA>        C
    ## 495            <NA>        S
    ## 496            <NA>        C
    ## 497             D20        C
    ## 498            <NA>        S
    ## 499         C22 C26        S
    ## 500            <NA>        S
    ## 501            <NA>        S
    ## 502            <NA>        Q
    ## 503            <NA>        Q
    ## 504            <NA>        S
    ## 505             B79        S
    ## 506             C65        C
    ## 507            <NA>        S
    ## 508            <NA>        S
    ## 509            <NA>        S
    ## 510            <NA>        S
    ## 511            <NA>        Q
    ## 512            <NA>        S
    ## 513             E25        S
    ## 514            <NA>        C
    ## 515            <NA>        S
    ## 516             D46        S
    ## 517             F33        S
    ## 518            <NA>        Q
    ## 519            <NA>        S
    ## 520            <NA>        S
    ## 521             B73        S
    ## 522            <NA>        S
    ## 523            <NA>        C
    ## 524             B18        C
    ## 525            <NA>        C
    ## 526            <NA>        Q
    ## 527            <NA>        S
    ## 528             C95        S
    ## 529            <NA>        S
    ## 530            <NA>        S
    ## 531            <NA>        S
    ## 532            <NA>        C
    ## 533            <NA>        C
    ## 534            <NA>        C
    ## 535            <NA>        S
    ## 536            <NA>        S
    ## 537             B38        S
    ## 538            <NA>        C
    ## 539            <NA>        S
    ## 540             B39        C
    ## 541             B22        S
    ## 542            <NA>        S
    ## 543            <NA>        S
    ## 544            <NA>        S
    ## 545             C86        C
    ## 546            <NA>        S
    ## 547            <NA>        S
    ## 548            <NA>        C
    ## 549            <NA>        S
    ## 550            <NA>        S
    ## 551             C70        C
    ## 552            <NA>        S
    ## 553            <NA>        Q
    ## 554            <NA>        C
    ## 555            <NA>        S
    ## 556            <NA>        S
    ## 557             A16        C
    ## 558            <NA>        C
    ## 559             E67        S
    ## 560            <NA>        S
    ## 561            <NA>        Q
    ## 562            <NA>        S
    ## 563            <NA>        S
    ## 564            <NA>        S
    ## 565            <NA>        S
    ## 566            <NA>        S
    ## 567            <NA>        S
    ## 568            <NA>        S
    ## 569            <NA>        C
    ## 570            <NA>        S
    ## 571            <NA>        S
    ## 572            C101        S
    ## 573             E25        S
    ## 574            <NA>        Q
    ## 575            <NA>        S
    ## 576            <NA>        S
    ## 577            <NA>        S
    ## 578             E44        S
    ## 579            <NA>        C
    ## 580            <NA>        S
    ## 581            <NA>        S
    ## 582             C68        C
    ## 583            <NA>        S
    ## 584             A10        C
    ## 585            <NA>        C
    ## 586             E68        S
    ## 587            <NA>        S
    ## 588             B41        C
    ## 589            <NA>        S
    ## 590            <NA>        S
    ## 591            <NA>        S
    ## 592             D20        C
    ## 593            <NA>        S
    ## 594            <NA>        Q
    ## 595            <NA>        S
    ## 596            <NA>        S
    ## 597            <NA>        S
    ## 598            <NA>        S
    ## 599            <NA>        C
    ## 600             A20        C
    ## 601            <NA>        S
    ## 602            <NA>        S
    ## 603            <NA>        S
    ## 604            <NA>        S
    ## 605            <NA>        C
    ## 606            <NA>        S
    ## 607            <NA>        S
    ## 608            <NA>        S
    ## 609            <NA>        C
    ## 610            C125        S
    ## 611            <NA>        S
    ## 612            <NA>        S
    ## 613            <NA>        Q
    ## 614            <NA>        Q
    ## 615            <NA>        S
    ## 616            <NA>        S
    ## 617            <NA>        S
    ## 618            <NA>        S
    ## 619              F4        S
    ## 620            <NA>        S
    ## 621            <NA>        C
    ## 622             D19        S
    ## 623            <NA>        C
    ## 624            <NA>        S
    ## 625            <NA>        S
    ## 626             D50        S
    ## 627            <NA>        Q
    ## 628              D9        S
    ## 629            <NA>        S
    ## 630            <NA>        Q
    ## 631             A23        S
    ## 632            <NA>        S
    ## 633             B50        C
    ## 634            <NA>        S
    ## 635            <NA>        S
    ## 636            <NA>        S
    ## 637            <NA>        S
    ## 638            <NA>        S
    ## 639            <NA>        S
    ## 640            <NA>        S
    ## 641            <NA>        S
    ## 642             B35        C
    ## 643            <NA>        S
    ## 644            <NA>        S
    ## 645            <NA>        C
    ## 646             D33        C
    ## 647            <NA>        S
    ## 648             A26        C
    ## 649            <NA>        S
    ## 650            <NA>        S
    ## 651            <NA>        S
    ## 652            <NA>        S
    ## 653            <NA>        S
    ## 654            <NA>        Q
    ## 655            <NA>        Q
    ## 656            <NA>        S
    ## 657            <NA>        S
    ## 658            <NA>        Q
    ## 659            <NA>        S
    ## 660             D48        C
    ## 661            <NA>        S
    ## 662            <NA>        C
    ## 663             E58        S
    ## 664            <NA>        S
    ## 665            <NA>        S
    ## 666            <NA>        S
    ## 667            <NA>        S
    ## 668            <NA>        S
    ## 669            <NA>        S
    ## 670            C126        S
    ## 671            <NA>        S
    ## 672             B71        S
    ## 673            <NA>        S
    ## 674            <NA>        S
    ## 675            <NA>        S
    ## 676            <NA>        S
    ## 677            <NA>        S
    ## 678            <NA>        S
    ## 679            <NA>        S
    ## 680     B51 B53 B55        C
    ## 681            <NA>        Q
    ## 682             D49        C
    ## 683            <NA>        S
    ## 684            <NA>        S
    ## 685            <NA>        S
    ## 686            <NA>        C
    ## 687            <NA>        S
    ## 688            <NA>        S
    ## 689            <NA>        S
    ## 690              B5        S
    ## 691             B20        S
    ## 692            <NA>        C
    ## 693            <NA>        S
    ## 694            <NA>        C
    ## 695            <NA>        S
    ## 696            <NA>        S
    ## 697            <NA>        S
    ## 698            <NA>        Q
    ## 699             C68        C
    ## 700           F G63        S
    ## 701         C62 C64        C
    ## 702             E24        S
    ## 703            <NA>        C
    ## 704            <NA>        Q
    ## 705            <NA>        S
    ## 706            <NA>        S
    ## 707            <NA>        S
    ## 708             E24        S
    ## 709            <NA>        S
    ## 710            <NA>        C
    ## 711             C90        C
    ## 712            C124        S
    ## 713            C126        S
    ## 714            <NA>        S
    ## 715            <NA>        S
    ## 716           F G73        S
    ## 717             C45        C
    ## 718            E101        S
    ## 719            <NA>        Q
    ## 720            <NA>        S
    ## 721            <NA>        S
    ## 722            <NA>        S
    ## 723            <NA>        S
    ## 724            <NA>        S
    ## 725              E8        S
    ## 726            <NA>        S
    ## 727            <NA>        S
    ## 728            <NA>        Q
    ## 729            <NA>        S
    ## 730            <NA>        S
    ## 731              B5        S
    ## 732            <NA>        C
    ## 733            <NA>        S
    ## 734            <NA>        S
    ## 735            <NA>        S
    ## 736            <NA>        S
    ## 737            <NA>        S
    ## 738            B101        C
    ## 739            <NA>        S
    ## 740            <NA>        S
    ## 741             D45        S
    ## 742             C46        S
    ## 743 B57 B59 B63 B66        C
    ## 744            <NA>        S
    ## 745            <NA>        S
    ## 746             B22        S
    ## 747            <NA>        S
    ## 748            <NA>        S
    ## 749             D30        S
    ## 750            <NA>        Q
    ## 751            <NA>        S
    ## 752            E121        S
    ## 753            <NA>        S
    ## 754            <NA>        S
    ## 755            <NA>        S
    ## 756            <NA>        S
    ## 757            <NA>        S
    ## 758            <NA>        S
    ## 759            <NA>        S
    ## 760             B77        S
    ## 761            <NA>        S
    ## 762            <NA>        S
    ## 763            <NA>        C
    ## 764         B96 B98        S
    ## 765            <NA>        S
    ## 766             D11        S
    ## 767            <NA>        C
    ## 768            <NA>        Q
    ## 769            <NA>        Q
    ## 770            <NA>        S
    ## 771            <NA>        S
    ## 772            <NA>        S
    ## 773             E77        S
    ## 774            <NA>        C
    ## 775            <NA>        S
    ## 776            <NA>        S
    ## 777             F38        Q
    ## 778            <NA>        S
    ## 779            <NA>        Q
    ## 780              B3        S
    ## 781            <NA>        C
    ## 782             B20        S
    ## 783              D6        S
    ## 784            <NA>        S
    ## 785            <NA>        S
    ## 786            <NA>        S
    ## 787            <NA>        S
    ## 788            <NA>        Q
    ## 789            <NA>        S
    ## 790         B82 B84        C
    ## 791            <NA>        Q
    ## 792            <NA>        S
    ## 793            <NA>        S
    ## 794            <NA>        C
    ## 795            <NA>        S
    ## 796            <NA>        S
    ## 797             D17        S
    ## 798            <NA>        S
    ## 799            <NA>        C
    ## 800            <NA>        S
    ## 801            <NA>        S
    ## 802            <NA>        S
    ## 803         B96 B98        S
    ## 804            <NA>        C
    ## 805            <NA>        S
    ## 806            <NA>        S
    ## 807             A36        S
    ## 808            <NA>        S
    ## 809            <NA>        S
    ## 810              E8        S
    ## 811            <NA>        S
    ## 812            <NA>        S
    ## 813            <NA>        S
    ## 814            <NA>        S
    ## 815            <NA>        S
    ## 816            B102        S
    ## 817            <NA>        S
    ## 818            <NA>        C
    ## 819            <NA>        S
    ## 820            <NA>        S
    ## 821             B69        S
    ## 822            <NA>        S
    ## 823            <NA>        S
    ## 824            E121        S
    ## 825            <NA>        S
    ## 826            <NA>        Q
    ## 827            <NA>        S
    ## 828            <NA>        C
    ## 829            <NA>        Q
    ## 830             B28     <NA>
    ## 831            <NA>        C
    ## 832            <NA>        S
    ## 833            <NA>        C
    ## 834            <NA>        S
    ## 835            <NA>        S
    ## 836             E49        C
    ## 837            <NA>        S
    ## 838            <NA>        S
    ## 839            <NA>        S
    ## 840             C47        C
    ## 841            <NA>        S
    ## 842            <NA>        S
    ## 843            <NA>        C
    ## 844            <NA>        C
    ## 845            <NA>        S
    ## 846            <NA>        S
    ## 847            <NA>        S
    ## 848            <NA>        C
    ## 849            <NA>        S
    ## 850             C92        C
    ## 851            <NA>        S
    ## 852            <NA>        S
    ## 853            <NA>        C
    ## 854             D28        S
    ## 855            <NA>        S
    ## 856            <NA>        S
    ## 857            <NA>        S
    ## 858             E17        S
    ## 859            <NA>        C
    ## 860            <NA>        C
    ## 861            <NA>        S
    ## 862            <NA>        S
    ## 863             D17        S
    ## 864            <NA>        S
    ## 865            <NA>        S
    ## 866            <NA>        S
    ## 867            <NA>        C
    ## 868             A24        S
    ## 869            <NA>        S
    ## 870            <NA>        S
    ## 871            <NA>        S
    ## 872             D35        S
    ## 873     B51 B53 B55        S
    ## 874            <NA>        S
    ## 875            <NA>        C
    ## 876            <NA>        C
    ## 877            <NA>        S
    ## 878            <NA>        S
    ## 879            <NA>        S
    ## 880             C50        C
    ## 881            <NA>        S
    ## 882            <NA>        S
    ## 883            <NA>        S
    ## 884            <NA>        S
    ## 885            <NA>        S
    ## 886            <NA>        Q
    ## 887            <NA>        S
    ## 888             B42        S
    ## 889            <NA>        S
    ## 890            C148        C
    ## 891            <NA>        Q

``` r
plot(imputation)
```

![](titanic_files/figure-markdown_github/unnamed-chunk-29-1.png)

``` r
stripplot(imputation, pch = 20, cex = 1.2)
```

![](titanic_files/figure-markdown_github/unnamed-chunk-29-2.png) Podemos
combinar <tt>mice</tt> y <tt>caret</tt> para crear modelos de predicción
con varias imputaciones.

``` r
## Imputación resultado
data_raw_imputation_1 <- 
  complete(imputation) %>%
  mutate(Survived = as.factor(ifelse(Survived == 1, 'Yes', 'No'))) %>%
  mutate(Pclass = as.factor(Pclass)) %>%
  mutate(Fare_Interval = as.factor(
    case_when(
      Fare >= 30 ~ 'More.than.30',
      Fare >= 20 & Fare < 30 ~ 'Between.20.30',
      Fare < 20 & Fare >= 10 ~ 'Between.10.20',
      Fare < 10 ~ 'Less.than.10'))) %>%
  select(Survived, Age, Pclass, Sex, Fare_Interval)
  
train   <- data_raw_imputation_1[ trainIndex, ] 
val     <- data_raw_imputation_1[-trainIndex, ]
rPartModel_1 <- train(Survived ~ Age + Pclass + Sex + Fare_Interval, data = data_raw_imputation_1, method = "rpart", metric = "ROC", trControl = rpartCtrl, tuneGrid = rpartParametersGrid)
```

    ## + Fold01: cp=0.01 
    ## - Fold01: cp=0.01 
    ## + Fold02: cp=0.01 
    ## - Fold02: cp=0.01 
    ## + Fold03: cp=0.01 
    ## - Fold03: cp=0.01 
    ## + Fold04: cp=0.01 
    ## - Fold04: cp=0.01 
    ## + Fold05: cp=0.01 
    ## - Fold05: cp=0.01 
    ## + Fold06: cp=0.01 
    ## - Fold06: cp=0.01 
    ## + Fold07: cp=0.01 
    ## - Fold07: cp=0.01 
    ## + Fold08: cp=0.01 
    ## - Fold08: cp=0.01 
    ## + Fold09: cp=0.01 
    ## - Fold09: cp=0.01 
    ## + Fold10: cp=0.01 
    ## - Fold10: cp=0.01 
    ## Aggregating results
    ## Fitting final model on full training set

``` r
## Imputación alternativa 1
data_raw_imputation_2 <- 
  complete(imputation, 2) %>%
  mutate(Survived = as.factor(ifelse(Survived == 1, 'Yes', 'No'))) %>%
  mutate(Pclass = as.factor(Pclass)) %>%
  mutate(Fare_Interval = as.factor(
    case_when(
      Fare >= 30 ~ 'More.than.30',
      Fare >= 20 & Fare < 30 ~ 'Between.20.30',
      Fare < 20 & Fare >= 10 ~ 'Between.10.20',
      Fare < 10 ~ 'Less.than.10'))) %>%
  select(Survived, Age, Pclass, Sex, Fare_Interval)

train   <- data_raw_imputation_2[ trainIndex, ] 
val     <- data_raw_imputation_2[-trainIndex, ]
rPartModel_2 <- train(Survived ~ Age + Pclass + Sex + Fare_Interval, data = data_raw_imputation_2, method = "rpart", metric = "ROC", trControl = rpartCtrl, tuneGrid = rpartParametersGrid)
```

    ## + Fold01: cp=0.01 
    ## - Fold01: cp=0.01 
    ## + Fold02: cp=0.01 
    ## - Fold02: cp=0.01 
    ## + Fold03: cp=0.01 
    ## - Fold03: cp=0.01 
    ## + Fold04: cp=0.01 
    ## - Fold04: cp=0.01 
    ## + Fold05: cp=0.01 
    ## - Fold05: cp=0.01 
    ## + Fold06: cp=0.01 
    ## - Fold06: cp=0.01 
    ## + Fold07: cp=0.01 
    ## - Fold07: cp=0.01 
    ## + Fold08: cp=0.01 
    ## - Fold08: cp=0.01 
    ## + Fold09: cp=0.01 
    ## - Fold09: cp=0.01 
    ## + Fold10: cp=0.01 
    ## - Fold10: cp=0.01 
    ## Aggregating results
    ## Fitting final model on full training set

Y después seleccionar el que mejor ha funcionado, en entrenamiento o
validación. (En este caso no hay diferencias porque los valores
imputados en *Age* son mínimos.)

``` r
# Comparación
prediction_1 <- predict(rPartModel_1, val, type = "raw") 
cm_train_1 <- confusionMatrix(prediction_1, val[["Survived"]])

prediction_2 <- predict(rPartModel_2, val, type = "raw") 
cm_train_2 <- confusionMatrix(prediction_2, val[["Survived"]])
```

<!-- ## Valores con ruido -->
<!-- Para gestionar valores con ruido, utilizamos las herramientas incluidas en [<tt>NoiseFiltersR</tt>](https://cran.r-project.org/web/packages/NoiseFiltersR/index.html). -->
<!-- ```{r} -->
<!-- # Instalar RWeka (install.packages("rJava",type='source')) -->
<!-- library(NoiseFiltersR) -->
<!-- data <- data_raw %>%  -->
<!--   mutate(Survived = as.factor(Survived)) %>% -->
<!--   mutate(Pclass = as.factor(Pclass))   %>% -->
<!--   mutate(Age = as.factor(Age))      %>% -->
<!--   mutate(Sex = as.factor(Sex))      %>% -->
<!--   select(Pclass, Survived, Age, Sex) -->
<!-- noise_filter <- AENN(Survived ~., data) -->
<!-- summary(noise_filter) -->
<!-- identical(noise_filter$cleanData, data[setdiff(1:nrow(data), noise_filter$remIdx), ]) -->
<!-- ``` -->
<!-- <script type="text/javascript"> -->
<!--   <!-- https://stackoverflow.com/questions/39281266/use-internal-links-in-rmarkdown-html-output/39293457 -->
–&gt; <!--   // When the document is fully rendered... -->
<!--   $(document).ready(function() { -->
<!--     // ...select all header elements... -->
<!--     $('h1, h2, h3, h4, h5').each(function() { -->
<!--       // ...and add an id to them corresponding to their 'titles' -->
<!--       $(this).attr('id', $(this).html()); --> <!--     }); -->
<!--   }); --> <!-- </script> -->
