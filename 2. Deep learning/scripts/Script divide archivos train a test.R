# FUNCIÓN PARA CLASIFICAR IMÁGENES SEGÚN SU ETIQUETA EN DIRECTORIOS

# --> Dado un directorio etiquetado lleno de imágenes, distribuye un porcentaje de éstas (para conjunto test y/o validación) en otro directorio.
#   IMPORTANTE: Crear antes el directorio y los subdirectorios de etiquetas donde se va a ubicar dichas imágenes.

separaTrainTest <- function(carpeta_train, carpeta_test, porcentaje = 0.2) {
  clases<-list.dirs(path = carpeta_train, full.names = FALSE)

  for (clase in clases){
    if(clase != "") {
      carpeta_clase_train <- paste(carpeta_train,clase,sep = "/")
      carpeta_clase_test <- paste(carpeta_test,clase,sep = "/")

      todos <- list.files(path = carpeta_clase_train)
      a_copiar <- sample(todos, length(todos)*porcentaje)

      for (fichero in a_copiar){
        file.copy(paste(carpeta_clase_train, fichero, sep = "/"), carpeta_clase_test)
        file.remove(paste(carpeta_clase_train, fichero, sep = "/"))
      }
    }
  }
}

# Ejecuta la función
# param 1: Directorio origen
# param 2: Directorio destino (previamente creado con sus subdirectorios de etiquetas)
separaTrainTest("./data/train_images", "./data/test_images")
