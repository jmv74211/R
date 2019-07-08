## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2018-2019
## Juan Gómez Romero
## -------------------------------------------------------------------------------------

## -------------------------------------------------------------------------------------
## LIBRARY
library(filesstrings)

## -------------------------------------------------------------------------------------
## CONSTANTES
dataset_dir      <- './petfinder-adoption-prediction/'
train_images_dir <- paste0(dataset_dir, 'train_images/')
train_data_file  <- paste0(dataset_dir, 'train.csv')
test_images_dir  <- paste0(dataset_dir, 'test_images/')
test_data_file   <- paste0(dataset_dir, 'test.csv')

## -------------------------------------------------------------------------------------
## FUNCIONES
make_dirs_from_train_classes <- function(csv_file, img_folder) {
  data <- read_csv(csv_file)
  
  # crear carpetas por cada clase
  for(class in unique(data$AdoptionSpeed)) {
    new_dir <- paste0(img_folder, class)
    
    if(!dir.exists(new_dir)) {
      dir.create(new_dir)
    }
  }
  
  # mover ficheros
  for(i in seq_len(nrow(data))) {
    pet <- data[i,]
    
    for(j in 1:pet$PhotoAmt) {
      file_source <- paste0(img_folder, pet$PetID, "-", j, ".jpg")
      dir_dest    <- paste0(img_folder, pet$AdoptionSpeed)
      
      if(file.exists(file_source)) {
        file.move(file_source, dir_dest, overwrite = TRUE)
      }
    }
  }
}

make_dir_for_test <- function(img_folder) {
  new_dir <- paste0(img_folder, 'unknown')
  files   <- paste0(img_folder, list.files(img_folder))
  dir.create(new_dir)
  move_files(files, new_dir, overwrite = TRUE)
}

## -------------------------------------------------------------------------------------

make_dirs_from_train_classes(train_data_file, train_images_dir)
make_dir_for_test(test_images_dir)


