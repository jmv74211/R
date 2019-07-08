## -------------------------------------------------------------------------------------
## Jonathan Martín Valera
## Práctica 2: Deep Learning para multi-clasificación
## -------------------------------------------------------------------------------------


# Se cargan las bibliotecas necesarias para deeplearning
library(keras)

# Se definen las rutas donde se ubican las imágenes de entrenamiento, validación y test
train_dir      <- '/home/rstudio-user/data/train_images'
validation_dir <- '/home/rstudio-user/data/validation_images'
test_dir       <- '/home/rstudio-user/data/test_images'

# Se definen los generadores de la imagen, haciendo un reescalado 1/255
train_datagen      <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen       <- image_data_generator(rescale = 1/255)

# Se definen las imágenes de entrenamiento a través de un directorio clasificado por etiquetas
train_data <- flow_images_from_directory(
  directory = train_dir,
  generator = train_datagen,
  target_size = c(150, 150),     # (w, h) --> (150, 150)
  batch_size = 100,              # grupos de 100 imágenes
  class_mode = "categorical"     # etiquetas multiclase
)

# Se definen las imágenes de validación través de un directorio clasificado por etiquetas
validation_data <- flow_images_from_directory(
  directory = validation_dir,
  generator = validation_datagen,
  target_size = c(150, 150),     # (w, h) --> (150, 150)
  batch_size = 100,              # grupos de 100 imágenes
  class_mode = "categorical"     # etiquetas multiclase
)

# Se definen las imágenes de test través de un directorio clasificado por etiquetas
test_data <- flow_images_from_directory(
  directory = test_dir,
  generator = test_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 100,            # grupos de 100 imágenes
  class_mode = "categorical"   # etiquetas multiclase
)

# Se define la dimensión de los datos
input_shape_images <- c(150,150,3)
# Se especifica el número de clases de este problema.
num_classes <- 5

################################################ MODEL V1 ################################################

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32,  kernel_size = c(3, 3), activation = "relu", input_shape = input_shape_images) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64,  kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)


history <- model %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate <- model %>% evaluate_generator(test_data, steps = 5)
print(test_rate)

model %>% save_model_hdf5("modelv1.h5")

################################################ MODEL V2 ################################################

model_v2 <- keras_model_sequential() %>%
  layer_separable_conv_2d(filters = 32, kernel_size = 3,
                          activation = "relu",
                          input_shape = input_shape_images) %>%
  layer_separable_conv_2d(filters = 64, kernel_size = 3,
                          activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_separable_conv_2d(filters = 64, kernel_size = 3,
                          activation = "relu") %>%
  layer_separable_conv_2d(filters = 128, kernel_size = 3,
                          activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_separable_conv_2d(filters = 64, kernel_size = 3,
                          activation = "relu") %>%
  layer_separable_conv_2d(filters = 128, kernel_size = 3,
                          activation = "relu") %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v2)

model_v2 %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history_v2 <- model_v2 %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v2 <- model_v2 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v2)

model_v2 %>% save_model_hdf5("modelv2.h5")


################################################ MODEL V3 ################################################

model_v3 <- keras_model_sequential() %>%
  layer_separable_conv_2d(filters = 32, kernel_size = 3,
                          activation = "relu",
                          input_shape = input_shape_images) %>%
  layer_separable_conv_2d(filters = 64, kernel_size = 3,
                          activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_separable_conv_2d(filters = 64, kernel_size = 3,
                          activation = "relu") %>%
  layer_separable_conv_2d(filters = 128, kernel_size = 3,
                          activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_separable_conv_2d(filters = 64, kernel_size = 3,
                          activation = "relu") %>%
  layer_separable_conv_2d(filters = 128, kernel_size = 3,
                          activation = "relu") %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v3)

model_v3 %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history_v3 <- model_v3 %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v3 <- model_v3 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v3)

#model_v3 %>% save_model_hdf5("modelv3.h5")

################################################ MODEL V4 ################################################

model_v4 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32,  kernel_size = c(5, 5), activation = "relu", input_shape = input_shape_images) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64,  kernel_size = c(5, 5), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(5, 5), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 256, kernel_size = c(5, 5), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v4)

model_v4 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)


history <- model_v4 %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v4 <- model_v4 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v4)

model_v4 %>% save_model_hdf5("modelv4.h5")

################################################ MODEL V5 ################################################


model_v5 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32,  kernel_size = c(3, 3), activation = "relu", input_shape = input_shape_images) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64,  kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 512, kernel_size = c(5, 5), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v5)

model_v5 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)


history <- model_v5%>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v5 <- model_v5 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v5)

model_v5 %>% save_model_hdf5("modelv5.h5")

################################################ MODEL V6 ################################################


conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# 2. Congelar pesos de la capa convolutiva VGG16 y compilar
freeze_weights(conv_base)

model_v6 <- keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v6)

model_v6 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)


history <- model_v6%>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v6 <- model_v6 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v6)

model_v6 %>% save_model_hdf5("modelv6.h5")

################################################ MODEL V7 ################################################

conv_base_v7 <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)


model_v7 <- keras_model_sequential() %>%
  conv_base_v7 %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v7)

model_v7 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)


history <- model_v7%>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v7 <- model_v7 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v7)

model_v7 %>% save_model_hdf5("modelv7.h5")

################################################ MODEL V8 ################################################

conv_base_v8 <- application_inception_v3(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# 2. Congelar pesos de la capa convolutiva
freeze_weights(conv_base_v8)

model_v8 <- keras_model_sequential() %>%
  conv_base_v8 %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v8)

model_v8 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)


history <- model_v8%>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v8 <- model_v8 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v8)

model_v8 %>% save_model_hdf5("modelv8.h5")

################################################ MODEL V9 ################################################

conv_base_v9 <- application_vgg19(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# 2. Congelar pesos de la capa convolutiva
freeze_weights(conv_base_v9)

model_v9 <- keras_model_sequential() %>%
  conv_base_v9 %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v9)

model_v9 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)


history <- model_v9%>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v9 <- model_v9 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v9)

model_v9 %>% save_model_hdf5("modelv9.h5")

################################################ MODEL V10 ################################################

conv_base_v10 <- application_inception_resnet_v2(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# 2. Congelar pesos de la capa convolutiva
freeze_weights(conv_base_v10)

model_v10 <- keras_model_sequential() %>%
  conv_base_v10 %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v10)

model_v10 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)

history <- model_v10%>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v10 <- model_v10 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v10)

model_v10 %>% save_model_hdf5("modelv10.h5")


################################################ MODEL V11 ################################################

# DATA AUGMENTATION con el mejor modelo sin trasnfer learning, en este caso el modelo 1

data_augmentation_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

train_augmented_data <- flow_images_from_directory(
  directory = train_dir,
  generator = data_augmentation_datagen,  # ¡usando nuevo datagen!
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode = "categorical"        # etiquetas multiclase
)

history <- model %>% 
  fit_generator(
    train_augmented_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  )

test_rate_v11 <- model %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v11)

model %>% save_model_hdf5("modelv11.h5")

################################################ MODEL V12 ################################################

# DATA AUGMENTATION con el mejor modelo de transfer learning, es decir el modelo v6

data_augmentation_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

train_augmented_data <- flow_images_from_directory(
  directory = train_dir,
  generator = data_augmentation_datagen,  # ¡usando nuevo datagen!
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode = "categorical"        # etiquetas multiclase
)

history <- model_v6 %>% 
  fit_generator(
    train_augmented_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  )

test_rate_v12 <- model %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v12)

model_v6 %>% save_model_hdf5("modelv12.h5")

################################################ MODEL V13 ################################################

# Mejor modelo (v6) aplicando algoritmo de optimización SGD.

conv_base_v13 <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# 2. Congelar pesos de la capa convolutiva VGG16 y compilar
freeze_weights(conv_base_v13)

model_v13 <- keras_model_sequential() %>%
  conv_base_v13 %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v13)

model_v13 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "sgd",
  metrics = c("accuracy")
)


history <- model_v13%>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v13 <- model_v13 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v13)

model_v13 %>% save_model_hdf5("modelv13.h5")

################################################ MODEL V14 ################################################

# Mejor modelo (v6) aplicando algoritmo de optimización rmsprop.

conv_base_v14 <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# 2. Congelar pesos de la capa convolutiva VGG16 y compilar
freeze_weights(conv_base_v14)

model_v14 <- keras_model_sequential() %>%
  conv_base_v14 %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v14)

model_v14 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "rmsprop",
  metrics = c("accuracy")
)


history <- model_v14%>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v14 <- model_v14 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v14)

model_v14 %>% save_model_hdf5("modelv14.h5")

################################################ MODEL V15 ################################################

# Mejor modelo (v6) aplicando algoritmo de optimización adagrad.

conv_base_v15 <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# 2. Congelar pesos de la capa convolutiva VGG16 y compilar
freeze_weights(conv_base_v15)

model_v15 <- keras_model_sequential() %>%
  conv_base_v15 %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v15)

model_v15%>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adagrad",
  metrics = c("accuracy")
)

history <- model_v15%>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v15 <- model_v15 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v15)

model_v15 %>% save_model_hdf5("modelv15.h5")

################################################ MODEL V16 ################################################

# Mejor modelo (v6) aplicando algoritmo de optimización adadelta.

conv_base_v16 <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# 2. Congelar pesos de la capa convolutiva VGG16 y compilar
freeze_weights(conv_base_v16)

model_v16 <- keras_model_sequential() %>%
  conv_base_v16 %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v16)

model_v16 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adadelta",
  metrics = c("accuracy")
)

history <- model_v16 %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v16 <- model_v16 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v16)

model_v16 %>% save_model_hdf5("modelv16.h5")

################################################ MODEL V17 ################################################

# Mejor modelo (v6) aplicando algoritmo de optimización adamax.

conv_base_v17 <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# 2. Congelar pesos de la capa convolutiva VGG16 y compilar
freeze_weights(conv_base_v17)

model_v17 <- keras_model_sequential() %>%
  conv_base_v17 %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v17)

model_v17 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adamax",
  metrics = c("accuracy")
)

history <- model_v17 %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v17 <- model_v17 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v17)

model_v17 %>% save_model_hdf5("modelv17.h5")

################################################ MODEL V18 ################################################

# Mejor modelo (v6) aplicando algoritmo de optimización nadam.

conv_base_v18 <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# 2. Congelar pesos de la capa convolutiva VGG16 y compilar
freeze_weights(conv_base_v18)

model_v18 <- keras_model_sequential() %>%
  conv_base_v18 %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v18)

model_v18 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "nadam",
  metrics = c("accuracy")
)

history <- model_v18 %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v18 <- model_v18 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v18)

model_v18 %>% save_model_hdf5("modelv18.h5")

################################################ MODEL V19 ################################################

# Mejor modelo (v17) con 15 iteraciones para aplicar early stopping

conv_base_v19 <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# 2. Congelar pesos de la capa convolutiva VGG16 y compilar
freeze_weights(conv_base_v19)

model_v19 <- keras_model_sequential() %>%
  conv_base_v19 %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v19)

model_v19 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adamax",
  metrics = c("accuracy")
)

history <- model_v19 %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 15,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v19 <- model_v19 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v19)

model_v19 %>% save_model_hdf5("modelv19.h5")

################################################ MODEL V20 ################################################

# Mejor modelo (v17) con 2 iteraciones, tras haber observado de que a partir de la iteración 2 el error va en aumento

conv_base_v20 <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# 2. Congelar pesos de la capa convolutiva VGG16 y compilar
freeze_weights(conv_base_v20)

model_v20 <- keras_model_sequential() %>%
  conv_base_v20 %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v20)

model_v20 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adamax",
  metrics = c("accuracy")
)

history <- model_v20 %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 2,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v20 <- model_v20 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v20)

model_v20 %>% save_model_hdf5("modelv20.h5")

################################################ MODEL V21 ################################################

# Mejor modelo (v17) aplicando dropout en la red profunda

conv_base_v21 <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# 2. Congelar pesos de la capa convolutiva VGG16 y compilar
freeze_weights(conv_base_v21)

model_v21 <- keras_model_sequential() %>%
  conv_base_v21 %>%
  layer_flatten() %>%
  layer_dropout(rate=0.4) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v21)

model_v21 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adamax",
  metrics = c("accuracy")
)

history <- model_v21 %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v21 <- model_v21 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v21)

model_v21 %>% save_model_hdf5("modelv21.h5")

################################################ MODEL V22 ################################################

# Mejor modelo (v17) aplicando dropout en la red profunda

conv_base_v22 <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# 2. Congelar pesos de la capa convolutiva VGG16 y compilar
freeze_weights(conv_base_v22)

model_v22 <- keras_model_sequential() %>%
  conv_base_v22 %>%
  layer_flatten() %>%
  layer_dropout(rate=0.4) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate=0.5) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

summary(model_v22)

model_v22 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adamax",
  metrics = c("accuracy")
)

history <- model_v22 %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 50
  ) 

test_rate_v22 <- model_v22 %>% evaluate_generator(test_data, steps = 5)
print(test_rate_v22)

model_v22 %>% save_model_hdf5("modelv22.h5")
