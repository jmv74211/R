library("ggplot2")

"EJEMPLO DE GRÁFICOS CON GGPLOT"

###################################################################

#                               SGD

sgd_data_train_acc <- as.data.frame(c(0.2893, 0.3064, 0.3176, 0.3216, 0.3262))
colnames(sgd_data_train_acc) <-  c("sgd_data_train_acc")

sgd_data_train_loss<- as.data.frame(c(1.4580, 1.4358, 1.4286, 1.4248, 1.4190))
colnames(sgd_data_train_loss) <-  c("sgd_data_train_loss")

sgd_data_validation_acc <- as.data.frame(c(0.3000, 0.3102, 0.3124, 0.3096, 0.2969))
colnames(sgd_data_validation_acc) <-  c("sgd_data_validation_acc")

sgd_data_validation_loss<- as.data.frame(c(1.4419, 1.4379, 1.4373, 1.4346, 1.4545))
colnames(sgd_data_validation_loss) <-  c("sgd_data_validation_loss")

sgd_data_test_acc <- as.data.frame(c(0.285))
colnames(sgd_data_test_acc ) <-  c("sgd_data_test_acc")

sgd_data_test_loss <- as.data.frame(c(1.458))
colnames(sgd_data_test_loss) <-  c("sgd_data_test_loss")


####################################################################

###################################################################

#                               RMSPROP

rmsprop_data_train_acc <- as.data.frame(c(0.2777, 0.3221, 0.3482, 0.3689, 0.3860))
colnames(rmsprop_data_train_acc) <-  c("rmsprop_data_train_acc")

rmsprop_data_train_loss<- as.data.frame(c(2.2659, 1.4260, 1.3978, 1.3764, 1.3532 ), col.names = c('rmsprop_data_train_loss'))
colnames(rmsprop_data_train_loss) <-  c("rmsprop_data_train_loss")

rmsprop_data_validation_acc <- as.data.frame(c(0.3163, 0.3247, 0.3168, 0.3170, 0.3195))
colnames(rmsprop_data_validation_acc) <-  c("rmsprop_data_validation_acc")

rmsprop_data_validation_loss<- as.data.frame(c(1.4325, 1.4249, 1.4496, 1.4282, 1.4422))
colnames(rmsprop_data_validation_loss) <-  c("rmsprop_data_validation_loss")

rmsprop_data_test_acc <- as.data.frame(c(0.3318))
colnames(rmsprop_data_test_acc ) <-  c("rmsprop_data_test_acc")

rmsprop_data_test_loss <- as.data.frame(c(1.446))
colnames(rmsprop_data_test_loss) <-  c("rmsprop_data_test_loss")

####################################################################

###################################################################

#                               ADAGRAD

adagrad_data_train_acc <- as.data.frame(c(0.2813, 0.2819, 0.2819, 0.2819, 0.2919))
colnames(adagrad_data_train_acc) <-  c("adagrad_data_train_acc")

adagrad_data_train_loss<- as.data.frame(c(11.5537, 11.5707, 11.5757, 11.5771, 11.5714))
colnames(adagrad_data_train_loss) <-  c("adagrad_data_train_loss")

adagrad_data_validation_acc <- as.data.frame(c(0.2819, 0.2819, 0.2819, 0.2819, 0.2819))
colnames(adagrad_data_validation_acc) <-  c("adagrad_data_validation_acc")

adagrad_data_validation_loss<- as.data.frame(c(11.5752, 11.5729, 11.5752, 11.5752, 11.5939))
colnames(adagrad_data_validation_loss) <-  c("adagrad_data_validation_loss")

adagrad_data_test_acc <- as.data.frame(c(0.281))
colnames(adagrad_data_test_acc ) <-  c("adagrad_data_test_acc")

adagrad_data_test_loss <- as.data.frame(c(11.58))
colnames(adagrad_data_test_loss) <-  c("adagrad_data_test_loss")

####################################################################

###################################################################

#                               ADADELTA

adadelta_data_train_acc <- as.data.frame(c(0.2955, 0.3304, 0.3428, 0.3561, 0.3707))
colnames(adadelta_data_train_acc) <-  c("adadelta_data_train_acc")

adadelta_data_train_loss<- as.data.frame(c(1.4714, 1.4205, 1.4040, 1.3894, 1.3721))
colnames(adadelta_data_train_loss) <-  c("adadelta_data_train_loss")

adadelta_data_validation_acc <- as.data.frame(c(0.2913, 0.3238, 0.3175, 0.3203, 0.3226))
colnames(adadelta_data_validation_acc) <-  c("adadelta_data_validation_acc")

adadelta_data_validation_loss<- as.data.frame(c(1.4485, 1.4242, 1.4378, 1.4278, 1.4325))
colnames(adadelta_data_validation_loss) <-  c("adadelta_data_validation_loss")

adadelta_data_test_acc <- as.data.frame(c(0.314))
colnames(adadelta_data_test_acc ) <-  c("adadelta_data_test_acc")

adadelta_data_test_loss <- as.data.frame(c(1.437))
colnames(adadelta_data_test_loss) <-  c("adadelta_data_test_loss")

####################################################################

###################################################################

#                               ADAMAX

adamax_data_train_acc <- as.data.frame(c(0.3058, 0.3396, 0.3589, 0.3745, 0.3901))
colnames(adamax_data_train_acc) <-  c("adamax_data_train_acc")

adamax_data_train_loss<- as.data.frame(c(1.4675, 1.4072, 1.3864, 1.3674, 1.3445))
colnames(adamax_data_train_loss) <-  c("adamax_data_train_loss")


adamax_data_validation_acc <- as.data.frame(c(0.3082, 0.3043, 0.3152, 0.3115, 0.3259))
colnames(adamax_data_validation_acc) <-  c("adamax_data_validation_acc")

adamax_data_validation_loss<- as.data.frame(c(1.4333, 1.4597, 1.4361, 1.4365, 1.4417))
colnames(adamax_data_validation_loss) <-  c("adamax_data_validation_loss")

adamax_data_test_acc <- as.data.frame(c(0.323))
colnames(adamax_data_test_acc ) <-  c("adamax_data_test_acc")

adamax_data_test_loss <- as.data.frame(c(1.445))
colnames(adamax_data_test_loss) <-  c("adamax_data_test_loss")

####################################################################


###################################################################

#                               ADAM

adam_data_train_acc <- as.data.frame(c(0.2988, 0.3370, 0.3546, 0.3701, 0.3867))
colnames(adam_data_train_acc) <-  c("adam_data_train_acc")

adam_data_train_loss<- as.data.frame(c(1.4625, 1.4104, 1.3884, 1.3671, 1.3439))
colnames(adam_data_train_loss) <-  c("adam_data_train_loss")

adam_data_validation_acc <- as.data.frame(c(0.3201, 0.3156, 0.3097, 0.3206, 0.3159))
colnames(adam_data_validation_acc) <-  c("adam_data_validation_acc")

adam_data_validation_loss<- as.data.frame(c(1.4291, 1.4287, 1.4349, 1.4384, 1.4578))
colnames(adam_data_validation_loss) <-  c("adam_data_validation_loss")


adam_data_test_acc <- as.data.frame(c(0.310))
colnames(adam_data_test_acc ) <-  c("adam_data_test_acc")

adam_data_test_loss <- as.data.frame(c(1.461))
colnames(adam_data_test_loss) <-  c("adam_data_test_loss")

####################################################################

###################################################################

#                               NADAM

nadam_data_train_acc <- as.data.frame(c(0.3029, 0.3387, 0.3633, 0.3847, 0.4035))
colnames(nadam_data_train_acc) <-  c("nadam_data_train_acc")

nadam_data_train_loss<- as.data.frame(c(1.4973, 1.4077, 1.3795, 1.3518, 1.3234))
colnames(nadam_data_train_loss) <-  c("nadam_data_train_loss")

nadam_data_validation_acc <- as.data.frame(c(0.3207, 0.3256, 0.3241, 0.3199, 0.3140))
colnames(nadam_data_validation_acc) <-  c("nadam_data_validation_acc")

nadam_data_validation_loss<- as.data.frame(c(1.4307, 1.4228, 1.4265, 1.4395, 1.4726))
colnames(nadam_data_validation_loss) <-  c("nadam_data_validation_loss")

nadam_data_test_acc <- as.data.frame(c(0.307))
colnames(nadam_data_test_acc ) <-  c("nadam_data_test_acc")

nadam_data_test_loss <- as.data.frame(c(1.475))
colnames(nadam_data_test_loss) <-  c("nadam_data_test_loss")


####################################################################

# GRÁFICA LINEAL

ggplot() + 
  geom_line(data = sgd_data_train_acc,       aes(x= c(1:5),  y=sgd_data_train_acc, colour="SGD"))+
  geom_line(data = rmsprop_data_train_acc  , aes(x=,c(1:5),  y=rmsprop_data_train_acc, colour="RMSPROP"))+
  geom_line(data = adagrad_data_train_acc,       aes(x=,c(1:5),  y=adagrad_data_train_acc, colour="ADAGRAD"))+
  geom_line(data = adadelta_data_train_acc,       aes(x=,c(1:5),  y=adadelta_data_train_acc, colour="ADADELTA"))+
  geom_line(data = adamax_data_train_acc,       aes(x=,c(1:5),  y=adamax_data_train_acc, colour="ADAMAX"))+
  geom_line(data = adam_data_train_acc,       aes(x=,c(1:5),  y=adam_data_train_acc, colour="ADAM"))+
  geom_line(data = nadam_data_train_acc,       aes(x=,c(1:5),  y=nadam_data_train_acc, colour="NADAM"))+
  labs(x = "Épocas",y = "Acc")+
  scale_colour_manual("", 
                      breaks = c("SGD", "RMSPROP", "ADAGRAD","ADADELTA", "ADAMAX","ADAM","NADAM"),
                      values = c("SGD" ="red", "RMSPROP"="green", "ADAGRAD" ="blue","ADADELTA"="yellow", "ADAMAX"="purple",
                                 "ADAM"="orange", "NADAM"="brown"))



###################################################################################################################

# GRÁFICO DE BARRAS


data_test_acc<- data.frame(optimizer = c("SGD", "RMSPROP", "ADADELTA","ADAMAX","ADAM","NADAM"), 
                           acc = c(0.285, 0.318, 0.314, 0.323, 0.310, 0.307 ))


data_test_loss<- data.frame(optimizer = c("SGD", "RMSPROP", "ADADELTA","ADAMAX","ADAM","NADAM"), 
                           loss = c(1.458, 1.446, 1.437, 1.445,1.461 ,1.475 ))

ggplot() + 
  geom_bar( data = data_test_acc, aes(x=optimizer, y=acc, fill=optimizer), stat="identity")+ # Para realizar el gŕafico de barras
  coord_cartesian(ylim = c(0.27, 0.33))+
  geom_text(data = data_test_acc, aes(x=optimizer, y=acc, label=acc), vjust=-0.3, size=3.5)+ # Para mostrar número encima de las barras
  theme_minimal() # Para mostrar número encima de las barras


###################################################################################################################


