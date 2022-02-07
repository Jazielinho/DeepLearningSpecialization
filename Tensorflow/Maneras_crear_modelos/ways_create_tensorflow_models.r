library(keras)

# CARGANDO LOS DATOS
fashion_mnist <- dataset_fashion_mnist()

imagenes_entrenamiento <- fashion_mnist$train$x
clases_entrenamiento <- fashion_mnist$train$y

imagenes_prueba <- fashion_mnist$test$x
clases_prueba <- fashion_mnist$test$y


class_names <- c('T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

num_classes <- length(class_names)


# ALGUNAS IMAGENES
par(mfcol = c(4, 4))
par(mar = c(0, 0, 3, 0), 
    xaxs = 'i', 
    yaxs = 'i')

for (j in 1:16) { 
  im <- imagenes_entrenamiento[j, , ]
  im <- t(apply(im, 2, rev))
  class_name <- class_names[(clases_entrenamiento[j] + 1)]
  print(class_name)
  image(x = 1:28,
        y = 1:28,
        z = im,
        col = gray((0 :255)/255),
        xaxt = 'n', 
        main = class_name)
}


# PREPARANDO LOS DATOS
imagenes_entrenamiento <- imagenes_entrenamiento / 255.0
imagenes_prueba <- imagenes_prueba / 255.0

input_shape <- c(28, 28, 1)
imagenes_entrenamiento <- array_reshape(imagenes_entrenamiento,
                                        c(nrow(imagenes_entrenamiento),
                                          input_shape))
imagenes_prueba <- array_reshape(imagenes_prueba,
                                 c(nrow(imagenes_prueba),
                                   input_shape))


# FUNCION DE ENTRENAMIENTO
entrena_modelo <- function(modelo){
  modelo %>% compile(
    optimizer = optimizer_adam(),
    loss = loss_sparse_categorical_crossentropy,
    metrics = c(metric_sparse_categorical_accuracy)
  )

  history <- modelo %>% fit(
    imagenes_entrenamiento,
    clases_entrenamiento,
    epochs = 20,
    batch_size = 1024,
    validation_data = list(imagenes_prueba,
                           clases_prueba)
  )

  return(history$history)
}


# SECUENCIAL
retorna_modelo_secuencial <- function(){

  modelo <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32,
                  kernel_size = c(3, 3),
                  activation = 'relu',
                  input_shape = input_shape) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.25) %>%
    layer_conv_2d(filters = 64,
                  kernel_size = c(3, 3),
                  activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.25) %>%
    layer_conv_2d(filters = 128,
                  kernel_size = c(3, 3),
                  activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.4) %>%
    layer_flatten() %>%
    layer_dense(units = 128,
                activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = num_classes,
                activation = 'softmax')

  return(modelo)
}

modelo_secuencial <- retorna_modelo_secuencial()
summary(modelo_secuencial)
historia_secuencial_lista <- entrena_modelo(modelo_secuencial)



# FUNCIONAL
retorna_modelo_funcional <- function(){
  entrada <- layer_input(shape = input_shape)
  x <- layer_conv_2d(filters = 32,
                     kernel_size = c(3, 3),
                     activation = 'relu')(entrada)
  x <- layer_max_pooling_2d(pool_size = c(2, 2))(x)
  x <- layer_dropout(rate = 0.25)(x)
  x <- layer_conv_2d(filters = 64,
                     kernel_size = c(3, 3),
                     activation = 'relu')(x)
  x <- layer_max_pooling_2d(pool_size = c(2, 2))(x)
  x <- layer_dropout(rate = 0.25)(x)
  x <- layer_conv_2d(filters = 128,
                     kernel_size = c(3, 3),
                     activation = 'relu')(x)
  x <- layer_max_pooling_2d(pool_size = c(2, 2))(x)
  x <- layer_dropout(rate = 0.4)(x)
  x <- layer_flatten()(x)
  x <- layer_dense(units = 128,
                   activation = 'relu')(x)
  x <- layer_dropout(rate = 0.3)(x)
  salida <- layer_dense(units = num_classes,
                        activation = 'softmax')(x)
  modelo <- keras_model(inputs = entrada,
                        outputs = salida)
  return(modelo)
}


modelo_funcional <- retorna_modelo_funcional()
summary(modelo_funcional)
historia_funcional <- entrena_modelo(modelo_funcional)


# CLASES
retorna_modelo_clase <- function(){
  keras_model_custom(function(self){
    self$cnn_1 <- layer_conv_2d(filters = 32,
                               kernel_size = c(3, 3),
                               activation = 'relu')
    self$pool_1 <- layer_max_pooling_2d(pool_size = c(2, 2))
    self$dropout_1 <- layer_dropout(rate = 0.25)
    self$cnn_2 <- layer_conv_2d(filters = 64,
                               kernel_size = c(3, 3),
                               activation = 'relu')
    self$pool_2 <- layer_max_pooling_2d(pool_size = c(2, 2))
    self$dropout_2 <- layer_dropout(rate = 0.25)
    self$cnn_3 <- layer_conv_2d(filters = 128,
                               kernel_size = c(3, 3),
                               activation = 'relu')
    self$pool_3 <- layer_max_pooling_2d(pool_size = c(2, 2))
    self$dropout_3 <- layer_dropout(rate = 0.4)
    self$flatten <- layer_flatten()
    self$dense <- layer_dense(units = 128,
                              activation = 'relu')
    self$dropout <- layer_dropout(rate = 0.3)
    self$salida <- layer_dense(units = num_classes,
                                activation = 'softmax')

    function(inputs,
             mask = NULL,
             training = TRUE){
      x <- inputs
      x <- self$cnn_1(x)
      x <- self$pool_1(x)
      x <- self$dropout_1(x)
      x <- self$cnn_2(x)
      x <- self$pool_2(x)
      x <- self$dropout_2(x)
      x <- self$cnn_3(x)
      x <- self$pool_3(x)
      x <- self$dropout_3(x)
      x <- self$flatten(x)
      x <- self$dense(x)
      x <- self$dropout(x)
      x <- self$salida(x)
    }
  }
  )
}


modelo_clase <- retorna_modelo_clase()
historia_clase <- entrena_modelo(modelo_clase)



 # CLASES + LAYERS
cnn_bloque  <- function(filters, 
                        kernel_size, 
                        activation, 
                        pool_size, 
                        rate){
  keras_model_custom(function(self){
    self$cnn = layer_conv_2d(filters = filters, 
                             kernel_size = kernel_size, 
                             activation = activation)
    self$max_pool = layer_max_pooling_2d(pool_size = pool_size)
    self$dropout = layer_dropout(rate = rate)
    
    function(inputs, 
             mask = NULL, 
             traning = TRUE){
      x <- inputs
      x <- self$cnn(x)
      x <- self$max_pool(x)
      x <- self$dropout(x)
      x
    }
  })
}


devuelve_modelo_clase <- function(){
  keras_model_custom(function(self){
    self$cnn_bloque_lista <- c(
      cnn_bloque(filters = 32, 
                 kernel_size = c(3, 3), 
                 activation = 'relu', 
                 pool_size = c(2, 2), 
                 rate = 0.25),
      cnn_bloque(filters = 64, 
                 kernel_size = c(3, 3), 
                 activation = 'relu', 
                 pool_size = c(2, 2), 
                 rate = 0.25),
      cnn_bloque(filters = 128, 
                 kernel_size = c(3, 3), 
                 activation = 'relu', 
                 pool_size = c(2, 2), 
                 rate = 0.4)
    )
    self$flatten <- layer_flatten()
    self$dense <- layer_dense(units = 128, 
                              activation = 'relu')
    self$dropout <- layer_dropout(rate = 0.3)
    self$softmax <- layer_dense(units = num_classes, 
                                activation = 'softmax')
    
    function(inputs, 
             mask = NULL, 
             training = TRUE){
      x <- inputs
      for(indice in 1: length(self$cnn_bloque_lista)){
        layer <- self$cnn_bloque_lista[(indice - 1)]
        x <- layer(x)
      }
      x <- self$flatten(x)
      x <- self$dense(x)
      x <- self$dropout(x)
      x <- self$softmax(x)
      x
    }
  }
  )
}


modelo_clase <- devuelve_modelo_clase()

historia_clase <- entrena_modelo(modelo_clase)




# CNNBloque <- R6::R6Class(
#   "CNNBloque",
#   inherit = KerasLayer,
#   public = list(
#     
#     cnn = NULL,
#     max_pool = NULL,
#     dropout = NULL,
#     
#     initialize = function(filters, kernel_size, activation, pool_size, rate){
#       self$cnn = layer_conv_2d(filters = filters, kernel_size = kernel_size, activation = activation)
#       self$max_pool = layer_max_pooling_2d(pool_size = pool_size)
#       self$dropout = layer_dropout(rate = rate)
#       },
#     
#     call = function(inputs, mask = NULL){
#       x <- inputs
#       x <- self$cnn(x)
#       x <- self$max_pool(x)
#       x <- self$dropout(x)
#       x
#       }
#     )
# )
# 
# 
# cnn_bloque <- function(object, filters, kernel_size, activation, pool_size, rate, name = NULL, trainable = TRUE) {
#   create_layer(CNNBloque, object, list(
#     filters = filters,
#     kernel_size = kernel_size,
#     activation = activation,
#     pool_size = pool_size,
#     rate = rate,
#     name = name,
#     trainable = trainable
#   ))
# }
# 
# 
# devuelve_modelo_clase <- function(){
#   keras_model_custom(function(self){
#     # self$cnn_1 <- cnn_bloque(filters = 32, kernel_size = c(3, 3), activation = 'relu', pool_size = c(2, 2), rate = 0.25)
#     # self$cnn_2 <- cnn_bloque(filters = 64, kernel_size = c(3, 3), activation = 'relu', pool_size = c(2, 2), rate = 0.25)
#     # self$cnn_3 <- cnn_bloque(filters = 128, kernel_size = c(3, 3), activation = 'relu', pool_size = c(2, 2), rate = 0.4)
#     
#     self$cnn_bloque_lista <- c(
#       cnn_bloque(filters = 32, kernel_size = c(3, 3), activation = 'relu', pool_size = c(2, 2), rate = 0.25),
#       cnn_bloque(filters = 64, kernel_size = c(3, 3), activation = 'relu', pool_size = c(2, 2), rate = 0.25),
#       cnn_bloque(filters = 128, kernel_size = c(3, 3), activation = 'relu', pool_size = c(2, 2), rate = 0.4)
#     )
#     self$flatten <- layer_flatten()
#     self$dense <- layer_dense(units = 128, activation = 'relu')
#     self$dropout <- layer_dropout(rate = 0.3)
#     self$softmax <- layer_dense(units = num_classes, activation = 'softmax')
#     
#     function(inputs, mask = NULL, training = TRUE){
#       x <- inputs
#       # x <- self$cnn_1(x)
#       # x <- self$cnn_2(x)
#       # x <- self$cnn_3(x)
#       for(indice in 1: length(self$cnn_bloque_lista)){
#         layer <- self$cnn_bloque_lista[(indice - 1)]
#         x <- layer(x)
#       }
#       x <- self$flatten(x)
#       x <- self$dense(x)
#       x <- self$dropout(x)
#       x <- self$softmax(x)
#       x
#     }
#   }
#   )
# }

