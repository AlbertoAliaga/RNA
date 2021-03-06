library(RSNNS)

graficaError <- function(iterativeErrors){
        plot(1:nrow(iterativeErrors),iterativeErrors[,1], type="l", main="Evoluci�n del error",
             ylab="MSE",xlab="Ciclos",
             ylim=c(min(iterativeErrors),max(iterativeErrors)))
        lines(1:nrow(iterativeErrors),iterativeErrors[,2], col="red")
}


## funcion que calcula el error cuadratico medio
MSE <- function(pred,obs) {sum((pred-obs)^2)/length(obs)}


#CARGA DE DATOS
# se supone que los ficheros tienen encabezados
# si no los tienen, cambiar header a F
 trainSet <- read.csv("train.csv",dec=".",sep=",",header = T)
 validSet <- read.csv( "Valid.csv",dec=".",sep=",",header = T)
 testSet  <- read.csv("Test.csv",dec=".",sep=",",header = T)



salida <- ncol (trainSet)   #num de la columna de salida





#SELECCION DE LOS PARAMETROS
topologia        <- c(60) #PARAMETRO DEL TIPO c(A,B,C,...,X) A SIENDO LAS NEURONAS EN LA CAPA OCULTA 1, B LA CAPA 2 ...
razonAprendizaje <- 0.2 #NUMERO REAL ENTRE 0 y 1
ciclosMaximos    <- 2000 #NUMERO ENTERO MAYOR QUE 0

#EJECUCION DEL APRENDIZAJE Y GENERACION DEL MODELO

set.seed(1)
model <- mlp(x= trainSet[,-salida],
             y= trainSet[, salida],
             inputsTest=  validSet[,-salida],
             targetsTest= validSet[, salida],
             size= topologia,
             maxit=ciclosMaximos,
             learnFuncParams=c(razonAprendizaje),
             shufflePatterns = F
             )

#GRAFICO DE LA EVOLUCION DEL ERROR
#plotIterativeError(model)

# lo desactivamos porque a veces tiene problemas con las escalas.
# establece el m�nimo y el m�ximo en funci�n del error de entrenamiento y a veces
# no muestra el de validaci�n si es m�s peque�o.
#Usamos graficaError() pero antes hay que calcular iterativeErrors que contiene
#los errores MSE por ciclo de train y valid

# DATAFRAME CON LOS ERRORES POR CICLo: de entrenamiento y de validacion
iterativeErrors <- data.frame(MSETrain= (model$IterativeFitError/ nrow(trainSet)),
                              MSEValid= (model$IterativeTestError/nrow(validSet)))

graficaError(iterativeErrors)


######################################################################
#SE OBTIENE EL NuMERO DE CICLOS DONDE EL ERROR DE VALIDACION ES MINIMO
#######################################################################

nuevosCiclos <- which.min(model$IterativeTestError)

#ENTRENAMOS LA MISMA RED CON LAS ITERACIONES QUE GENERAN MENOR ERROR DE VALIDACION
set.seed(1)
model <- mlp(x= trainSet[,-salida],
             y= trainSet[, salida],
             inputsTest=  validSet[,-salida],
             targetsTest= validSet[, salida],
             size= topologia,
             maxit=nuevosCiclos,
             learnFuncParams=c(razonAprendizaje),
             shufflePatterns = F
)
#GRAFICO DE LA EVOLUCION DEL ERROR
#plotIterativeError(model)

iterativeErrors1 <- data.frame(MSETrain= (model$IterativeFitError/ nrow(trainSet)),
                              MSEValid= (model$IterativeTestError/nrow(validSet)))
graficaError(iterativeErrors1)

#CALCULO DE PREDICCIONES
prediccionesTrain <- predict(model,trainSet[,-salida])
prediccionesValid <- predict(model,validSet[,-salida])
prediccionesTest  <- predict(model, testSet[,-salida])

#CALCULO DE LOS ERRORES
errors <- c(TrainMSE= MSE(pred= prediccionesTrain,obs= trainSet[,salida]),
            ValidMSE= MSE(pred= prediccionesValid,obs= validSet[,salida]),
            TestMSE=  MSE(pred= prediccionesTest ,obs=  testSet[,salida]))
errors





#SALIDAS DE LA RED
outputsTrain <- data.frame(pred= prediccionesTrain,obs= trainSet[,salida])
outputsValid <- data.frame(pred= prediccionesValid,obs= validSet[,salida])
outputsTest  <- data.frame(pred= prediccionesTest, obs=  testSet[,salida])




#GUARDANDO RESULTADOS
saveRDS(model,"nnet.rds")
write.csv2(errors,"finalErrors.csv")
write.csv2(iterativeErrors,"iterativeErrors.csv")
write.csv2(outputsTrain,"netOutputsTrain.csv")
write.csv2(outputsValid,"netOutputsValid.csv")
write.csv2(outputsTest, "netOutputsTest.csv")
