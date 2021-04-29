library("data.table")
library("Rcpp")
library("here")

setwd(here())

data("iris")
iris_dt <- as.data.table(iris)

fwrite('../datasets/prueba.gz')
