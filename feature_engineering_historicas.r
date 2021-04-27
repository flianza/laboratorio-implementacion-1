rm(list=ls())
gc()

library("data.table")
library("Rcpp")
library("here")

setwd(here())

cppFunction('NumericVector fhistC(NumericVector pcolumna, IntegerVector pdesde) {
  /* Aqui se cargan los valores para la regresion */
  double  x[100] ;
  double  y[100] ;

  int n = pcolumna.size();
  NumericVector out( 5*n );

  for(int i = 0; i < n; i++)
  {
    //lag
    if( pdesde[i]-1 < i )  out[ i + 4*n ]  =  pcolumna[i-1] ;
    else                   out[ i + 4*n ]  =  NA_REAL ;


    int  libre    = 0 ;
    int  xvalor   = 1 ;

    for( int j= pdesde[i]-1;  j<=i; j++ )
    {
       double a = pcolumna[j] ;

       if( !R_IsNA( a ) ) 
       {
          y[ libre ]= a ;
          x[ libre ]= xvalor ;
          libre++ ;
       }

       xvalor++ ;
    }

    /* Si hay al menos dos valores */
    if( libre > 1 )
    {
      double  xsum  = x[0] ;
      double  ysum  = y[0] ;
      double  xysum = xsum * ysum ;
      double  xxsum = xsum * xsum ;
      double  vmin  = y[0] ;
      double  vmax  = y[0] ;

      for( int h=1; h<libre; h++)
      { 
        xsum  += x[h] ;
        ysum  += y[h] ; 
        xysum += x[h]*y[h] ;
        xxsum += x[h]*x[h] ;

        if( y[h] < vmin )  vmin = y[h] ;
        if( y[h] > vmax )  vmax = y[h] ;
      }

      out[ i ]  =  (libre*xysum - xsum*ysum)/(libre*xxsum -xsum*xsum) ;
      out[ i + n ]    =  vmin ;
      out[ i + 2*n ]  =  vmax ;
      out[ i + 3*n ]  =  ysum / libre ;
    }
    else
    {
      out[ i       ]  =  NA_REAL ; 
      out[ i + n   ]  =  NA_REAL ;
      out[ i + 2*n ]  =  NA_REAL ;
      out[ i + 3*n ]  =  NA_REAL ;
    }
  }

  return  out;
}')

VENTANA <- 6
t1 <- Sys.time()

dataset <- fread("./datasets/data_fe.gz")

setorder(dataset, numero_de_cliente, foto_mes)

last <- nrow(dataset)
kcampo_id_idx <- match("numero_de_cliente", names(dataset))
vector_ids <- dataset[[kcampo_id_idx]]

vector_desde <- seq(-VENTANA + 2, nrow(dataset) - VENTANA + 1)
vector_desde[1:VENTANA] <- 1
for(i in 2:last) {
  if(vector_ids[i-1] != vector_ids[i]) {
    vector_desde[i] <- i 
  }
}
for(i in 2:last) {
  if(vector_desde[i] < vector_desde[i-1]) {
    vector_desde[i] <- vector_desde[i-1]
  }
}

columnas_no_procesar <- c("numero_de_cliente", "foto_mes", "clase_ternaria")
columnas_a_procesar <- setdiff(names(dataset), columnas_no_procesar)  

for(campo in columnas_a_procesar) {
  campo_idx <- match(campo, names(dataset))
  col_original <- dataset[[campo_idx]]
  
  nueva_col <- fhistC(col_original, vector_desde) 
  
  dataset[, paste(campo, "__tend", sep="") := nueva_col[(0*last+1):(1*last)]]
  dataset[, paste(campo, "__min", sep="") := nueva_col[(1*last+1):(2*last)]]
  dataset[, paste(campo, "__max", sep="") := nueva_col[(2*last+1):(3*last)]]
  dataset[, paste(campo, "__avg", sep="") := nueva_col[(3*last+1):(4*last)]]
  dataset[, paste(campo, "__lag", sep="") := nueva_col[(4*last+1):(5*last)]]
}

nuevo_orden <- c(setdiff(colnames(dataset), "clase_ternaria"), "clase_ternaria")
setcolorder(dataset, nuevo_orden)

fwrite(dataset, file="./datasets/data_fe_hist.gz")

t1 <- Sys.time()
tiempo <- as.numeric(t1 - t0, units = "secs")

cat("El Feature Engineering ha corrido en: ", tiempo, " segundos.\n")

rm(list=ls())
gc()

