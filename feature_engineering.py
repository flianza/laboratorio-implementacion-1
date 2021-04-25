from datatable import Frame, f, ifelse, shift, fread
import datatable as dt
import random


def run(dataset: Frame) -> Frame:
    # Errores del dataset
    dataset[f.foto_mes == 201701, 'ccajas_consultas'] = None
    dataset[f.foto_mes == 201702, 'ccajas_consultas'] = None

    dataset[f.foto_mes == 201801, 'internet'] = None
    dataset[f.foto_mes == 201801, 'thomebanking'] = None
    dataset[f.foto_mes == 201801, 'chomebanking_transacciones'] = None
    dataset[f.foto_mes == 201801, 'tcallcenter'] = None
    dataset[f.foto_mes == 201801, 'ccallcenter_transacciones'] = None
    dataset[f.foto_mes == 201801, 'cprestamos_personales'] = None
    dataset[f.foto_mes == 201801, 'mprestamos_personales'] = None
    dataset[f.foto_mes == 201801, 'mprestamos_hipotecarios'] = None
    dataset[f.foto_mes == 201801, 'ccajas_transacciones'] = None
    dataset[f.foto_mes == 201801, 'ccajas_consultas'] = None
    dataset[f.foto_mes == 201801, 'ccajas_depositos'] = None
    dataset[f.foto_mes == 201801, 'ccajas_extracciones'] = None
    dataset[f.foto_mes == 201801, 'ccajas_otras'] = None

    dataset[f.foto_mes == 201806, 'tcallcenter'] = None
    dataset[f.foto_mes == 201806, 'ccallcenter_transacciones'] = None

    dataset[f.foto_mes == 201904, 'ctarjeta_visa_debitos_automaticos'] = None
    dataset[f.foto_mes == 201904, 'mttarjeta_visa_debitos_automaticos'] = None

    dataset[f.foto_mes == 201905, 'mrentabilidad'] = None
    dataset[f.foto_mes == 201905, 'mrentabilidad_annual'] = None
    dataset[f.foto_mes == 201905, 'mcomisiones'] = None
    dataset[f.foto_mes == 201905, 'mpasivos_margen'] = None
    dataset[f.foto_mes == 201905, 'mactivos_margen'] = None
    dataset[f.foto_mes == 201905, 'ctarjeta_visa_debitos_automaticos'] = None
    dataset[f.foto_mes == 201905, 'ccomisiones_otras'] = None
    dataset[f.foto_mes == 201905, 'mcomisiones_otras'] = None

    dataset[f.foto_mes == 201910, 'mpasivos_margen'] = None
    dataset[f.foto_mes == 201910, 'mactivos_margen'] = None
    dataset[f.foto_mes == 201910, 'ccomisiones_otras'] = None
    dataset[f.foto_mes == 201910, 'mcomisiones_otras'] = None
    dataset[f.foto_mes == 201910, 'mcomisiones'] = None
    dataset[f.foto_mes == 201910, 'mrentabilidad'] = None
    dataset[f.foto_mes == 201910, 'mrentabilidad_annual'] = None
    dataset[f.foto_mes == 201910, 'chomebanking_transacciones'] = None
    dataset[f.foto_mes == 201910, 'ctarjeta_visa_descuentos'] = None
    dataset[f.foto_mes == 201910, 'ctarjeta_master_descuentos'] = None
    dataset[f.foto_mes == 201910, 'mtarjeta_visa_descuentos'] = None
    dataset[f.foto_mes == 201910, 'mtarjeta_master_descuentos'] = None
    dataset[f.foto_mes == 201910, 'ccajeros_propios_descuentos'] = None
    dataset[f.foto_mes == 201910, 'mcajeros_propios_descuentos'] = None

    dataset[f.foto_mes == 202001, 'cliente_vip'] = None

    # Agregado de nuevas variables
    dataset['tarjetas_status01'] = dataset[:, dt.rowmax([f.Master_status, f.Visa_status])]
    dataset['tarjetas_status02'] = dataset[:, dt.rowmin([f.Master_status, f.Visa_status])]
    dataset['tarjetas_fultimo_cierre01'] = dataset[:, dt.rowmax([f.Master_fultimo_cierre, f.Visa_fultimo_cierre])]
    dataset['tarjetas_fultimo_cierre02'] = dataset[:, dt.rowmin([f.Master_fultimo_cierre, f.Visa_fultimo_cierre])]
    dataset['tarjetas_Finiciomora'] = dataset[:, dt.rowmin([f.Master_Finiciomora, f.Visa_Finiciomora])]
    dataset['tarjetas_Fvencimiento'] = dataset[:, dt.rowmin([f.Master_Fvencimiento, f.Visa_Fvencimiento])]
    dataset['tarjetas_delinquency'] = dataset[:, dt.rowmax([f.Master_delinquency, f.Visa_delinquency])]
    dataset['tarjetas_mfinanciacion_limite'] = dataset[:, dt.rowsum([f.Master_mfinanciacion_limite, f.Visa_mfinanciacion_limite])]
    dataset['tarjetas_msaldototal'] = dataset[:, dt.rowsum([f.Master_msaldototal, f.Visa_msaldototal])]
    dataset['tarjetas_msaldopesos'] = dataset[:, dt.rowsum([f.Master_msaldopesos, f.Visa_msaldopesos])]
    dataset['tarjetas_msaldodolares'] = dataset[:, dt.rowsum([f.Master_msaldodolares, f.Visa_msaldodolares])]
    dataset['tarjetas_mconsumospesos'] = dataset[:, dt.rowsum([f.Master_mconsumospesos, f.Visa_mconsumospesos])]
    dataset['tarjetas_mconsumosdolares'] = dataset[:, dt.rowsum([f.Master_mconsumosdolares, f.Visa_mconsumosdolares])]
    dataset['tarjetas_mlimitecompra'] = dataset[:, dt.rowsum([f.Master_mlimitecompra, f.Visa_mlimitecompra])]
    dataset['tarjetas_madelantopesos'] = dataset[:, dt.rowsum([f.Master_madelantopesos, f.Visa_madelantopesos])]
    dataset['tarjetas_madelantodolares'] = dataset[:, dt.rowsum([f.Master_madelantodolares, f.Visa_madelantodolares])]
    dataset['tarjetas_fultimo_cierre'] = dataset[:, dt.rowmax([f.Master_fultimo_cierre, f.Visa_fultimo_cierre])]
    dataset['tarjetas_mpagado'] = dataset[:, dt.rowsum([f.Master_mpagado, f.Visa_mpagado])]
    dataset['tarjetas_mpagospesos'] = dataset[:, dt.rowsum([f.Master_mpagospesos, f.Visa_mpagospesos])]
    dataset['tarjetas_mpagosdolares'] = dataset[:, dt.rowsum([f.Master_mpagosdolares, f.Visa_mpagosdolares])]
    dataset['tarjetas_fechaalta'] = dataset[:, dt.rowmax([f.Master_fechaalta, f.Visa_fechaalta])]
    dataset['tarjetas_mconsumototal'] = dataset[:, dt.rowsum([f.Master_mconsumototal, f.Visa_mconsumototal])]
    dataset['tarjetas_cconsumos'] = dataset[:, dt.rowsum([f.Master_cconsumos, f.Visa_cconsumos])]
    dataset['tarjetas_cadelantosefectivo'] = dataset[:, dt.rowsum([f.Master_cadelantosefectivo, f.Visa_cadelantosefectivo])]
    dataset['tarjetas_mpagominimo'] = dataset[:, dt.rowsum([f.Master_mpagominimo, f.Visa_mpagominimo])]

    # a partir de aqui juego con la suma de Mastercard y Visa
    # dataset[, mvr_Master_mlimitecompra := Master_mlimitecompra / mv_mlimitecompra]
    # dataset[, mvr_Visa_mlimitecompra := Visa_mlimitecompra / mv_mlimitecompra]
    # dataset[, mvr_msaldototal := mv_msaldototal / mv_mlimitecompra]
    # dataset[, mvr_msaldopesos := mv_msaldopesos / mv_mlimitecompra]
    # dataset[, mvr_msaldopesos2 := mv_msaldopesos / mv_msaldototal]
    # dataset[, mvr_msaldodolares := mv_msaldodolares / mv_mlimitecompra]
    # dataset[, mvr_msaldodolares2 := mv_msaldodolares / mv_msaldototal]
    # dataset[, mvr_mconsumospesos := mv_mconsumospesos / mv_mlimitecompra]
    # dataset[, mvr_mconsumosdolares := mv_mconsumosdolares / mv_mlimitecompra]
    # dataset[, mvr_madelantopesos := mv_madelantopesos / mv_mlimitecompra]
    # dataset[, mvr_madelantodolares := mv_madelantodolares / mv_mlimitecompra]
    # dataset[, mvr_mpagado := mv_mpagado / mv_mlimitecompra]
    # dataset[, mvr_mpagospesos := mv_mpagospesos / mv_mlimitecompra]
    # dataset[, mvr_mpagosdolares := mv_mpagosdolares / mv_mlimitecompra]
    # dataset[, mvr_mconsumototal := mv_mconsumototal / mv_mlimitecompra]
    # dataset[, mvr_mpagominimo := mv_mpagominimo / mv_mlimitecompra]

    return dataset


def agregar_canaritos(dataset: Frame, cantidad: int = 20) -> Frame:
    for i in range(cantidad):
        nombre_canarito = f'canarito{i}'
        dataset[nombre_canarito] = random.uniform(size=dataset.shape[0])
    return dataset


if __name__ == '__main__':
    dataset = fread('datasets/originales/datos.gz')
    dataset = run(dataset)
