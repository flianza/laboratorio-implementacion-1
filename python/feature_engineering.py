from datatable import Frame, f, fread, ifelse, math
import datatable as dt
import random
from utils import timer


def run(dataset: Frame) -> Frame:
    dataset = arreglar_errores_dataset_original(dataset)
    dataset = agregar_variables_nuevas(dataset)
    dataset = arreglar_infinitos(dataset)

    return dataset


@timer
def arreglar_errores_dataset_original(dataset: Frame) -> Frame:
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

    return dataset


@timer
def agregar_variables_nuevas(dataset: Frame) -> Frame:
    dataset['tarjetas_status01'] = dataset[:, dt.rowmax([f.Master_status, f.Visa_status])] # 3
    dataset['tarjetas_status02'] = dataset[:, dt.rowmin([f.Master_status, f.Visa_status])] # 2
    dataset['tarjetas_fultimo_cierre01'] = dataset[:, dt.rowmax([f.Master_fultimo_cierre, f.Visa_fultimo_cierre])] # 479
    dataset['tarjetas_fultimo_cierre02'] = dataset[:, dt.rowmin([f.Master_fultimo_cierre, f.Visa_fultimo_cierre])] # 421
    dataset['tarjetas_Finiciomora'] = dataset[:, dt.rowmin([f.Master_Finiciomora, f.Visa_Finiciomora])] # 12
    dataset['tarjetas_Fvencimiento'] = dataset[:, dt.rowmin([f.Master_Fvencimiento, f.Visa_Fvencimiento])] # 359
    dataset['tarjetas_delinquency'] = dataset[:, dt.rowmax([f.Master_delinquency, f.Visa_delinquency])] # 18
    dataset['tarjetas_mfinanciacion_limite'] = dataset[:, dt.rowsum([f.Master_mfinanciacion_limite, f.Visa_mfinanciacion_limite])] # 230
    dataset['tarjetas_msaldototal'] = dataset[:, f.Master_msaldototal + f.Visa_msaldototal] # 57
    dataset['tarjetas_msaldopesos'] = dataset[:, f.Master_msaldopesos + f.Visa_msaldopesos] # 46
    dataset['tarjetas_msaldodolares'] = dataset[:, f.Master_msaldodolares + f.Visa_msaldodolares] # 1142 pero una derivada 104
    dataset['tarjetas_mconsumospesos'] = dataset[:, f.Master_mconsumospesos + f.Visa_mconsumospesos] # 400
    dataset['tarjetas_mconsumosdolares'] = dataset[:, f.Master_mconsumosdolares + f.Visa_mconsumosdolares] # 891 pero con derivadas 352
    dataset['tarjetas_mlimitecompra'] = dataset[:, f.Master_mlimitecompra + f.Visa_mlimitecompra] # 186 pero con derivadas 26
    dataset['tarjetas_madelantopesos'] = dataset[:, f.Master_madelantopesos + f.Visa_madelantopesos] # 666 pero derivadas 26
    dataset['tarjetas_madelantodolares'] = dataset[:, f.Master_madelantodolares + f.Visa_madelantodolares] # 294 y derivadas 33
    dataset['tarjetas_fultimo_cierre'] = dataset[:, dt.rowmax([f.Master_fultimo_cierre, f.Visa_fultimo_cierre])] # 448
    dataset['tarjetas_mpagado'] = dataset[:, f.Master_mpagado + f.Visa_mpagado] # 384 y derivadas 29
    dataset['tarjetas_mpagospesos'] = dataset[:, f.Master_mpagospesos + f.Visa_mpagospesos] # 28
    dataset['tarjetas_mpagosdolares'] = dataset[:, f.Master_mpagosdolares + f.Visa_mpagosdolares] # 1017 y derivadas 255
    dataset['tarjetas_fechaalta'] = dataset[:, dt.rowmax([f.Master_fechaalta, f.Visa_fechaalta])] # 159
    dataset['tarjetas_mconsumototal'] = dataset[:, f.Master_mconsumototal + f.Visa_mconsumototal] # 512 y derivadas 365
    dataset['tarjetas_cconsumos'] = dataset[:, f.Master_cconsumos + f.Visa_cconsumos] # 424
    dataset['tarjetas_cadelantosefectivo'] = dataset[:, f.Master_cadelantosefectivo + f.Visa_cadelantosefectivo] # 750
    dataset['tarjetas_mpagominimo'] = dataset[:, f.Master_mpagominimo + f.Visa_mpagominimo] # 98
    dataset['ratio_tarjetas_msaldodolares__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_msaldodolares / f.tarjetas_mlimitecompra] # 104
    dataset['ratio_tarjetas_msaldodolares__tarjetas_msaldototal'] = dataset[:, f.tarjetas_msaldodolares / f.tarjetas_msaldototal] # 611
    dataset['ratio_tarjetas_mconsumospesos__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mconsumospesos / f.tarjetas_mlimitecompra] # 244
    dataset['ratio_tarjetas_madelantopesos__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_madelantopesos / f.tarjetas_mlimitecompra] # 26
    dataset['ratio_tarjetas_madelantodolares__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_madelantodolares / f.tarjetas_mlimitecompra] # 33
    dataset['ratio_tarjetas_mpagospesos__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mpagospesos / f.tarjetas_mlimitecompra] # 38
    dataset['ratio_tarjetas_mpagominimo__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mpagominimo / f.tarjetas_mlimitecompra] # 100
    dataset['ratio_tarjetas_mpagado__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mpagado / f.tarjetas_mlimitecompra] # 29
    dataset['ratio_tarjetas_mpagosdolares__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mpagosdolares / f.tarjetas_mlimitecompra] # 255
    dataset['ratio_tarjetas_mconsumototal__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mconsumototal / f.tarjetas_mlimitecompra] # 365
    dataset['ratio_tarjetas_mconsumosdolares__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mconsumosdolares / f.tarjetas_mlimitecompra] # 352
    dataset['ratio_tarjetas_msaldopesos__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_msaldopesos / f.tarjetas_mlimitecompra] # 270
    dataset['ratio_tarjetas_msaldopesos__tarjetas_msaldototal'] = dataset[:, f.tarjetas_msaldopesos / f.tarjetas_msaldototal] # 414
    dataset['ratio_Master_mlimitecompra__tarjetas_mlimitecompra'] = dataset[:, f.Master_mlimitecompra / f.tarjetas_mlimitecompra] # 367
    dataset['ratio_Visa_mlimitecompra__tarjetas_mlimitecompra'] = dataset[:, f.Visa_mlimitecompra / f.tarjetas_mlimitecompra] # 192

    # v2
    dataset['ctarjetas_credito'] = dataset[:, f.ctarjeta_master + f.ctarjeta_visa] # 27
    dataset['ctarjetas'] = dataset[:, f.ctarjetas_credito + f.ctarjeta_debito] # 623
    dataset['ratio_mprestamos_personales__cprestamos_personales'] = dataset[:, f.mprestamos_personales / f.cprestamos_personales] # 127
    dataset['cextracciones'] = dataset[:, f.cextraccion_autoservicio + f.ccajas_extracciones] # 157
    dataset['ratio_mextraccion_autoservicio__mcuentas_saldo'] = dataset[:, f.mextraccion_autoservicio / f.mcuentas_saldo] # 565
    dataset['ccomisiones'] = dataset[:, f.ccomisiones_mantenimiento + f.ccomisiones_otras] # 578
    dataset['ratio_mcomisiones__ccomisiones'] = dataset[:, f.mcomisiones / f.ccomisiones] # 508
    dataset['ctransacciones'] = dataset[:, f.ccallcenter_transacciones + f.chomebanking_transacciones + f.ccajas_transacciones] # 485
    dataset['ratio_ctransacciones__cproductos'] = dataset[:, f.ctransacciones / f.cproductos] # 472

    # v3
    dataset['mpayroll_total'] = dataset[:, f.mpayroll + f.mpayroll2] # 68
    dataset['ratio_mpayroll_total__cliente_edad'] = dataset[:, f.mpayroll_total / f.cliente_edad] # 87
    dataset['ratio_mcaja_ahorro__cliente_edad'] = dataset[:, f.mcaja_ahorro / f.cliente_edad] # 23
    dataset['ratio_mcuentas_saldo__cliente_edad'] = dataset[:, f.mcuentas_saldo / f.cliente_edad] # 102
    dataset['cseguros_total'] = dataset[:, f.cseguro_vida + f.cseguro_auto + f.cseguro_vivienda + f.cseguro_accidentes_personales] # 454
    dataset['ratio_cseguros_total__cliente_antiguedad'] = dataset[:, f.cseguros_total / f.cliente_antiguedad] # 628

    # Resultaron no ser importantes

    # v1
    # dataset['ratio_tarjetas_msaldototal__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mlimitecompra / f.tarjetas_mlimitecompra] # 2544

    # v2
    # dataset['ratio_mrentabilidad__cproductos'] = dataset[:, f.mrentabilidad / f.cproductos] # 911
    # dataset['dif_tarjetas_mconsumototal__tarjetas_mpagado'] = dataset[:, f.tarjetas_mconsumototal - f.tarjetas_mpagado] # 1277
    # dataset['ratio_mrentabilidad__mcomisiones'] = dataset[:, f.mrentabilidad / f.mcomisiones] # 1100

    # v3
    # dataset['ratio_mrentabilidad__mcuentas_saldo'] = dataset[:, f.mrentabilidad / f.mcuentas_saldo] # 2042
    # dataset['ratio_mrentabilidad__cliente_antiguedad'] = dataset[:, f.mrentabilidad / f.cliente_antiguedad] # 1854
    # dataset['ratio_mrentabilidad__cliente_edad'] = dataset[:, f.mrentabilidad / f.cliente_edad] # 1811
    # dataset['ratio_cliente_antiguedad__cliente_edad'] = dataset[:, f.cliente_antiguedad / f.cliente_edad] # 1719

    return dataset


@timer
def eliminar_columnas(dataset: Frame) -> Frame:
    columnas_a_eliminar = [
        'tarjetas_msaldodolares',
        'tarjetas_mconsumosdolares',
        'tarjetas_mpagosdolares',
    ]

    for columna in columnas_a_eliminar:
        dataset[columna] = None

    return dataset


@timer
def arreglar_infinitos(dataset: Frame) -> Frame:
    for column in dataset.names:
        if column != 'clase_ternaria':
            dataset[column] = dataset[:, ifelse(math.isinf(f[column]) == 1, None, f[column])]

    return dataset


@timer
def agregar_canaritos(dataset: Frame, cantidad: int = 20) -> Frame:
    for i in range(cantidad):
        nombre_canarito = f'canarito{i}'
        dataset[nombre_canarito] = random.uniform(size=dataset.shape[0])
    return dataset


@timer
def leer_dataset() -> Frame:
    return fread('../datasetsOri/paquete_premium.txt.gz')


if __name__ == '__main__':
    dataset = leer_dataset()
    dataset = run(dataset)
    dataset.to_csv(path='../datasets/datos_fe_v4.gz', compression='gzip')

