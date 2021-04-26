from datatable import Frame, f, ifelse, shift, fread, by, shift
import pandas as pd
import datatable as dt
import random

from utils import timer


VENTANA = 6


def run(dataset: Frame) -> Frame:
    dataset = arreglar_errores_dataset_original(dataset)
    dataset = agregar_variables_nuevas(dataset)
    dataset = agregar_variables_historicas_python(dataset)

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

    dataset['ratio_Master_mlimitecompra__tarjetas_mlimitecompra'] = dataset[:, f.Master_mlimitecompra / f.tarjetas_mlimitecompra]
    dataset['ratio_Visa_mlimitecompra__tarjetas_mlimitecompra'] = dataset[:, f.Visa_mlimitecompra / f.tarjetas_mlimitecompra]
    dataset['ratio_tarjetas_msaldototal__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mlimitecompra / f.tarjetas_mlimitecompra]
    dataset['ratio_tarjetas_msaldopesos__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_msaldopesos / f.tarjetas_mlimitecompra]
    dataset['ratio_tarjetas_msaldopesos__tarjetas_msaldototal'] = dataset[:, f.tarjetas_msaldopesos / f.tarjetas_msaldototal]
    dataset['ratio_tarjetas_msaldodolares__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_msaldodolares / f.tarjetas_mlimitecompra]
    dataset['ratio_tarjetas_msaldodolares__tarjetas_msaldototal'] = dataset[:, f.tarjetas_msaldodolares / f.tarjetas_msaldototal]
    dataset['ratio_tarjetas_mconsumospesos__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mconsumospesos / f.tarjetas_mlimitecompra]
    dataset['ratio_tarjetas_mconsumosdolares__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mconsumosdolares / f.tarjetas_mlimitecompra]
    dataset['ratio_tarjetas_madelantopesos__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_madelantopesos / f.tarjetas_mlimitecompra]
    dataset['ratio_tarjetas_madelantodolares__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_madelantodolares / f.tarjetas_mlimitecompra]
    dataset['ratio_tarjetas_mpagado__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mpagado / f.tarjetas_mlimitecompra]
    dataset['ratio_tarjetas_mpagospesos__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mpagospesos / f.tarjetas_mlimitecompra]
    dataset['ratio_tarjetas_mpagosdolares__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mpagosdolares / f.tarjetas_mlimitecompra]
    dataset['ratio_tarjetas_mconsumototal__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mconsumototal / f.tarjetas_mlimitecompra]
    dataset['ratio_tarjetas_mpagominimo__tarjetas_mlimitecompra'] = dataset[:, f.tarjetas_mpagominimo / f.tarjetas_mlimitecompra]

    return dataset


@timer
def agregar_variables_historicas_python(dataset: Frame) -> Frame:
    dataset = dataset[:, :, dt.sort('numero_de_cliente', 'foto_mes')]
    valores_historicos = obtener_valores_historicos_para_columna('mcuentas_saldo', dataset)
    dataset.cbind(valores_historicos)
    return dataset


@timer
def obtener_valores_historicos_para_columna(columna: str, dataset: Frame) -> Frame:
    previo = create_roll_columns(dataset[:, ['numero_de_cliente', columna]].to_pandas(),
                                 ['numero_de_cliente'],
                                 VENTANA,
                                 [columna],
                                 ['max', 'min'])
    previo = Frame(previo)

    previo[f'{columna}_previo'] = dataset[:, shift(f[columna]), by('numero_de_cliente')][columna]

    previo = previo[:, f[:].remove([f[columna], f.numero_de_cliente])]

    return previo


def create_roll_columns (x: pd.DataFrame, g_c: [str], roll: int, roll_cols: [str], roll_types: [str]):
    for i in roll_cols:
        rolling = x.groupby(g_c)[i].rolling(roll, min_periods=1)
        for aggregation in roll_types:
            if aggregation == 'max':
                nm = f'{i}_roll_max'
                x[nm] = rolling.max().reset_index(0, drop=True)
            if aggregation == 'min':
                nm = f'{i}_roll_min'
                x[nm] = rolling.min().reset_index(0, drop=True)
    return x


@timer
def agregar_canaritos(dataset: Frame, cantidad: int = 20) -> Frame:
    for i in range(cantidad):
        nombre_canarito = f'canarito{i}'
        dataset[nombre_canarito] = random.uniform(size=dataset.shape[0])
    return dataset


@timer
def leer_dataset() -> Frame:
    return fread('datasets/originales/datos.gz')


if __name__ == '__main__':
    dataset = leer_dataset()
    dataset = run(dataset)
    print(dataset.head(25))

