import polars as pl

cat_features =[
    'mcc_code',
    'event_type_nm',
    'event_desc',
    'channel_indicator_type',
    'channel_indicator_sub_type',
    'pos_cd',
    'timezone', 
    'operating_system_type',
    'developer_tools',
    'phone_voip_call_state',
    'is_desc_changed',
    'is_pos_changed',        
    'is_currency_changed',
    'is_mcc_changed'
]

def apply_type_casting(df_lazy):
    """
    Применение оптимальных типов данных для уменьшения объема памяти и ускорения обработки.
    """
    df_lazy = df_lazy.with_columns(
        pl.col('event_dttm').str.to_datetime(),
        pl.col('event_type_nm').cast(pl.Int8, strict=False),
        pl.col('event_desc').cast(pl.Int16, strict=False),
        pl.col('channel_indicator_type').cast(pl.Int8, strict=False),
        pl.col('channel_indicator_sub_type').cast(pl.Int16, strict=False),
        pl.col('operaton_amt').cast(pl.Float32, strict=False),
        pl.col('currency_iso_cd').cast(pl.Int16, strict=False),
        pl.col('pos_cd').cast(pl.Int16, strict=False),
        pl.col('timezone').cast(pl.Int16, strict=False),
        pl.col('operating_system_type').cast(pl.Int8, strict=False),
        pl.col('developer_tools').cast(pl.Int8, strict=False),
        pl.col('phone_voip_call_state').cast(pl.Int8, strict=False),
        pl.col('web_rdp_connection').cast(pl.Int8, strict=False),
        pl.col('battery').str.replace('%', '').cast(pl.Int8, strict=False),
        pl.col('mcc_code').cast(pl.Int8, strict=False),
        pl.col('screen_size').str.split_exact('x', 1).struct.rename_fields(['screen_w', 'screen_h']).alias('screen_dims')
    ).with_columns(
        pl.col('event_dttm').dt.date().alias('event_date')
    ).unnest('screen_dims').with_columns(
        pl.col('screen_w').cast(pl.Int16, strict=False),
        pl.col('screen_h').cast(pl.Int16, strict=False)
    ).drop(['accept_language', 'browser_language', 'screen_size', 'device_system_version'])
    
    return df_lazy

def generate_features(df_lazy):
    """
    Генерация новых признаков на основе существующих данных для улучшения качества модели.
    """
    df_lazy = df_lazy.with_columns(pl.col('operaton_amt').fill_null(0.0))
    
    df_lazy = df_lazy.sort(['customer_id', 'event_dttm'])

    # Лаговые признаки для предыдущей транзакции клиента
    df_lazy = df_lazy.with_columns(
        prev_dttm = pl.col('event_dttm').shift(1).over('customer_id'),
        prev_amt = pl.col('operaton_amt').shift(1).over('customer_id'),
        prev_mcc = pl.col('mcc_code').shift(1).over('customer_id'),
        prev_timezone = pl.col('timezone').shift(1).over('customer_id'),
        prev_os = pl.col('operating_system_type').shift(1).over('customer_id'),
        prev_pos = pl.col('pos_cd').shift(1).over('customer_id'),
        prev_currency = pl.col('currency_iso_cd').shift(1).over('customer_id')
    ).with_columns(
        time_since_last_op_sec = (pl.col('event_dttm') - pl.col('prev_dttm')).dt.total_seconds(),
        amt_ratio_to_prev = pl.col('operaton_amt') / (pl.col('prev_amt') + 0.1),
        is_mcc_changed = pl.when(pl.col('mcc_code') != pl.col('prev_mcc')).then(pl.lit(1)).otherwise(pl.lit(0)),
        is_timezone_changed = pl.when(pl.col('timezone') != pl.col('prev_timezone')).then(pl.lit(1)).otherwise(pl.lit(0)),
        is_device_changed = pl.when(pl.col('operating_system_type') != pl.col('prev_os')).then(pl.lit(1)).otherwise(pl.lit(0))
    ).drop(['prev_dttm', 'prev_amt', 'prev_mcc', 'prev_timezone', 'prev_os'])

    # Профильные признаки нарастающим итогом для каждого клиента
    df_lazy = df_lazy.with_columns(
        client_op_seq_num = pl.col('event_dttm').cum_count().over('customer_id')
    ).with_columns(
        client_expanding_mean_amt = pl.col('operaton_amt').cum_sum().over('customer_id') / pl.col('client_op_seq_num')
    ).with_columns(
        amt_diff_from_exp_mean = pl.col('operaton_amt') - pl.col('client_expanding_mean_amt')
    )
    
    # Данные за час/сутки
    df_lazy = df_lazy.with_columns(
        # Вычитаем 1, чтобы получить количество именно предыдущих операций
        op_count_1h = pl.col('event_dttm').is_not_null().cast(pl.Int32).rolling_sum_by(window_size='1h', by='event_dttm').over('customer_id') - 1,
        op_count_24h = pl.col('event_dttm').is_not_null().cast(pl.Int32).rolling_sum_by(window_size='24h', by='event_dttm').over('customer_id') - 1,
        
        # Вычитаем текущую сумму, чтобы получить сумму именно предыдущих операций
        amt_sum_1h = pl.col('operaton_amt').rolling_sum_by(window_size='1h', by='event_dttm').over('customer_id') - pl.col('operaton_amt'),
        amt_sum_24h = pl.col('operaton_amt').rolling_sum_by(window_size='24h', by='event_dttm').over('customer_id') - pl.col('operaton_amt')
    )
    
    # Предыдущий тип события и предыдущее описание
    df_lazy = df_lazy.with_columns(
        prev_event_type_nm = pl.col('event_type_nm').shift(1).over('customer_id'),
        prev_event_desc = pl.col('event_desc').shift(1).over('customer_id')
    )
    
    # Извлечение часа из времени и признак ночи
    df_lazy = df_lazy.with_columns(
        hour = pl.col('event_dttm').dt.hour().cast(pl.UInt8)
    ).with_columns(
        is_night = pl.when((pl.col('hour') >= 0) & (pl.col('hour') < 6)).then(1).otherwise(0).cast(pl.UInt8)
    )
    
    # 
    df_lazy = df_lazy.with_columns(
        is_desc_changed = pl.when(pl.col('event_desc') != pl.col('prev_event_desc')).then(pl.lit(1)).otherwise(pl.lit(0)),
        is_pos_changed = pl.when(pl.col('pos_cd') != pl.col('prev_pos')).then(pl.lit(1)).otherwise(pl.lit(0)),
        is_currency_changed = pl.when(pl.col('currency_iso_cd') != pl.col('prev_currency')).then(pl.lit(1)).otherwise(pl.lit(0))
    ).with_columns(
        desc_change_count = pl.col('is_desc_changed').rolling_sum_by(window_size='1h', by='event_dttm').over('customer_id') - pl.col('is_desc_changed')
    ).drop(['prev_pos', 'prev_currency'])
    
    garbage_cols =[
        'prev_event_type_nm', 'prev_event_desc', 
        'is_timezone_changed', 'is_device_changed', 'currency_iso_cd', 
        'battery', 'web_rdp_connection', 'compromised'
    ]
    df_lazy = df_lazy.drop(garbage_cols)
    
    for col in cat_features:
        df_lazy = df_lazy.with_columns(
            pl.col(col)
            .cast(pl.String) # Переводим в строку
            .fill_null('-1') # Заполняем нативные null
            .str.replace('(?i)^nan$|<NA>|None', '-1') # Убиваем текстовый мусор
            .str.replace(r'\.0$', '') # Убираем .0 (например, 15.0 -> 15)
        )
    
    return df_lazy

