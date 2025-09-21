import os
import pandas as pd
from prophet import Prophet

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def _find_col(df, name):
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    return None

def forecast(file, column, years=10):
    """
    Robust forecast:
    - Agar CSV time-series (year + column) bo'lsa Prophet bilan forecast qaytaradi.
    - Agar CSV single-row (Current_Value) bo'lsa -> years davomida constant forecast (flat).
    Returns DataFrame with columns ['ds','yhat','yhat_lower','yhat_upper'].
    """
    filepath = os.path.join(DATA_DIR, file)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fayl topilmadi: {filepath}")

    df = pd.read_csv(filepath)

    # 1) time-series rejimi
    year_col = _find_col(df, 'year') or _find_col(df, 'Year') or _find_col(df, 'ds')
    metric_col = _find_col(df, column)
    if year_col and metric_col:
        ts = df[[year_col, metric_col]].dropna()
        ts = ts.rename(columns={year_col: 'ds', metric_col: 'y'})
        # try parse year-only format first, otherwise parse generically
        try:
            ts['ds'] = pd.to_datetime(ts['ds'].astype(str), format='%Y')
        except Exception:
            ts['ds'] = pd.to_datetime(ts['ds'], errors='coerce')
        ts = ts.dropna(subset=['ds', 'y'])
        if ts.empty:
            raise ValueError(f"No valid time-series rows in {file} for column {column}")
        model = Prophet()
        model.fit(ts)
        future = model.make_future_dataframe(periods=years, freq='Y')
        forecast_df = model.predict(future)
        return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # 2) single-value rejimi (Current_Value)
    cur_col = _find_col(df, 'current_value') or _find_col(df, 'current') or _find_col(df, 'value')
    if cur_col:
        cur_val = float(df[cur_col].iloc[-1])
        start_year = pd.Timestamp.now().year
        ds = pd.date_range(start=f'{start_year}', periods=years+1, freq='Y')
        yhat = [cur_val] * (years+1)
        res = pd.DataFrame({
            'ds': ds,
            'yhat': yhat,
            'yhat_lower': yhat,
            'yhat_upper': yhat
        })
        return res

    raise ValueError(f"File {file} does not contain a time-series '{column}' or a 'Current_Value' column.")
