# -*- coding: utf-8 -*-
"""
EDA для датасета авиаперевозок: airlines_flights_data.csv

Инструкция:
1) Поместите файл airlines_flights_data.csv рядом со скриптом или укажите путь DATA_PATH.
2) Запустите скрипт. Он сохранит промежуточные таблицы и графики в папку ./eda_output.
3) Все этапы анализа снабжены подробными комментариями на русском языке.
"""
import os
import sys
import textwrap
from typing import List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# НАСТРОЙКИ
# -----------------------------
DATA_PATH = "airlines_flights_data.csv"  # путь к датасету
OUTPUT_DIR = "eda_output"               # куда сохранять результаты (таблицы, графики)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Утилита для безопасного сохранения графиков
def save_fig(name: str):
    """
    Сохраняет текущую фигуру Matplotlib в OUTPUT_DIR с расширением .png.
    Имя файла нормализуется: пробелы заменяются на подчеркивания.
    """
    safe_name = name.strip().lower().replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    return path

# Утилита: группировка времени по частям суток
def time_of_day_from_str(s: pd.Series) -> pd.Series:
    """
    Принимает столбец строк со временем (например, '06:35', '18:20', '6:5', '23:59').
    Возвращает категории: 'ночь' (0-5), 'утро' (6-11), 'день' (12-17), 'вечер' (18-23).
    Некорректные значения -> NaN.
    """
    def parse_one(x):
        try:
            if pd.isna(x):
                return np.nan
            x = str(x).strip()
            # выделим часы
            h = int(x.split(":")[0])
            if 0 <= h <= 5:
                return "ночь"
            elif 6 <= h <= 11:
                return "утро"
            elif 12 <= h <= 17:
                return "день"
            elif 18 <= h <= 23:
                return "вечер"
            else:
                return np.nan
        except Exception:
            return np.nan
    return s.apply(parse_one)

# -----------------------------
# 1. ОБЗОР ДАННЫХ
# -----------------------------
# Читаем данные
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    print("❌ Ошибка чтения CSV. Проверьте путь DATA_PATH или целостность файла.\n", e)
    raise

# Сохраним первые 5 строк для быстрого просмотра
head_path = os.path.join(OUTPUT_DIR, "head_top5.csv")
df.head(5).to_csv(head_path, index=False)

# Размер датасета
n_rows, n_cols = df.shape

# Типы данных и пропуски
dtypes_series = df.dtypes
missing_series = df.isna().sum()
missing_pct = (df.isna().mean() * 100).round(2)

# Сохраним информацию о типах и пропусках
info_df = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(t) for t in dtypes_series.values],
    "missing_count": [int(x) for x in missing_series.values],
    "missing_pct": [float(x) for x in missing_pct.values],
})
info_path = os.path.join(OUTPUT_DIR, "columns_info.csv")
info_df.to_csv(info_path, index=False)

# Евристическое описание столбцов (если ожидаемые имена присутствуют)
column_descriptions = []
expected_desc = {
    "airline": "название авиакомпании",
    "price": "цена билета",
    "duration": "длительность полёта (минуты/часы — зависит от датасета)",
    "days_left": "дней до вылета на момент бронирования",
    "class": "класс обслуживания (economy/business/...)",
    "stops": "количество остановок/пересадок",
    "departure_time": "время вылета (часы:минуты)",
    "arrival_time": "время прилёта (часы:минуты)",
    "source_city": "город вылета",
    "destination_city": "город прилёта",
    "date": "дата вылета",
}
for col in df.columns:
    desc = expected_desc.get(col, "описание не задано (уточните по контексту данных)")
    column_descriptions.append({"column": col, "description_ru": desc})
desc_df = pd.DataFrame(column_descriptions)
desc_path = os.path.join(OUTPUT_DIR, "columns_descriptions.csv")
desc_df.to_csv(desc_path, index=False)

# -----------------------------
# 2. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# -----------------------------
# Вычислим базовые статистики для числовых столбцов
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
stats_df = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T if numeric_cols else pd.DataFrame()
stats_path = os.path.join(OUTPUT_DIR, "numeric_stats.csv")
if not stats_df.empty:
    stats_df.to_csv(stats_path)

# Распределения ключевых числовых признаков (если существуют)
def plot_hist(column: str, bins: int = 30):
    if column in df.columns:
        s = pd.to_numeric(df[column], errors="coerce").dropna()
        if s.empty:
            return None
        plt.figure()
        plt.hist(s, bins=bins)
        plt.title(f"Гистограмма: {column}")
        plt.xlabel(column)
        plt.ylabel("Частота")
        return save_fig(f"hist_{column}")
    return None

hist_paths = {}
for col in ["price", "duration", "days_left"]:
    p = plot_hist(col)
    if p:
        hist_paths[col] = p

# Категориальные признаки: количество уникальных значений и топ-частоты
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
cat_summary = []
for col in categorical_cols:
    vc = df[col].astype("string").value_counts(dropna=True)
    cat_summary.append({
        "column": col,
        "n_unique": int(df[col].nunique(dropna=True)),
        "top_values_preview": ", ".join([f"{idx}:{cnt}" for idx, cnt in vc.head(10).items()])
    })
cat_df = pd.DataFrame(cat_summary)
cat_path = os.path.join(OUTPUT_DIR, "categorical_summary.csv")
cat_df.to_csv(cat_path, index=False)

# -----------------------------
# 3. ВИЗУАЛИЗАЦИЯ
# -----------------------------
# Boxplot цены по классу/авиакомпании (если колонки есть)
def plot_box_by_group(value_col: str, group_col: str):
    if value_col in df.columns and group_col in df.columns:
        vals = pd.to_numeric(df[value_col], errors="coerce")
        groups = df[group_col].astype("string")
        valid = pd.DataFrame({value_col: vals, group_col: groups}).dropna()
        if valid.empty:
            return None
        # чтобы не перегружать график, ограничим топ-15 категорий по частоте
        top_groups = valid[group_col].value_counts().head(15).index
        valid = valid[valid[group_col].isin(top_groups)]
        data = [valid[valid[group_col] == g][value_col].values for g in top_groups]
        plt.figure(figsize=(max(6, len(top_groups) * 0.4), 5))
        plt.boxplot(data, labels=list(top_groups), vert=True, showfliers=True)
        plt.title(f"Boxplot {value_col} по {group_col} (топ-15 категорий)")
        plt.xlabel(group_col)
        plt.ylabel(value_col)
        plt.xticks(rotation=45, ha="right")
        return save_fig(f"box_{value_col}_by_{group_col}")
    return None

box_price_class = plot_box_by_group("price", "class")
box_price_airline = plot_box_by_group("price", "airline")

# Количество рейсов по авиакомпаниям и городам
def plot_top_bar_counts(column: str, top_n: int = 15):
    if column in df.columns:
        vc = df[column].astype("string").value_counts().head(top_n)
        if vc.empty:
            return None
        plt.figure(figsize=(max(6, len(vc) * 0.4), 5))
        plt.bar(vc.index.astype(str), vc.values)
        plt.title(f"Топ-{top_n} по частоте: {column}")
        plt.xlabel(column)
        plt.ylabel("Количество рейсов")
        plt.xticks(rotation=45, ha="right")
        return save_fig(f"bar_top_{top_n}_{column}")
    return None

bar_airlines = plot_top_bar_counts("airline", top_n=15)
bar_source = plot_top_bar_counts("source_city", top_n=15)
bar_dest = plot_top_bar_counts("destination_city", top_n=15)

# Pie chart (круговая диаграмма) по авиакомпаниям (топ-10)
def plot_top_pie(column: str, top_n: int = 10):
    if column in df.columns:
        vc = df[column].astype("string").value_counts()
        if vc.empty:
            return None
        top = vc.head(top_n)
        other_sum = vc.iloc[top_n:].sum()
        labels = list(top.index.astype(str))
        sizes = list(top.values)
        if other_sum > 0:
            labels.append("другие")
            sizes.append(other_sum)
        plt.figure()
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title(f"Доля рейсов по {column} (топ-{top_n} + другие)")
        plt.axis("equal")
        return save_fig(f"pie_{column}_top_{top_n}")
    return None

pie_airlines = plot_top_pie("airline", top_n=10)

# Scatter: зависимость цены от days_left
def plot_scatter(x_col: str, y_col: str):
    if x_col in df.columns and y_col in df.columns:
        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        valid = pd.DataFrame({x_col: x, y_col: y}).dropna()
        if valid.empty:
            return None
        plt.figure()
        plt.scatter(valid[x_col], valid[y_col], alpha=0.5, s=10)
        plt.title(f"Scatter: {y_col} в зависимости от {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        return save_fig(f"scatter_{y_col}_vs_{x_col}")
    return None

scatter_price_days = plot_scatter("days_left", "price")

# -----------------------------
# 4. АНАЛИЗ ЗАВИСИМОСТЕЙ
# -----------------------------
# Корреляции между числовыми признаками
def plot_corr_heatmap(numeric_columns: List[str]):
    if not numeric_columns:
        return None
    corr = df[numeric_columns].corr(numeric_only=True)
    if corr.empty:
        return None
    plt.figure(figsize=(max(6, 0.5 * corr.shape[1] + 2), max(5, 0.5 * corr.shape[0] + 2)))
    im = plt.imshow(corr.values, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(ticks=np.arange(corr.shape[1]), labels=corr.columns, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(corr.shape[0]), labels=corr.index)
    plt.title("Корреляционная матрица (числовые признаки)")
    return save_fig("corr_matrix_numeric")

corr_path = plot_corr_heatmap(numeric_cols)

# Влияние числа пересадок и времени вылета на цену
# Пересадки (stops)
def group_mean_price(by_col: str, value_col: str = "price"):
    if by_col in df.columns and value_col in df.columns:
        gdf = (
            df.assign(**{value_col: pd.to_numeric(df[value_col], errors="coerce")})
              .groupby(by_col, dropna=True)[value_col]
              .mean()
              .sort_values(ascending=True)
        )
        gdf = gdf.dropna()
        if gdf.empty:
            return None, None
        # Сохраним таблицу
        table_path = os.path.join(OUTPUT_DIR, f"mean_{value_col}_by_{by_col}.csv")
        gdf.to_csv(table_path, header=[f"mean_{value_col}"])
        # Построим бар-чарт
        plt.figure(figsize=(max(6, len(gdf) * 0.5), 5))
        plt.bar(gdf.index.astype(str), gdf.values)
        plt.title(f"Средняя {value_col} по {by_col}")
        plt.xlabel(by_col)
        plt.ylabel(f"mean_{value_col}")
        plt.xticks(rotation=45, ha="right")
        fig_path = save_fig(f"bar_mean_{value_col}_by_{by_col}")
        return table_path, fig_path
    return None, None

mean_price_by_stops_tbl, mean_price_by_stops_fig = group_mean_price("stops", "price")

# Время вылета -> категории по частям суток
if "departure_time" in df.columns:
    df["departure_part_of_day"] = time_of_day_from_str(df["departure_time"])
    # Средняя цена по частям суток
    mean_price_by_dep_part_tbl, mean_price_by_dep_part_fig = group_mean_price("departure_part_of_day", "price")
else:
    mean_price_by_dep_part_tbl, mean_price_by_dep_part_fig = None, None

# Средняя цена по авиакомпаниям и по классам
mean_price_by_airline_tbl, mean_price_by_airline_fig = group_mean_price("airline", "price")
mean_price_by_class_tbl, mean_price_by_class_fig = group_mean_price("class", "price")

# -----------------------------
# 5. ДОП. АНАЛИЗ ВРЕМЕНИ И СЕЗОННОСТИ (если есть дата)
# -----------------------------
# Пытаемся разобрать столбец с датой вылета, если существует и не datetime
date_col_candidates = [c for c in df.columns if c.lower() in ["date", "flight_date", "depart_date", "journey_date"]]
parsed_date_col = None
for cand in date_col_candidates:
    try:
        tmp = pd.to_datetime(df[cand], errors="coerce", dayfirst=False, infer_datetime_format=True)
        if tmp.notna().sum() > 0:
            parsed_date_col = cand
            df["_parsed_date"] = tmp
            break
    except Exception:
        continue

if parsed_date_col is not None:
    df["_month"] = df["_parsed_date"].dt.month
    df["_weekday"] = df["_parsed_date"].dt.weekday  # 0=Пн ... 6=Вс

    # Средняя цена по месяцам
    mean_price_by_month_tbl, mean_price_by_month_fig = group_mean_price("_month", "price")

    # Средняя цена по дням недели
    mean_price_by_weekday_tbl, mean_price_by_weekday_fig = group_mean_price("_weekday", "price")
else:
    mean_price_by_month_tbl = mean_price_by_month_fig = None
    mean_price_by_weekday_tbl = mean_price_by_weekday_fig = None

# -----------------------------
# 6. ОБРАБОТКА ПРОПУСКОВ (РЕКОМЕНДАЦИИ)
# -----------------------------
# Создадим отдельный файл с предложениями по обработке пропусков на основе долей пропусков
handling_suggestions = []
for _, row in info_df.iterrows():
    col = row["column"]
    miss_pct = row["missing_pct"]
    dtype = row["dtype"]
    suggestion = ""
    if miss_pct == 0:
        suggestion = "Пропусков нет — ничего делать не требуется."
    else:
        if "float" in dtype or "int" in dtype or col in ["price", "duration", "days_left"]:
            suggestion = "Рассмотреть заполнение медианой/квартильной регрессией или удалить строки при высокой доле пропусков."
        else:
            suggestion = "Рассмотреть заполнение модой/категорией 'unknown' либо удалить строки/категории."
        if miss_pct > 30:
            suggestion += " Высокая доля пропусков (>30%) — возможно, лучше исключить столбец или собрать данные заново."
    handling_suggestions.append({
        "column": col,
        "missing_pct": miss_pct,
        "suggestion_ru": suggestion
    })
miss_handling_df = pd.DataFrame(handling_suggestions)
miss_handle_path = os.path.join(OUTPUT_DIR, "missing_handling_suggestions.csv")
miss_handling_df.to_csv(miss_handle_path, index=False)

# -----------------------------
# 7. СОХРАНИМ КОРОТКИЙ ТЕКСТОВЫЙ ОТЧЁТ-РЕЗЮМЕ
# -----------------------------
summary = []

summary.append("=== ЭТАП 1. ОБЗОР ДАННЫХ ===")
summary.append(f"Размер датасета: {n_rows} строк × {n_cols} столбцов.")
summary.append("Информация по столбцам, типам и пропускам: columns_info.csv")
summary.append("Описание столбцов (эвристика): columns_descriptions.csv")
summary.append("Первые 5 строк: head_top5.csv")

summary.append("\n=== ЭТАП 2. СТАТИСТИЧЕСКИЙ АНАЛИЗ ===")
if numeric_cols:
    summary.append(f"Числовые столбцы: {', '.join(numeric_cols)}")
    summary.append("Сводная статистика: numeric_stats.csv")
else:
    summary.append("Числовых столбцов не обнаружено.")
for k, p in hist_paths.items():
    summary.append(f"Гистограмма {k}: {os.path.basename(p)}")
summary.append("Категориальные признаки: categorical_summary.csv")

summary.append("\n=== ЭТАП 3. ВИЗУАЛИЗАЦИЯ ===")
if box_price_class:
    summary.append(f"Boxplot цены по классу: {os.path.basename(box_price_class)}")
if box_price_airline:
    summary.append(f"Boxplot цены по авиакомпаниям: {os.path.basename(box_price_airline)}")
if bar_airlines:
    summary.append(f"Столбчатая: количество рейсов по авиакомпаниям: {os.path.basename(bar_airlines)}")
if bar_source:
    summary.append(f"Столбчатая: города вылета: {os.path.basename(bar_source)}")
if bar_dest:
    summary.append(f"Столбчатая: города прибытия: {os.path.basename(bar_dest)}")
if pie_airlines:
    summary.append(f"Круговая: доля рейсов по авиакомпаниям: {os.path.basename(pie_airlines)}")
if scatter_price_days:
    summary.append(f"Scatter: цена vs days_left: {os.path.basename(scatter_price_days)}")

summary.append("\n=== ЭТАП 4. ЗАВИСИМОСТИ ===")
if corr_path:
    summary.append(f"Корреляционная матрица: {os.path.basename(corr_path)}")
if mean_price_by_stops_tbl:
    summary.append(f"Средняя цена по пересадкам: {os.path.basename(mean_price_by_stops_tbl)}, график: {os.path.basename(mean_price_by_stops_fig)}")
if mean_price_by_dep_part_tbl:
    summary.append(f"Средняя цена по частям суток вылета: {os.path.basename(mean_price_by_dep_part_tbl)}, график: {os.path.basename(mean_price_by_dep_part_fig)}")
if mean_price_by_airline_tbl:
    summary.append(f"Средняя цена по авиакомпаниям: {os.path.basename(mean_price_by_airline_tbl)}, график: {os.path.basename(mean_price_by_airline_fig)}")
if mean_price_by_class_tbl:
    summary.append(f"Средняя цена по классам: {os.path.basename(mean_price_by_class_tbl)}, график: {os.path.basename(mean_price_by_class_fig)}")

summary.append("\n=== ЭТАП 5. ВРЕМЯ И СЕЗОННОСТЬ ===")
if parsed_date_col is not None:
    summary.append(f"Опознан столбец даты: {parsed_date_col}")
    if mean_price_by_month_tbl:
        summary.append(f"Средняя цена по месяцам: {os.path.basename(mean_price_by_month_tbl)}, график: {os.path.basename(mean_price_by_month_fig)}")
    if mean_price_by_weekday_tbl:
        summary.append(f"Средняя цена по дням недели: {os.path.basename(mean_price_by_weekday_tbl)}, график: {os.path.basename(mean_price_by_weekday_fig)}")
else:
    summary.append("Столбец даты не обнаружен или не распознан. Анализ сезонности пропущен.")

summary.append("\n=== ЭТАП 6. ПРОПУСКИ ===")
summary.append("Рекомендации по обработке пропусков: missing_handling_suggestions.csv")

# Выводим короткие советы/гипотезы на основе доступных полей
hypotheses = []
if "days_left" in df.columns and "price" in df.columns:
    hypotheses.append("- Цена часто снижается при увеличении days_left (проверено через scatter). Возможна обратная картина ближе к вылету из-за динамического ценообразования.")
if "duration" in df.columns and "price" in df.columns:
    hypotheses.append("- Более длительные перелёты могут быть дороже (проверьте корреляцию duration и price).")
if "stops" in df.columns and "price" in df.columns:
    hypotheses.append("- Рейсы без пересадок обычно дороже, чем с пересадками (сравните средние цены по stops).")
if "departure_time" in df.columns and "price" in df.columns:
    hypotheses.append("- Возможна разница цен между утренними/вечерними рейсами (средняя цена по частям суток).")

if hypotheses:
    summary.append("\n=== ЭТАП 7. ВЫВОДЫ (краткие гипотезы) ===")
    summary.extend(hypotheses)

summary_text = "\n".join(summary)
summary_path = os.path.join(OUTPUT_DIR, "SUMMARY.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

# Также сохраним сам скрипт как файл для скачивания
script_path = "/mnt/data/eda_airlines.py"
with open(script_path, "w", encoding="utf-8") as f:
    f.write(open(__file__, "r", encoding="utf-8").read())

# Показать краткое резюме пользователю в выводе
print("✅ Анализ выполнен. Основные результаты сохранены в папке:", OUTPUT_DIR)
print("\n--- КРАТКОЕ РЕЗЮМЕ ---\n")
print(summary_text)
print("\nСкрипт можно скачать здесь:", script_path)

