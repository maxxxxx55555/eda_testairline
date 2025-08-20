import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from statsmodels.nonparametric.smoothers_lowess import lowess

# Папка для сохранения картинок (создаётся рядом с кодом)
out_dir = 'report_images'
os.makedirs(out_dir, exist_ok=True)

# Загружаем CSV из той же папки, где лежит код
file_path = 'airlines_flights_data.csv'
df = pd.read_csv(file_path)

# Очистка строковых признаков
for col in ['airline','flight','source_city','departure_time',
            'stops','arrival_time','destination_city','class']:
    df[col] = df[col].astype(str).str.strip()

# Создание дополнительных признаков
df['route'] = df['source_city'] + ' → ' + df['destination_city']
df['nonstop'] = (df['stops'] == 'zero').astype(int)
df['red_eye'] = df['departure_time'].isin(['Late_Night','Early_Morning']).astype(int)

sns.set_theme(style='whitegrid', font_scale=0.9)

# 1) Гистограммы числовых
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
nums = ['price','duration','days_left']
titles = ['Распределение цены (р.)',
          'Распределение длительности (часы)',
          'Распределение дней до вылета (дни)']
for ax, col, title in zip(axes, nums, titles):
    sns.histplot(df[col], bins=50, ax=ax, kde=False)
    ax.set_title(title)
    ax.set_xlabel('Значение')
    ax.set_ylabel('Количество')
    median = df[col].median()
    q95 = df[col].quantile(0.95)
    ax.axvline(median, color='red', linestyle='--', label=f'Медиана = {median:.1f}')
    ax.axvline(q95, color='green', linestyle=':', label=f'95-й персентиль = {q95:.1f}')
    ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'hist_numeric.png'), dpi=300)
plt.close(fig)

# 2) Обрезка и лог-распределение цены
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
price = df['price']
lower = price.quantile(0.01)
upper = price.quantile(0.99)
trimmed = price[(price >= lower) & (price <= upper)]
sns.histplot(trimmed, bins=50, ax=axes[0])
axes[0].set_title('Цена после обрезки 1-99%')
axes[0].set_xlabel('Цена (р.)')
axes[0].set_ylabel('Количество')
axes[0].axvline(trimmed.median(), color='red', linestyle='--',
                label=f'Медиана = {trimmed.median():.1f}')
axes[0].legend()

log_price = np.log1p(price)
sns.histplot(log_price, bins=50, ax=axes[1])
axes[1].set_title('Распределение log(1+price)')
axes[1].set_xlabel('log(1+price)')
axes[1].set_ylabel('Количество')
axes[1].axvline(log_price.median(), color='red', linestyle='--',
                label=f'Медиана = {log_price.median():.2f}')
axes[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'price_log_trim.png'), dpi=300)
plt.close(fig)

# 3) Горизонтальные бары категорий
categorical_plots = [
    ('airline', 'Авиалиния', None),
    ('source_city', 'Город отправления', None),
    ('destination_city', 'Город назначения', None),
    ('departure_time', 'Время вылета', None),
    ('arrival_time', 'Время прилёта', None),
    ('stops', 'Число пересадок', None),
    ('class', 'Класс обслуживания', None),
    ('flight', 'Код рейса', 10)
]
fig, axes = plt.subplots(4, 2, figsize=(14, 18))
for ax, (col, title, top_n) in zip(axes.flatten(), categorical_plots):
    vc = df[col].value_counts()
    if top_n:
        vc = vc.head(top_n)
    vc = vc.sort_values()
    share = vc / len(df)
    ax.barh(vc.index, vc.values)
    ax.set_title(title)
    ax.set_xlabel('Количество')
    for i, (val, pct) in enumerate(zip(vc.values, share.values)):
        ax.text(val, i, f'{pct*100:.1f}%', va='center', ha='left', fontsize=7)
    ax.set_ylabel('')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'bar_categories.png'), dpi=300)
plt.close(fig)

# 4) Boxplot цена по классу/пересадкам/авиалиниям (исправленное palette+legend)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# a) class
sns.boxplot(
    data=df, x='class', y='price',
    hue='class', dodge=False,  # <— добавили hue, отключим легенду
    showfliers=False, ax=axes[0], palette='pastel'
)
axes[0].set_yscale('log')
axes[0].set_title('Цена по классу')
axes[0].set_xlabel('Класс')
axes[0].set_ylabel('Цена (лог масштаб)')
leg = axes[0].get_legend()
if leg is not None:
    leg.remove()

# b) stops
sns.boxplot(
    data=df, x='stops', y='price',
    hue='stops', dodge=False,
    showfliers=False, ax=axes[1],
    order=['zero','one','two_or_more'], palette='pastel'
)
axes[1].set_yscale('log')
axes[1].set_title('Цена по числу пересадок')
axes[1].set_xlabel('Пересадки')
axes[1].set_ylabel('Цена (лог масштаб)')
leg = axes[1].get_legend()
if leg is not None:
    leg.remove()

# c) airline
sns.boxplot(
    data=df, x='airline', y='price',
    hue='airline', dodge=False,
    showfliers=False, ax=axes[2], palette='pastel'
)
axes[2].set_yscale('log')
axes[2].set_title('Цена по авиалиниям')
axes[2].set_xlabel('Авиалиния')
axes[2].set_ylabel('Цена (лог масштаб)')
leg = axes[2].get_legend()
if leg is not None:
    leg.remove()

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'price_by_factors.png'), dpi=300)
plt.close(fig)

# 5) Цена по времени вылета/прилёта (тут hue уже есть — всё ок)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x='departure_time', y='price',
            hue='class', showfliers=False,
            palette='Pastel1', ax=axes[0])
axes[0].set_yscale('log')
axes[0].set_title('Цена по времени вылета')
axes[0].set_xlabel('Время вылета')
axes[0].set_ylabel('Цена (лог масштаб)')
axes[0].legend(title='Класс', loc='upper right')

sns.boxplot(data=df, x='arrival_time', y='price',
            hue='class', showfliers=False,
            palette='Pastel1', ax=axes[1])
axes[1].set_yscale('log')
axes[1].set_title('Цена по времени прилёта')
axes[1].set_xlabel('Время прилёта')
axes[1].set_ylabel('Цена (лог масштаб)')
axes[1].legend(title='Класс', loc='upper right')

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'price_by_departure_arrival.png'), dpi=300)
plt.close(fig)

# 6) Медианные цены по топ-10 маршрутам (маршрут × класс)
route_counts = df['route'].value_counts().head(10)
top_routes = route_counts.index
subset = df[df['route'].isin(top_routes)]
agg = subset.groupby(['route','class'], observed=False)['price'].median().reset_index()

route_order = list(top_routes)
fig, ax = plt.subplots(figsize=(14, 5))
sns.barplot(data=agg, x='route', y='price',
            hue='class', order=route_order,
            palette='pastel', ax=ax)
ax.set_yscale('log')
ax.set_title('Медианная цена по топ-10 маршрутам')
ax.set_xlabel('Маршрут')
ax.set_ylabel('Медианная цена (лог масштаб)')
# БЕЗ set_xticklabels — безопасный поворот:
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Класс')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'price_by_route_top10.png'), dpi=300)
plt.close(fig)

# 7) Scatter: days_left vs price
sample = df.sample(n=30000, random_state=42)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
for ax, cls in zip(axes, ['Economy','Business']):
    sub = sample[sample['class'] == cls]
    sns.scatterplot(data=sub, x='days_left', y='price', alpha=0.2, s=10, ax=ax, color='grey')
    med = df[df['class'] == cls].groupby('days_left', observed=False)['price'].median().reset_index()
    ax.plot(med['days_left'], med['price'], color='red', linewidth=2)
    ax.set_yscale('log')
    ax.set_title(f'Цена vs дни до вылета ({cls})')
    ax.set_xlabel('Дни до вылета')
    ax.set_ylabel('Цена (лог масштаб)')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'scatter_days_price.png'), dpi=300)
plt.close(fig)

# 8) Медиана цены по бинам days_left (разбивки: класс, пересадки)
bins_labels = ['1-3','4-7','8-14','15-21','22-35','36-49']
df['days_bin'] = pd.cut(df['days_left'], bins=[0,3,7,14,21,35,49],
                        labels=bins_labels, include_lowest=True, right=True)

bin_class = df.groupby(['days_bin','class'], observed=False)['price'].median().reset_index()
bin_class['days_bin'] = pd.Categorical(bin_class['days_bin'], categories=bins_labels, ordered=True)

bin_stops = df.groupby(['days_bin','stops'], observed=False)['price'].median().reset_index()
bin_stops['days_bin'] = pd.Categorical(bin_stops['days_bin'], categories=bins_labels, ordered=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=bin_class, x='days_bin', y='price', hue='class', palette='pastel', ax=axes[0])
axes[0].set_yscale('log')
axes[0].set_title('Медианная цена по дням до вылета и классу')
axes[0].set_xlabel('Бины дней до вылета')
axes[0].set_ylabel('Медианная цена (лог)')
axes[0].legend(title='Класс')

sns.barplot(data=bin_stops, x='days_bin', y='price', hue='stops', palette='pastel', ax=axes[1],
            order=bins_labels)
axes[1].set_yscale('log')
axes[1].set_title('Медианная цена по дням до вылета и пересадкам')
axes[1].set_xlabel('Бины дней до вылета')
axes[1].set_ylabel('Медианная цена (лог)')
axes[1].legend(title='Пересадки')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'price_bins.png'), dpi=300)
plt.close(fig)

# 9) Scatter: duration vs price (раскраска по пересадкам)
sample2 = df.sample(n=30000, random_state=0)
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=sample2, x='duration', y='price', hue='stops', alpha=0.3, s=20, ax=ax)
sample2['duration_int'] = sample2['duration'].round(0).astype(int)
median_by_dur = sample2.groupby('duration_int', observed=False)['price'].median().reset_index()
ax.plot(median_by_dur['duration_int'], median_by_dur['price'], color='black', linewidth=2, label='Медиана по длительности')
ax.set_yscale('log')
ax.set_title('Цена vs длительность полёта (цвет – пересадки)')
ax.set_xlabel('Длительность (часы)')
ax.set_ylabel('Цена (лог масштаб)')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'duration_vs_price.png'), dpi=300)
plt.close(fig)

# 10) Тепловые карты и корреляции
airline_class = df.pivot_table(index='airline', columns='class', values='price', aggfunc='median', observed=False)
route_class = subset.pivot_table(index='route', columns='class', values='price', aggfunc='median', observed=False).loc[route_order]
spearman = df[['price','duration','days_left']].corr(method='spearman')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.heatmap(airline_class, annot=True, fmt='.0f', cmap='YlGnBu', ax=axes[0])
axes[0].set_title('Медианная цена: авиалиния × класс')
sns.heatmap(route_class, annot=True, fmt='.0f', cmap='YlGnBu', ax=axes[1])
axes[1].set_title('Медианная цена: маршрут (топ 10) × класс')
sns.heatmap(spearman, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=axes[2])
axes[2].set_title('Корреляции (Спирмен)')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'heatmaps_correlations.png'), dpi=300)
plt.close(fig)

# 11) Boxplot: цена по пересадкам с разбивкой по классу (hue есть — всё ок)
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x='stops', y='price', hue='class',
            showfliers=False, palette='pastel',
            order=['zero','one','two_or_more'], ax=ax)
ax.set_yscale('log')
ax.set_title('Цена по пересадкам с разбивкой по классу')
ax.set_xlabel('Пересадки')
ax.set_ylabel('Цена (лог масштаб)')
ax.legend(title='Класс')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'price_by_stops_class.png'), dpi=300)
plt.close(fig)
