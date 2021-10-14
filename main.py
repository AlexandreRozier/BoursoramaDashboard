# %%
from datetime import datetime, timedelta
from random import randint

import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash import html
from dash.dcc import Dropdown, Graph
from dash.dependencies import Input, Output
# Preprocess data
from matplotlib.cm import get_cmap

df = pd.read_csv('export-operations-13-10-2021_14-27-42.csv', sep=";")
df['amount'] = df.amount.str.replace(',', '.').str.replace(" ", "").astype(float)
df.loc[df.amount > 2000, 'category'] = 'salaire'
df.dateVal = df.dateVal.apply(pd.to_datetime)

df['year'] = df.dateVal.dt.year
df['month'] = df.dateVal.dt.month
df.sort_values('dateVal', inplace=True)

app = dash.Dash(__name__)

years = df.year.unique()

cmap = get_cmap('Spectral')
categories = df.category.unique()
colors = cmap(np.linspace(0, 1, len(categories)))
color_discrete_map = {cat: f'rgb({colors[i][0]},' \
                           f'{colors[i][1]},' \
                           f'{colors[i][2]})' for i, cat in enumerate(categories)}

# Dash app
app.layout = html.Div([
    Dropdown(
        id="dropdown",
        options=[{"label": x, "value": x} for x in years],
        value=sorted(years)[-1],
        clearable=False,
    ),
    Graph(id="balance-ts", style={'width': '100vw', 'height': '70vh', 'display': 'inline-block'}),
    *[Graph(id=f"bar-chart-{13 - i}", style={'width': '40vw', 'height': '100vh', 'display': 'inline-block'}) for i in
      range(1, 13)],

])


@app.callback(
    Output("balance-ts", "figure"),
    [Input("dropdown", "value")])
def update_balance(year):
    mask = (df["year"] == year)
    masked_df = df.loc[mask]
    masked_df.dateVal = masked_df.dateVal.apply(lambda d: d + timedelta(hours=randint(0, 24),
                                                                        minutes=randint(0, 60)))

    fig = px.line(masked_df, x="dateVal", y="accountbalance", markers='category',
                  title="Balance over the year",
                  hover_data=["amount", "label"], )
    min_month = masked_df.dateVal.dt.month.min()
    max_month = masked_df.dateVal.dt.month.max()

    fig.update_layout(
        shapes=[dict(
            x0=datetime(year=year, month=month, day=1), x1=datetime(year=year, month=month, day=1), y0=0, y1=1,
            xref='x', yref='paper', line_dash='dash',
            line_width=1) for month in range(min_month, max_month + 1)],
        # annotations=[dict(x=linedate, y=0.8, xref='x', yref='paper', font=dict(
        #     color="blue", size=14),
        #                   showarrow=False, xanchor='left', text='Vertical line between two dates')]
    )
    fig.update_traces(textposition="bottom right")
    return fig


months = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Aout', 'Septembre', 'Octobre', 'Novembre',
          'Décembre']


def toggle_expense_barplot(month):
    @app.callback(
        Output(f"bar-chart-{month}", "style"),
        [Input("dropdown", "value")])
    def toggle(year):
        max_month = df[df.year == year].dateVal.dt.month.max()
        if not max_month or max_month < month:
            return {'display': 'none'}
        else:
            return {'display': 'inline-block'}

    return toggle


def update_expense_barplots(month):
    @app.callback(
        Output(f"bar-chart-{month}", "figure"),
        [Input("dropdown", "value")])
    def update(year):
        mask = (df["year"] == year) & (df['month'] == month)
        masked_df = df.loc[mask]
        min_amount = df[df.year == year].amount.min()
        max_amount = df[df.year == year].amount.max()
        fig = px.bar(masked_df, x="category", y="amount",
                     text='amount', color='category',
                     color_discrete_map=color_discrete_map, title=f"{months[month - 1]} {year}",

                     hover_data=["amount", "label"])
        fig.update_traces(texttemplate='%{text:.2s}', )
        fig.update_layout(uniformtext_minsize=8,
                          showlegend=False,
                          uniformtext_mode='hide',
                          yaxis_range=[min_amount, max_amount]
                          )
        fig.update_yaxes(matches=None)
        return fig

    return update


for j in range(1, 13):
    toggle_expense_barplot(j)
    update_expense_barplots(j)
app.run_server(debug=True, use_reloader=True)

# %%
