import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ── DB Connection ────────────────────────────────────────────────────────────
conn = sqlite3.connect("hantavirus.db", check_same_thread=False)

# ── Data Queries ─────────────────────────────────────────────────────────────

# Chart 1 & 2: Severity / Fatality by Age & Gender
severity_dist = pd.read_sql_query("""
    SELECT age_group, gender,
        CASE
            WHEN severity = 'Mild'     THEN 0
            WHEN severity = 'Moderate' THEN 1
            WHEN severity = 'Severe'   THEN 2
            WHEN severity = 'Critical' THEN 3
        END AS severity_score
    FROM hv_clinical
""", conn)

fatality_age = pd.read_sql_query("""
    SELECT age_group, gender,
        SUM(CASE WHEN outcome = 'Recovered' THEN 1 ELSE 0 END) AS recovered,
        SUM(CASE WHEN outcome = 'Deceased'  THEN 1 ELSE 0 END) AS deceased
    FROM hv_clinical
    GROUP BY age_group, gender
""", conn)
fatality_age["fatality_rate"] = fatality_age["deceased"] / (fatality_age["recovered"] + fatality_age["deceased"])

# Chart 3: Severity by Strain
severity_by_strain = pd.read_sql_query("""
    SELECT COUNT(severity) AS severity_count, severity, virus_strain
    FROM hv_clinical
    GROUP BY severity, virus_strain
    ORDER BY virus_strain,
        CASE severity WHEN 'Mild' THEN 1 WHEN 'Moderate' THEN 2
                      WHEN 'Severe' THEN 3 WHEN 'Critical' THEN 4 END
""", conn)

# Chart 4: Fatality by Strain
deceased_by_strain = pd.read_sql_query("""
    SELECT virus_strain,
        SUM(CASE WHEN outcome = 'Recovered' THEN 1 ELSE 0 END) AS recovered,
        SUM(CASE WHEN outcome = 'Deceased'  THEN 1 ELSE 0 END) AS deceased
    FROM hv_clinical
    GROUP BY virus_strain ORDER BY virus_strain
""", conn)
deceased_by_strain["fatality_rate"] = deceased_by_strain["deceased"] / (deceased_by_strain["recovered"] + deceased_by_strain["deceased"])
deceased_by_strain["recovery_rate"] = 1 - deceased_by_strain["fatality_rate"]

# Chart 5: Fatality by Symptom
fatality_by_symptoms = pd.read_sql_query("""
    SELECT 'Fever' AS symptom, fever AS present,
        SUM(CASE WHEN outcome='Recovered' THEN 1 ELSE 0 END) AS recovered,
        SUM(CASE WHEN outcome='Deceased'  THEN 1 ELSE 0 END) AS deceased
    FROM hv_clinical WHERE fever IS NOT NULL AND fever = 1.0 GROUP BY fever
    UNION ALL
    SELECT 'Myalgia', myalgia,
        SUM(CASE WHEN outcome='Recovered' THEN 1 ELSE 0 END),
        SUM(CASE WHEN outcome='Deceased'  THEN 1 ELSE 0 END)
    FROM hv_clinical WHERE myalgia IS NOT NULL AND myalgia = 1.0 GROUP BY myalgia
    UNION ALL
    SELECT 'Blurred Vision', blurred_vision,
        SUM(CASE WHEN outcome='Recovered' THEN 1 ELSE 0 END),
        SUM(CASE WHEN outcome='Deceased'  THEN 1 ELSE 0 END)
    FROM hv_clinical WHERE blurred_vision IS NOT NULL AND blurred_vision = 1.0 GROUP BY blurred_vision
    UNION ALL
    SELECT 'Hypotension', hypotension,
        SUM(CASE WHEN outcome='Recovered' THEN 1 ELSE 0 END),
        SUM(CASE WHEN outcome='Deceased'  THEN 1 ELSE 0 END)
    FROM hv_clinical WHERE hypotension IS NOT NULL AND hypotension = 1.0 GROUP BY hypotension
    UNION ALL
    SELECT 'Dyspnea', dyspnea,
        SUM(CASE WHEN outcome='Recovered' THEN 1 ELSE 0 END),
        SUM(CASE WHEN outcome='Deceased'  THEN 1 ELSE 0 END)
    FROM hv_clinical WHERE dyspnea IS NOT NULL AND dyspnea = 1.0 GROUP BY dyspnea
    UNION ALL
    SELECT 'Hemorrhage', hemorrhage,
        SUM(CASE WHEN outcome='Recovered' THEN 1 ELSE 0 END),
        SUM(CASE WHEN outcome='Deceased'  THEN 1 ELSE 0 END)
    FROM hv_clinical WHERE hemorrhage IS NOT NULL AND hemorrhage = 1.0 GROUP BY hemorrhage
""", conn)
fatality_by_symptoms["fatality_rate"]  = fatality_by_symptoms["deceased"]  / (fatality_by_symptoms["recovered"] + fatality_by_symptoms["deceased"])
fatality_by_symptoms["recovery_rate"]  = 1 - fatality_by_symptoms["fatality_rate"]

# Charts 6 & 7: Cases by Country
total_cases = pd.read_sql_query("""
    SELECT country, SUM(confirmed_cases) AS total_cases, SUM(deaths) AS total_deaths
    FROM hv_country_yearly GROUP BY country ORDER BY total_cases DESC LIMIT 10
""", conn)
total_cases["fatality_rate"] = total_cases["total_deaths"] / total_cases["total_cases"]

country_cases = pd.read_sql_query("""
    SELECT year, country, confirmed_cases FROM hv_country_yearly
    WHERE year >= 2015 AND confirmed_cases > 100 ORDER BY country, year
""", conn)

# Chart 8: Strain map
strains_regions = pd.read_sql_query("""
    SELECT DISTINCT a.virus_strain AS virus_strain,
        CASE a.country_iso3
            WHEN 'ARG' THEN 'Argentina' WHEN 'CHL' THEN 'Chile'
            WHEN 'BRA' THEN 'Brazil'    WHEN 'PAN' THEN 'Panama'
            WHEN 'GRC' THEN 'Greece'    WHEN 'HRV' THEN 'Croatia'
            WHEN 'SVN' THEN 'Slovenia'  WHEN 'CHN' THEN 'China'
            WHEN 'KOR' THEN 'South Korea' WHEN 'PRY' THEN 'Paraguay'
            WHEN 'DEU' THEN 'Germany'   WHEN 'FIN' THEN 'Finland'
            WHEN 'SWE' THEN 'Sweden'    WHEN 'USA' THEN 'United States'
            ELSE NULL END AS country,
        COUNT(a.virus_strain) AS case_count, b.latitude, b.longitude
    FROM hv_clinical AS a
    JOIN (SELECT DISTINCT iso3, latitude, longitude FROM hv_country_yearly) AS b
      ON a.country_iso3 = b.iso3
    GROUP BY a.virus_strain, a.country_iso3 ORDER BY a.virus_strain
""", conn)
jitter = 0.8
strains_regions["lat_jitter"] = strains_regions["latitude"] + np.random.uniform(-jitter, jitter, len(strains_regions))
strains_regions["lon_jitter"] = strains_regions["longitude"] + np.random.uniform(-jitter, jitter, len(strains_regions))

# Chart 9: Fatality by country
fatality_by_country = pd.read_sql_query("""
    SELECT country, SUM(confirmed_cases) AS total_cases, SUM(deaths) AS total_deaths,
        CAST(SUM(deaths) AS FLOAT) / SUM(confirmed_cases) AS fatality_rate
    FROM hv_country_yearly GROUP BY country ORDER BY fatality_rate DESC
""", conn)

# Charts 10 & 11: Seasonal
seasonal_peaks = pd.read_sql_query("""
    SELECT iso3,
        CASE month WHEN 1 THEN 'Jan' WHEN 2 THEN 'Feb' WHEN 3 THEN 'Mar'
            WHEN 4 THEN 'Apr' WHEN 5 THEN 'May' WHEN 6 THEN 'Jun'
            WHEN 7 THEN 'Jul' WHEN 8 THEN 'Aug' WHEN 9 THEN 'Sep'
            WHEN 10 THEN 'Oct' WHEN 11 THEN 'Nov' WHEN 12 THEN 'Dec'
        END AS month_name, month, SUM(cases) AS monthly_cases
    FROM hv_monthly_trends GROUP BY iso3, month ORDER BY iso3, month
""", conn)

cases_by_month = pd.read_sql_query("""
    SELECT SUM(cases) AS monthly_cases, iso3, month FROM hv_monthly_trends
    GROUP BY month, iso3 HAVING NOT iso3='CHN' AND NOT iso3='FIN' ORDER BY month
""", conn)

# Chart 12: Incidence by year
incidence_year = pd.read_sql_query("""
    SELECT year, SUM(confirmed_cases) AS cases, SUM(deaths) AS deaths
    FROM hv_country_yearly GROUP BY year ORDER BY year
""", conn)

# Charts 13 & 14: Strain transmissibility
strains_transmission = pd.read_sql_query("""
    SELECT b.year_identified, b.strain_name, b.geographic_range,
        SUM(CASE WHEN a.outcome='Recovered' THEN 1 ELSE 0 END) AS recovered,
        SUM(CASE WHEN a.outcome='Deceased'  THEN 1 ELSE 0 END) AS deceased,
        COUNT(a.virus_strain) AS cases
    FROM hv_clinical AS a
    JOIN (SELECT DISTINCT year_identified, strain_name, geographic_range FROM hv_virus_strains) AS b
      ON a.virus_strain = b.strain_name
    GROUP BY a.virus_strain ORDER BY year_identified, a.virus_strain
""", conn)
strains_transmission["fatality_rate"] = strains_transmission["deceased"] / strains_transmission["cases"]
strains_transmission["recovery_rate"] = 1 - strains_transmission["fatality_rate"]

# ── Theme ─────────────────────────────────────────────────────────────────────
BG       = "#0f0f0f"
CARD_BG  = "#161616"
BORDER   = "#2a2a2a"
TEXT     = "#e8e8e8"
MUTED    = "#888888"
ACCENT   = "#00c896"
ACCENT2  = "#ff6b6b"
FONT     = "DM Mono, monospace"

plotly_theme = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor="#1a1a1a",
    font=dict(color=TEXT, family=FONT, size=12),
    title_font=dict(color=TEXT, family=FONT, size=14),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
    xaxis=dict(gridcolor="#2a2a2a", linecolor=BORDER, tickcolor=MUTED),
    yaxis=dict(gridcolor="#2a2a2a", linecolor=BORDER, tickcolor=MUTED),
    margin=dict(t=50, l=50, r=20, b=50),
    colorway=[ACCENT, ACCENT2, "#f4c542", "#6ec6ff", "#c084fc", "#fb923c", "#34d399", "#f472b6"],
)

def apply_theme(fig):
    fig.update_layout(**plotly_theme)
    return fig

# ── Build Figures ─────────────────────────────────────────────────────────────

def fig_severity_heatmap():
    pivot = severity_dist.pivot_table(index="gender", columns="age_group", values="severity_score")
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,"#1a3a2a"],[1, ACCENT]],
        showscale=True, colorbar=dict(title="Severity Score", tickfont=dict(color=TEXT))
    ))
    fig.update_layout(title="Severity by Age Group & Gender", **plotly_theme)
    return fig

def fig_fatality_heatmap():
    pivot = fatality_age.pivot_table(index="gender", columns="age_group", values="fatality_rate")
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,"#1a1a2a"],[1, ACCENT2]],
        showscale=True, colorbar=dict(title="Fatality Rate", tickfont=dict(color=TEXT)),
        hovertemplate="%{x}<br>%{y}<br>Rate: %{z:.1%}<extra></extra>"
    ))
    fig.update_layout(title="Fatality Rate by Age Group & Gender", **plotly_theme)
    return fig

def fig_severity_by_strain():
    fig = px.bar(severity_by_strain, x="virus_strain", y="severity_count", color="severity",
                 barmode="group",
                 color_discrete_map={"Mild":"#34d399","Moderate":"#f4c542","Severe":ACCENT2,"Critical":"#dc2626"},
                 labels={"virus_strain":"Virus Strain","severity_count":"Cases","severity":"Severity"})
    fig.update_layout(title="Severity Distribution by Strain", **plotly_theme)
    return fig

def fig_fatality_by_strain():
    fig = go.Figure()
    fig.add_bar(name="Recovery Rate", x=deceased_by_strain["virus_strain"], y=deceased_by_strain["recovery_rate"],
                marker_color=ACCENT, hovertemplate="%{x}<br>Recovery: %{y:.1%}<extra></extra>")
    fig.add_bar(name="Fatality Rate", x=deceased_by_strain["virus_strain"], y=deceased_by_strain["fatality_rate"],
                marker_color=ACCENT2, hovertemplate="%{x}<br>Fatality: %{y:.1%}<extra></extra>")
    fig.update_layout(title="Recovery vs Fatality Rate by Strain", barmode="group",
                      yaxis_tickformat=".0%", **plotly_theme)
    return fig

def fig_symptoms():
    fig = go.Figure()
    fig.add_bar(name="Recovery Rate", x=fatality_by_symptoms["symptom"], y=fatality_by_symptoms["recovery_rate"],
                marker_color=ACCENT, hovertemplate="%{x}<br>Recovery: %{y:.1%}<extra></extra>")
    fig.add_bar(name="Fatality Rate", x=fatality_by_symptoms["symptom"], y=fatality_by_symptoms["fatality_rate"],
                marker_color=ACCENT2, hovertemplate="%{x}<br>Fatality: %{y:.1%}<extra></extra>")
    fig.update_layout(title="Recovery vs Fatality Rate by Symptom", barmode="group",
                      yaxis_tickformat=".0%", **plotly_theme)
    return fig

def fig_total_cases():
    fig = go.Figure()
    fig.add_bar(name="Total Cases", x=total_cases["country"], y=total_cases["total_cases"], marker_color=ACCENT)
    fig.add_bar(name="Total Deaths", x=total_cases["country"], y=total_cases["total_deaths"], marker_color=ACCENT2)
    fig.update_layout(title="Total Cases & Deaths by Country (Top 10)", barmode="group",
                      yaxis_tickformat=",", **plotly_theme)
    return fig

def fig_yearly_country():
    fig = px.line(country_cases, x="year", y="confirmed_cases", color="country",
                  labels={"confirmed_cases":"Yearly Cases","year":"Year","country":"Country"})
    fig.update_layout(title="Yearly Cases by Country (>100/yr, 2015–2025)",
                      yaxis_tickformat=",", **plotly_theme)
    return fig

def fig_strain_map():
    fig = px.scatter_geo(strains_regions, lat="lat_jitter", lon="lon_jitter",
                         color="virus_strain", size="case_count",
                         hover_name="country",
                         hover_data={"lat_jitter":False,"lon_jitter":False,"case_count":True},
                         labels={"virus_strain":"Strain","case_count":"Cases"})
    fig.update_layout(title="Virus Strain Distribution by Region",
                      paper_bgcolor=CARD_BG, font=dict(color=TEXT, family=FONT),
                      geo=dict(bgcolor=CARD_BG, landcolor="#1e2a1e", oceancolor="#0d1a2e",
                               showocean=True, showland=True, showcountries=True, countrycolor=BORDER),
                      margin=dict(t=50,l=0,r=0,b=0))
    return fig

def fig_fatality_country():
    fig = px.bar(fatality_by_country, x="country", y="fatality_rate",
                 labels={"fatality_rate":"Fatality Rate","country":"Country"},
                 color="fatality_rate", color_continuous_scale=[[0,ACCENT],[1,ACCENT2]])
    fig.update_layout(title="Case Fatality Rate by Country", yaxis_tickformat=".0%",
                      coloraxis_showscale=False, **plotly_theme)
    return fig

def fig_seasonal(exclude_high=False):
    df = cases_by_month if exclude_high else seasonal_peaks.rename(columns={"monthly_cases":"monthly_cases","month_name":"month_name"})
    if not exclude_high:
        df = seasonal_peaks.copy()
        x_col, y_col = "month_name", "monthly_cases"
    else:
        x_col, y_col = "month", "monthly_cases"
    fig = px.line(df, x=x_col, y=y_col, color="iso3",
                  labels={x_col:"Month", y_col:"Cases", "iso3":"Country"})
    title = "Monthly Cases by Country (excl. CHN & FIN)" if exclude_high else "Monthly Cases by Country"
    fig.update_layout(title=title, yaxis_tickformat=",", **plotly_theme)
    return fig

def fig_incidence():
    fig = go.Figure()
    fig.add_scatter(name="Cases", x=incidence_year["year"], y=incidence_year["cases"],
                    mode="lines+markers", line=dict(color=ACCENT, width=2))
    fig.add_scatter(name="Deaths", x=incidence_year["year"], y=incidence_year["deaths"],
                    mode="lines+markers", line=dict(color=ACCENT2, width=2))
    fig.update_layout(title="Global Incidence Trend Over Time", yaxis_tickformat=",", **plotly_theme)
    return fig

def fig_strain_cases_pie():
    fig = px.pie(strains_transmission, values="cases", names="strain_name",
                 color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(title="Distribution of Cases by Strain",
                      paper_bgcolor=CARD_BG, font=dict(color=TEXT, family=FONT),
                      margin=dict(t=50,l=20,r=20,b=20))
    fig.update_traces(textfont_color=TEXT)
    return fig

def fig_strain_deaths_pie():
    fig = px.pie(strains_transmission, values="deceased", names="strain_name",
                 color_discrete_sequence=px.colors.sequential.Plasma)
    fig.update_layout(title="Distribution of Deaths by Strain",
                      paper_bgcolor=CARD_BG, font=dict(color=TEXT, family=FONT),
                      margin=dict(t=50,l=20,r=20,b=20))
    fig.update_traces(textfont_color=TEXT)
    return fig

# ── Layout helpers ────────────────────────────────────────────────────────────

def section_header(title, subtitle=""):
    return html.Div([
        html.H2(title, style={"color": ACCENT, "fontFamily": FONT, "fontSize": "13px",
                               "letterSpacing": "3px", "textTransform": "uppercase",
                               "margin": "0 0 4px 0", "fontWeight": "600"}),
        html.P(subtitle, style={"color": MUTED, "fontFamily": FONT, "fontSize": "12px",
                                 "margin": "0 0 20px 0"}) if subtitle else None
    ], style={"borderLeft": f"3px solid {ACCENT}", "paddingLeft": "12px", "marginBottom": "24px"})

def card(children, span=1):
    return html.Div(children, style={
        "backgroundColor": CARD_BG, "border": f"1px solid {BORDER}",
        "borderRadius": "4px", "padding": "20px",
        "gridColumn": f"span {span}"
    })

def graph_card(fig_fn, span=1):
    return card(dcc.Graph(figure=fig_fn(), config={"displayModeBar": False}), span)

# ── App Layout ────────────────────────────────────────────────────────────────

app = Dash(__name__, title="Hantavirus Global Epidemiology")

app.layout = html.Div(style={
    "backgroundColor": BG, "minHeight": "100vh",
    "fontFamily": FONT, "color": TEXT, "padding": "40px"
}, children=[

    # ── Header ──
    html.Div([
        html.Div([
            html.Span("HV", style={"color": ACCENT, "fontSize": "28px", "fontWeight": "700",
                                    "letterSpacing": "2px", "marginRight": "12px"}),
            html.Span("ANALYSIS", style={"color": MUTED, "fontSize": "13px",
                                          "letterSpacing": "6px", "verticalAlign": "middle"}),
        ]),
        html.P("Hantavirus — Global Epidemiology Dashboard",
               style={"color": MUTED, "fontSize": "12px", "margin": "8px 0 0 0",
                      "letterSpacing": "1px"}),
        html.Hr(style={"borderColor": BORDER, "margin": "24px 0"})
    ]),

    # ── Section 1: Clinical & Outcomes ──
    section_header("01 — Clinical & Outcomes",
                   "Severity and fatality patterns across demographics, strains, and symptoms"),

    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginBottom":"16px"}, children=[
        graph_card(fig_severity_heatmap),
        graph_card(fig_fatality_heatmap),
    ]),
    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginBottom":"40px"}, children=[
        graph_card(fig_severity_by_strain),
        graph_card(fig_fatality_by_strain),
    ]),
    html.Div(style={"marginBottom":"40px"}, children=[
        graph_card(fig_symptoms, span=1)
    ]),

    html.Hr(style={"borderColor": BORDER, "margin": "8px 0 40px 0"}),

    # ── Section 2: Geographic ──
    section_header("02 — Geographic & Country-Level",
                   "Case burden, fatality rates, and strain distribution across regions"),

    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginBottom":"16px"}, children=[
        graph_card(fig_total_cases),
        graph_card(fig_yearly_country),
    ]),
    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginBottom":"40px"}, children=[
        graph_card(fig_strain_map),
        graph_card(fig_fatality_country),
    ]),

    html.Hr(style={"borderColor": BORDER, "margin": "8px 0 40px 0"}),

    # ── Section 3: Temporal & Seasonal ──
    section_header("03 — Temporal & Seasonal Trends",
                   "Seasonal peaks, regional variation, and global incidence over time"),

    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginBottom":"16px"}, children=[
        graph_card(fig_seasonal),
        graph_card(lambda: fig_seasonal(exclude_high=True)),
    ]),
    html.Div(style={"marginBottom":"40px"}, children=[
        graph_card(fig_incidence)
    ]),

    html.Hr(style={"borderColor": BORDER, "margin": "8px 0 40px 0"}),

    # ── Section 4: Strain Analysis ──
    section_header("04 — Virus Strain Analysis",
                   "Comparative transmissibility and lethality across strains"),

    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginBottom":"40px"}, children=[
        graph_card(fig_strain_cases_pie),
        graph_card(fig_strain_deaths_pie),
    ]),

    # ── Footer ──
    html.Div([
        html.Hr(style={"borderColor": BORDER, "margin": "0 0 16px 0"}),
        html.P(f"Data: Kaggle — Hantavirus Global Epidemiology  ·  "
               f"Built with Plotly Dash",
               style={"color": MUTED, "fontSize": "11px", "letterSpacing": "1px"})
    ])
])

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
