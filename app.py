# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:06:49 2025

@author: 20223281
"""

import dash
from dash import dcc, html, Input, Output, dash_table, State, ctx, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import dash.dash_table.FormatTemplate as FormatTemplate
import datetime

# ==========================================
# 0. CONFIGURATIE & CONSTANTS
# ==========================================

CONSTANTS = {
    'TAX': {
        'HEFFINGSVRIJ_OLD': 51396,
        'THRESHOLD_INCOME': 26500,
        'BOX3_RATE_OLD': 0.36,
    },
    'DEFAULTS': {
        'SIM_YEARS': 35,
        'START_YEAR': datetime.datetime.now().year
    },
    'PRESETS': {
        'student': {
            'name': '🎓 Student',
            'description': 'Je studeert nog en leent maximaal bij. Focus op studie, later pas op huis.',
            'invest': 2000, 'savings': 1500, 'debt': 25000, 'buffer': 1000,
            'status': 'student', 'loan': 900, 'months_borrow': 24, 'months_grace': 24,
            'income': 12000, 'income_growth': 5.0, 'years': 35, 'house_year': 10
        },
        'starter': {
            'name': '🚀 Starter',
            'description': 'Net afgestudeerd, modaal inkomen, gemiddelde schuld. Huis kopen is een doel.',
            'invest': 5000, 'savings': 10000, 'debt': 40000, 'buffer': 5000,
            'status': 'finished', 'loan': 0, 'months_borrow': 0, 'months_grace': 12,
            'income': 38000, 'income_growth': 3.0, 'years': 35, 'house_year': 5
        },
        'pro': {
            'name': '💼 Young Professional',
            'description': '5+ jaar werkend, hoger inkomen, schuld deels afgelost. Vermogen groeit.',
            'invest': 40000, 'savings': 25000, 'debt': 20000, 'buffer': 10000,
            'status': 'finished', 'loan': 0, 'months_borrow': 0, 'months_grace': 0,
            'income': 70000, 'income_growth': 2.5, 'years': 35, 'house_year': 2
        }
    }
}

# ==========================================
# 1. CLASSES (PORTFOLIO LOGICA)
# ==========================================

class Portfolio:
    def __init__(self, cash, invest, debt, min_buffer=0):
        self.cash = cash
        self.invest = invest
        self.debt = debt
        self.min_buffer = min_buffer
        self.nw_jan_1 = cash + invest - debt
        self.yearly_deposits = 0 
        self.loss_carry_forward = 0
        
    def grow(self, r_inv_m, r_sav_m, r_debt_m):
        self.cash += self.cash * r_sav_m
        self.invest += self.invest * r_inv_m
        if self.debt > 0:
            self.debt += self.debt * r_debt_m
            if self.debt < 1.0: self.debt = 0
            
    def add_loan(self, amount):
        self.invest += amount
        self.debt += amount
        
    def repay_from_salary(self, total_payment):
        if self.debt <= 0:
            return 0
        actual_payment = min(self.debt, total_payment)
        self.debt -= actual_payment
        self.yearly_deposits += actual_payment
        if self.debt < 1.0: self.debt = 0
        return actual_payment
        
    def invest_from_salary(self, amount):
        self.invest += amount
        self.yearly_deposits += amount
        
    def add_cash_from_salary(self, amount):
        self.cash += amount
        self.yearly_deposits += amount

    def get_net_worth(self):
        return self.cash + self.invest - self.debt

    def pay_tax(self, amount):
        take_from_invest = min(self.invest, amount)
        self.invest -= take_from_invest
        remaining = amount - take_from_invest
        
        if remaining > 0:
            extra_cash = max(0, self.cash - self.min_buffer)
            take_from_cash = min(extra_cash, remaining)
            self.cash -= take_from_cash
            remaining -= take_from_cash
            if remaining > 0:
                self.cash -= remaining 

class PortfolioVectorized:
    def __init__(self, n_sims, cash, invest, debt, min_buffer=0):
        self.cash = np.full(n_sims, cash, dtype=float)
        self.invest = np.full(n_sims, invest, dtype=float)
        self.debt = np.full(n_sims, debt, dtype=float)
        self.min_buffer = min_buffer
        self.nw_jan_1 = self.get_net_worth()
        self.yearly_deposits = np.zeros(n_sims)
        self.loss_carry_forward = np.zeros(n_sims)
        
    def grow(self, r_inv_m_array, r_sav_m, r_debt_m):
        self.cash *= (1 + r_sav_m)
        self.invest *= (1 + r_inv_m_array)
        has_debt = self.debt > 0
        self.debt[has_debt] *= (1 + r_debt_m)
            
    def add_loan(self, amount):
        self.invest += amount
        self.debt += amount
        
    def repay_from_salary(self, total_payment):
        payment = np.minimum(self.debt, total_payment)
        self.debt -= payment
        self.yearly_deposits += payment
        return payment
        
    def invest_from_salary(self, amount):
        self.invest += amount
        self.yearly_deposits += amount

    def get_net_worth(self):
        return self.cash + self.invest - self.debt

    def pay_tax(self, amount_array):
        take_from_invest = np.minimum(self.invest, amount_array)
        self.invest -= take_from_invest
        remaining = amount_array - take_from_invest
        
        mask_rem = remaining > 0
        if np.any(mask_rem):
            free_cash = np.maximum(0, self.cash - self.min_buffer)
            take_from_cash = np.minimum(free_cash, remaining)
            self.cash -= take_from_cash
            remaining -= take_from_cash
            
            mask_rem_2 = remaining > 0
            if np.any(mask_rem_2):
                self.cash[mask_rem_2] -= remaining[mask_rem_2]

# ==========================================
# 2. HULPFUNCTIES
# ==========================================

def validate_inputs(cur_inv, cur_sav, cur_debt, loan):
    if cur_inv < 0 or cur_sav < 0 or cur_debt < 0 or loan < 0:
        return False, "⚠️ Financiële waarden mogen niet negatief zijn."
    return True, ""

def calculate_draagkracht_monthly(annual_income, threshold, repay_years):
    excess_income = max(0, annual_income - threshold)
    percentage = 0.04 if repay_years == 35 else 0.12
    return (excess_income * percentage) / 12

def calculate_annuity(debt, rate_month, months):
    if months <= 0: return 0
    if debt <= 0: return 0
    if rate_month == 0: return debt / months
    return (debt * rate_month) / (1 - (1 + rate_month)**(-months))

def get_monthly_geometric_rate(annual_rate):
    return (1 + annual_rate) ** (1/12) - 1

def get_monthly_nominal_rate(annual_rate):
    return annual_rate / 12

# ==========================================
# 3. SIMULATIE FUNCTIES
# ==========================================

def run_simulation(
    sim_years_duration,
    market_return_decimal, rate_debt_decimal, inflation_decimal, rate_savings_decimal,
    tax_rate_decimal, tax_free_income_val, year_new_tax_system, 
    fict_sav_decimal, fict_inv_decimal, fict_debt_decimal,
    house_year, current_invest, current_savings, current_debt,
    loan_monthly, months_left_to_borrow, months_grace_remaining,
    years_repayment_total, emergency_buffer,
    use_draagkracht, start_income, income_growth_decimal,
    has_partner
):
    sim_years = sim_years_duration
    months = sim_years * 12
    start_year_sim = CONSTANTS['DEFAULTS']['START_YEAR']
    
    multiplier = 2 if has_partner else 1
    heffingsvrij_old = CONSTANTS['TAX']['HEFFINGSVRIJ_OLD'] * multiplier
    tax_free_income_new = tax_free_income_val * multiplier
    threshold_income = CONSTANTS['TAX']['THRESHOLD_INCOME']
    current_annual_income = start_income

    rm_inv = get_monthly_geometric_rate(market_return_decimal)
    rm_sav = get_monthly_geometric_rate(rate_savings_decimal)
    rm_inf = get_monthly_geometric_rate(inflation_decimal)
    rm_debt = get_monthly_nominal_rate(rate_debt_decimal)

    start_cash_a = current_savings
    start_invest_a = current_invest
    start_debt_a = 0
    
    if current_debt > 0:
        available_cash = max(0, start_cash_a - emergency_buffer)
        pay_cash = min(available_cash, current_debt)
        start_cash_a -= pay_cash
        rem_debt = current_debt - pay_cash
        pay_invest = min(start_invest_a, rem_debt)
        start_invest_a -= pay_invest
        start_debt_a = rem_debt - pay_invest

    scen_a = Portfolio(start_cash_a, start_invest_a, start_debt_a, emergency_buffer)
    scen_a_base = Portfolio(start_cash_a, start_invest_a, start_debt_a, emergency_buffer)
    scen_b = Portfolio(current_savings, current_invest, current_debt, emergency_buffer)
    
    data = {
        'Maand': [], 'Jaar': [], 
        'Netto_A': [], 'Netto_A_Base': [], 'Netto_B': [], 
        'Schuld_A': [], 'Schuld_B': [], 
        'Voordeel_B': [], 'Voordeel_B_Real': [],
        'Cashflow_A_Invest': [], 'Cashflow_B_Pay': []     
    }
    
    borrowing_phase_end = months_left_to_borrow
    repayment_phase_start = months_left_to_borrow + months_grace_remaining
    repayment_phase_end = repayment_phase_start + (years_repayment_total * 12)
    
    statutory_payment_b = 0
    if current_debt > 0 and months_left_to_borrow == 0 and months_grace_remaining == 0:
        statutory_payment_b = calculate_annuity(current_debt, rm_debt, years_repayment_total * 12)

    cumulative_inflation = 1.0

    for m in range(1, months + 1):
        current_year_cal = start_year_sim + ((m-1) // 12)
        
        scen_a.grow(rm_inv, rm_sav, rm_debt)
        scen_a_base.grow(rm_inv, rm_sav, rm_debt)
        scen_b.grow(rm_inv, rm_sav, rm_debt)
        
        cumulative_inflation *= (1 + rm_inf)

        if m <= borrowing_phase_end:
            scen_b.add_loan(loan_monthly)
            
        if m == repayment_phase_start + 1 and statutory_payment_b == 0:
            if scen_b.debt > 0:
                statutory_payment_b = calculate_annuity(scen_b.debt, rm_debt, years_repayment_total * 12)
        
        cf_a_invest = 0
        cf_b_pay = 0

        if m > repayment_phase_start:
            if m <= repayment_phase_end:
                payment_b_due = 0
                if scen_b.debt > 0:
                    payment_b_due = statutory_payment_b
                    if use_draagkracht:
                        max_cap = calculate_draagkracht_monthly(current_annual_income, threshold_income, years_repayment_total)
                        payment_b_due = min(statutory_payment_b, max_cap)
                    
                    actual_paid_b = scen_b.repay_from_salary(payment_b_due)
                    cf_b_pay = actual_paid_b
                else:
                    actual_paid_b = 0
                
                paid_a = 0
                if scen_a.debt > 0:
                    paid_a = scen_a.repay_from_salary(actual_paid_b)
                
                if scen_a_base.debt > 0:
                     scen_a_base.repay_from_salary(actual_paid_b)
                scen_a_base.debt = scen_a.debt 

                difference = actual_paid_b - paid_a
                
                if difference > 0:
                    scen_a.invest_from_salary(difference)
                    cf_a_invest = difference
                    scen_a_base.add_cash_from_salary(difference)

        if m == repayment_phase_end:
            scen_a.debt = 0
            scen_a_base.debt = 0
            scen_b.debt = 0

        if m % 12 == 0:
            current_annual_income *= (1 + income_growth_decimal)
            threshold_income *= (1 + inflation_decimal)
            heffingsvrij_old *= (1 + inflation_decimal)
            tax_free_income_new *= (1 + inflation_decimal)

            def apply_tax(portfolio, is_new_system):
                tax_amount = 0
                if not is_new_system:
                    nw = portfolio.get_net_worth()
                    grondslag = max(0, nw - heffingsvrij_old)
                    if grondslag > 0:
                        yield_val = (portfolio.cash * fict_sav_decimal) + (portfolio.invest * fict_inv_decimal) - (portfolio.debt * fict_debt_decimal)
                        if yield_val > 0:
                            tax_amount = grondslag * (yield_val / nw) * CONSTANTS['TAX']['BOX3_RATE_OLD']
                else:
                    gain = portfolio.get_net_worth() - portfolio.nw_jan_1 - portfolio.yearly_deposits
                    if gain < 0:
                        portfolio.loss_carry_forward += abs(gain)
                        tax_amount = 0
                    else:
                        offset = min(gain, portfolio.loss_carry_forward)
                        portfolio.loss_carry_forward -= offset
                        taxable_gain = gain - offset
                        tax_amount = max(0, (taxable_gain - tax_free_income_new) * tax_rate_decimal)
                
                portfolio.pay_tax(tax_amount)
                portfolio.nw_jan_1 = portfolio.get_net_worth()
                portfolio.yearly_deposits = 0

            is_new = current_year_cal >= year_new_tax_system
            apply_tax(scen_a, is_new)
            apply_tax(scen_a_base, is_new)
            apply_tax(scen_b, is_new)

        diff = scen_b.get_net_worth() - scen_a.get_net_worth()
        
        data['Maand'].append(m)
        data['Jaar'].append(m/12)
        data['Netto_A'].append(scen_a.get_net_worth())
        data['Netto_A_Base'].append(scen_a_base.get_net_worth())
        data['Netto_B'].append(scen_b.get_net_worth())
        data['Schuld_A'].append(scen_a.debt)
        data['Schuld_B'].append(scen_b.debt)
        data['Voordeel_B'].append(diff)
        data['Voordeel_B_Real'].append(diff / cumulative_inflation)
        data['Cashflow_A_Invest'].append(cf_a_invest)
        data['Cashflow_B_Pay'].append(cf_b_pay)

    df = pd.DataFrame(data)
    
    idx_house = int(house_year * 12) - 1
    if idx_house >= len(df): idx_house = len(df) - 1
    if idx_house < 0: idx_house = 0
    
    house_diff = df.iloc[idx_house]['Voordeel_B']
    house_diff_real = df.iloc[idx_house]['Voordeel_B_Real']
    salary_effect = df.iloc[-1]['Netto_A'] - df.iloc[-1]['Netto_A_Base']
    
    return df, house_diff, house_diff_real, salary_effect, statutory_payment_b

def run_monte_carlo(
    sim_years_duration,
    n_sims, volatility_decimal,
    market_return_decimal, rate_debt_decimal, inflation_decimal, rate_savings_decimal,
    tax_rate_decimal, tax_free_income_val, year_new_tax_system, 
    fict_sav_decimal, fict_inv_decimal, fict_debt_decimal,
    house_year,
    current_invest, current_savings, current_debt,
    loan_monthly, months_left_to_borrow, months_grace_remaining,
    years_repayment_total, emergency_buffer,
    use_draagkracht, start_income, income_growth_decimal,
    has_partner
):
    np.random.seed(42)
    sim_years = sim_years_duration
    months = sim_years * 12
    start_year_sim = CONSTANTS['DEFAULTS']['START_YEAR']
    
    multiplier = 2 if has_partner else 1
    heffingsvrij_old = CONSTANTS['TAX']['HEFFINGSVRIJ_OLD'] * multiplier
    tax_free_income_new = tax_free_income_val * multiplier
    threshold_income = CONSTANTS['TAX']['THRESHOLD_INCOME']
    current_annual_income = start_income
    
    mu_month = get_monthly_geometric_rate(market_return_decimal)
    sigma_month = volatility_decimal / np.sqrt(12)
    
    r_debt_month = get_monthly_nominal_rate(rate_debt_decimal)
    r_sav_month = get_monthly_geometric_rate(rate_savings_decimal)
    
    start_cash_a = current_savings
    start_invest_a = current_invest
    start_debt_a = 0
    if current_debt > 0:
        available_cash = max(0, start_cash_a - emergency_buffer)
        pay_c = min(available_cash, current_debt)
        start_cash_a -= pay_c
        rem_d = current_debt - pay_c
        pay_i = min(start_invest_a, rem_d)
        start_invest_a -= pay_i
        start_debt_a = rem_d - pay_i

    scen_a = PortfolioVectorized(n_sims, start_cash_a, start_invest_a, start_debt_a, emergency_buffer)
    scen_b = PortfolioVectorized(n_sims, current_savings, current_invest, current_debt, emergency_buffer)
    
    years_axis = np.arange(0, sim_years + 1)
    p10_diff, p50_diff, p90_diff = [], [], []
    
    borrowing_phase_end = months_left_to_borrow
    repayment_phase_start = months_left_to_borrow + months_grace_remaining
    repayment_phase_end = repayment_phase_start + (years_repayment_total * 12)
    
    statutory_payment_b = 0
    if current_debt > 0 and months_left_to_borrow == 0 and months_grace_remaining == 0:
         statutory_payment_b = calculate_annuity(current_debt, r_debt_month, years_repayment_total * 12)

    p10_diff.append(0); p50_diff.append(0); p90_diff.append(0)

    for m in range(1, months + 1):
        r_market = np.random.normal(mu_month, sigma_month, n_sims)
        r_market = np.clip(r_market, -0.30, 0.50)
        
        scen_a.grow(r_market, r_sav_month, r_debt_month)
        scen_b.grow(r_market, r_sav_month, r_debt_month)
        
        if m <= borrowing_phase_end:
            scen_b.add_loan(loan_monthly)
            
        if m == repayment_phase_start + 1 and statutory_payment_b == 0:
            d_val = scen_b.debt[0]
            if d_val > 0:
                statutory_payment_b = calculate_annuity(d_val, r_debt_month, years_repayment_total * 12)

        if m > repayment_phase_start:
             if m <= repayment_phase_end:
                 payment_b_due = statutory_payment_b
                 if use_draagkracht:
                     max_cap = calculate_draagkracht_monthly(current_annual_income, threshold_income, years_repayment_total)
                     payment_b_due = min(statutory_payment_b, max_cap)
                 
                 actual_paid_b = scen_b.repay_from_salary(payment_b_due)
                 paid_a = scen_a.repay_from_salary(actual_paid_b)
                 
                 diff = actual_paid_b - paid_a
                 scen_a.invest_from_salary(diff)

        if m == repayment_phase_end:
            scen_a.debt[:] = 0
            scen_b.debt[:] = 0

        if m % 12 == 0:
            current_year_cal = start_year_sim + ((m-1) // 12)
            current_annual_income *= (1 + income_growth_decimal)
            threshold_income *= (1 + inflation_decimal)
            heffingsvrij_old *= (1 + inflation_decimal)
            tax_free_income_new *= (1 + inflation_decimal)
            
            is_new = current_year_cal >= year_new_tax_system
            
            def calc_tax_vec(port):
                nw = port.get_net_worth()
                t_amt = np.zeros(n_sims)
                if not is_new:
                    grondslag = np.maximum(0, nw - heffingsvrij_old)
                    yield_val = (port.cash * fict_sav_decimal) + (port.invest * fict_inv_decimal) - (port.debt * fict_debt_decimal)
                    mask = (grondslag > 0) & (yield_val > 0)
                    t_amt[mask] = grondslag[mask] * (yield_val[mask] / nw[mask]) * CONSTANTS['TAX']['BOX3_RATE_OLD']
                else:
                    gain = nw - port.nw_jan_1 - port.yearly_deposits
                    losers_mask = gain < 0
                    port.loss_carry_forward[losers_mask] += np.abs(gain[losers_mask])
                    winners_mask = gain > 0
                    offset = np.minimum(gain[winners_mask], port.loss_carry_forward[winners_mask])
                    port.loss_carry_forward[winners_mask] -= offset
                    taxable_gain = gain[winners_mask] - offset
                    final_taxable = np.maximum(0, taxable_gain - tax_free_income_new)
                    t_amt[winners_mask] = final_taxable * tax_rate_decimal
                return t_amt

            scen_a.pay_tax(calc_tax_vec(scen_a))
            scen_a.nw_jan_1 = scen_a.get_net_worth()
            scen_a.yearly_deposits[:] = 0
            scen_b.pay_tax(calc_tax_vec(scen_b))
            scen_b.nw_jan_1 = scen_b.get_net_worth()
            scen_b.yearly_deposits[:] = 0
            
            diff = scen_b.get_net_worth() - scen_a.get_net_worth()
            deflator = (1 + inflation_decimal) ** (m/12)
            diff_real = diff / deflator
            
            p10_diff.append(np.percentile(diff_real, 10))
            p50_diff.append(np.median(diff_real))
            p90_diff.append(np.percentile(diff_real, 90))
            
        if m == months:
            diff_final = scen_b.get_net_worth() - scen_a.get_net_worth()
            deflator_final = (1 + inflation_decimal) ** (m/12)
            end_diffs_real = diff_final / deflator_final
            
    return years_axis, p10_diff, p50_diff, p90_diff, end_diffs_real

# ==========================================
# 3. LAYOUT & STYLING
# ==========================================

# SCROLL FIX: Set Container to 100vh and allow children to scroll independently
STYLE_CONTAINER = {
    'display': 'flex',
    'flexDirection': 'row',
    'fontFamily': '"Segoe UI", Roboto, sans-serif',
    'backgroundColor': '#f4f7f6',
    'height': '100vh',   # Full viewport height
    'overflow': 'hidden', # Disable body scroll
    'margin': 0, 'padding': 0
}

STYLE_SIDEBAR = {
    'flex': '0 0 380px',
    'width': '380px',
    'backgroundColor': '#fff',
    'borderRight': '1px solid #eee',
    'padding': '25px',
    'overflowY': 'auto', # Independent Scroll
    'height': '100%',    # Full height of container
}

STYLE_CONTENT = {
    'flex': '1',
    'padding': '20px',
    'maxWidth': '100%',
    'overflowY': 'auto', # Independent Scroll
    'height': '100%'     # Full height of container
}

STYLE_CARD = { 'backgroundColor': '#fff', 'borderRadius': '12px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.05)', 'padding': '25px', 'border': '1px solid #f0f0f0' }
STYLE_LABEL = { 'fontWeight': '600', 'color': '#2c3e50', 'fontSize': '13px', 'marginBottom': '5px', 'display': 'block' }
STYLE_INPUT = { 'width': '100%', 'padding': '8px', 'borderRadius': '6px', 'border': '1px solid #ddd', 'marginBottom': '15px', 'boxSizing': 'border-box' }
STYLE_BTN_SEC = {'backgroundColor': '#3498db', 'color': 'white', 'padding': '8px 16px', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer', 'marginRight': '10px'}
TOOLTIP_CFG = {"placement": "bottom", "always_visible": True}

# CSS GRID FOR 2x2 KPIs
STYLE_KPI_GRID = {
    'display': 'grid',
    'gridTemplateColumns': '1fr 1fr', # Two equal columns
    'gap': '20px',
    'marginBottom': '20px'
}

app = dash.Dash(__name__)
server = app.server
app.title = "DUO Planner V34"

app.layout = html.Div(style=STYLE_CONTAINER, children=[
    
    dcc.Store(id='simulation-results-store'),
    dcc.Download(id="download-dataframe-csv"),

    html.Div(style=STYLE_SIDEBAR, children=[
        html.H2("⚙️ Instellingen", style={'marginTop': '0', 'fontSize': '22px', 'color': '#333'}),
        
        html.H4("1. Kies je Profiel", style={'color': '#2980b9'}),
        dcc.Dropdown(
            id='scenario-preset',
            options=[
                {'label': '🎓 Student', 'value': 'student'},
                {'label': '🚀 Starter', 'value': 'starter'},
                {'label': '💼 Young Professional', 'value': 'pro'},
                {'label': '✏️ Aangepast', 'value': 'custom'}
            ],
            value='custom', clearable=False, style={'marginBottom': '5px'}
        ),
        html.Div(id='preset-indicator', style={'fontSize': '11px', 'color': '#7f8c8d', 'fontStyle': 'italic', 'marginBottom': '10px'}),
        html.Button("🔄 Herstel Profiel", id='btn-reset-preset', style={'backgroundColor': '#95a5a6', 'color': 'white', 'padding': '6px 12px', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '12px', 'width': '100%', 'marginBottom': '15px'}),

        html.H4("2. Financiële Situatie", style={'color': '#2980b9'}),
        html.Label("⏳ Simulatie Duur (Jaren)", style=STYLE_LABEL),
        dcc.Slider(id='sim-years', min=10, max=50, step=5, value=35, marks={10:'10', 25:'25', 35:'35', 50:'50'}, tooltip=TOOLTIP_CFG),
        html.Br(),

        html.Label("Belegd Vermogen (€)", style=STYLE_LABEL), dcc.Input(id='curr-invest', type='number', value=75000, style=STYLE_INPUT),
        html.Label("Spaargeld (€)", style=STYLE_LABEL), dcc.Input(id='curr-savings', type='number', value=10000, style=STYLE_INPUT),
        html.Label("Noodbuffer (€)", style=STYLE_LABEL), dcc.Input(id='emergency-buffer', type='number', value=5000, style=STYLE_INPUT),
        html.Label("Openstaande Schuld (€)", style=STYLE_LABEL), dcc.Input(id='curr-debt', type='number', value=0, style=STYLE_INPUT),

        html.H4("3. Lening & Inkomen", style={'color': '#2980b9'}),
        html.Label("Status Studie", style=STYLE_LABEL),
        dcc.RadioItems(id='study-status-toggle', options=[{'label': ' Studerend', 'value': 'student'}, {'label': ' Klaar', 'value': 'finished'}], value='student', labelStyle={'display': 'block', 'marginBottom': '5px', 'fontSize': '14px'}, style={'marginBottom': '15px'}),
        
        html.Div(id='borrowing-inputs-container', children=[
            html.Label("Nog te lenen (Maanden)", style=STYLE_LABEL), dcc.Slider(id='months-borrow', min=0, max=60, step=1, value=30, marks={0:'0', 60:'60'}, tooltip=TOOLTIP_CFG), html.Br(),
            html.Label("Leenbedrag / maand (€)", style=STYLE_LABEL), dcc.Input(id='loan-monthly', type='number', value=800, style=STYLE_INPUT),
        ]),
        html.Label("Aanloopfase Restant (Mnd)", style=STYLE_LABEL), dcc.Slider(id='months-grace', min=0, max=24, step=1, value=24, marks={0:'0', 24:'24'}, tooltip=TOOLTIP_CFG), html.Br(),
        html.Label("Aflostermijn", style=STYLE_LABEL), dcc.Dropdown(id='repay-years', options=[{'label': '35 jaar (SF35)', 'value': 35}, {'label': '15 jaar (SF15)', 'value': 15}], value=35, clearable=False, style={'marginBottom': '15px'}),
        
        html.Div(style={'backgroundColor': '#eafaf1', 'padding': '10px', 'borderRadius': '8px', 'border': '1px solid #d5f5e3', 'marginBottom': '15px'}, children=[
            html.Label("Draagkrachtmeting", style={**STYLE_LABEL, 'color': '#27ae60'}),
            dcc.Checklist(id='has-partner', options=[{'label': ' ❤️ Fiscaal Partner', 'value': 'yes'}], value=[], style={'fontSize': '13px', 'marginBottom': '10px'}),
            dcc.Checklist(id='use-draagkracht', options=[{'label': ' Houd rekening met inkomen', 'value': 'yes'}], value=['yes'], style={'marginBottom': '10px', 'fontSize': '14px'}),
            
            html.Label("Bruto Jaarinkomen (€)", id='label-income', style=STYLE_LABEL), 
            dcc.Input(id='start-income', type='number', value=45000, style=STYLE_INPUT),
            
            html.Label("Verwachte Inkomensgroei (%)", style=STYLE_LABEL), dcc.Input(id='income-growth', type='number', value=2.5, step=0.1, style=STYLE_INPUT),
        ]),

        html.H4("4. Markt & Inflatie", style={'color': '#2980b9'}),
        html.Label("Kies een ETF (Preset)", style=STYLE_LABEL), dcc.Dropdown(id='preset-selector', options=[{'label': 'Aangepast', 'value': 'custom'}, {'label': 'VUAA (S&P 500)', 'value': 'vuaa'}, {'label': 'VWCE (All-World)', 'value': 'vwce'}], value='custom', clearable=False, style={'marginBottom': '15px'}),
        html.Label("📈 Rendement Beurs (%)", style=STYLE_LABEL), dcc.Slider(id='return-slider', min=0, max=15, step=0.5, value=7, marks={0:'0%', 15:'15%'}, tooltip=TOOLTIP_CFG), html.Br(),
        html.Label("📊 MC Volatiliteit (%)", style=STYLE_LABEL), dcc.Slider(id='vol-slider', min=5, max=30, step=1, value=15, marks={5:'5%', 30:'30%'}, tooltip=TOOLTIP_CFG), html.Br(),
        
        html.Label("🎓 DUO Rente (%)", style=STYLE_LABEL), dcc.Input(id='duo-rate', type='number', value=2.57, step=0.01, style=STYLE_INPUT),
        html.Label("💸 Inflatie (%)", style=STYLE_LABEL), dcc.Slider(id='inflation-slider', min=0, max=5, step=0.5, value=2.0, marks={0:'0%', 5:'5%'}, tooltip=TOOLTIP_CFG), 
        
        html.Div(id='reality-check-text', style={'fontSize': '11px', 'color': '#e67e22', 'marginTop': '5px', 'fontStyle': 'italic'}),
        html.Br(),
        
        html.Label("🐷 Spaarrente (%)", style=STYLE_LABEL), dcc.Slider(id='savings-rate-slider', min=0, max=5, step=0.1, value=2.0, marks={0:'0%', 5:'5%'}, tooltip=TOOLTIP_CFG), html.Br(),

        html.H4("5. Fiscale Aannames", style={'color': '#2980b9'}),
        html.Div(style={'backgroundColor': '#f0f8ff', 'padding': '10px', 'borderRadius': '8px', 'border': '1px solid #cce5ff', 'marginBottom': '15px'}, children=[
            html.H5("Overbruggingswet", style={'marginTop':0, 'fontSize': '13px', 'color': '#004085'}),
            html.Label("Forfait Sparen (%)", style=STYLE_LABEL), dcc.Input(id='fict-sav', type='number', value=1.44, step=0.01, style=STYLE_INPUT),
            html.Label("Forfait Beleggen (%)", style=STYLE_LABEL), dcc.Input(id='fict-inv', type='number', value=7.78, step=0.01, style=STYLE_INPUT),
            html.Label("Forfait Schuld (%)", style=STYLE_LABEL), dcc.Input(id='fict-debt', type='number', value=2.62, step=0.01, style=STYLE_INPUT),
        ]),
        html.Div(style={'backgroundColor': '#fff3cd', 'padding': '10px', 'borderRadius': '8px', 'border': '1px solid #ffeeba'}, children=[
            html.H5("Nieuw Stelsel", style={'marginTop':0, 'fontSize': '13px', 'color': '#856404'}),
            html.Label("Startjaar Nieuw Stelsel", style=STYLE_LABEL), dcc.Dropdown(id='tax-year-input', options=[2026, 2027, 2028, 2029, 2030], value=2028, clearable=False, style={'marginBottom':'10px'}),
            html.Label("Tarief (%)", style=STYLE_LABEL), dcc.Input(id='tax-rate-input', type='number', value=36, step=1, style=STYLE_INPUT),
            html.Label("Heffingsvrij Inkomen (€)", style=STYLE_LABEL), dcc.Input(id='tax-free-input', type='number', value=2500, step=100, style=STYLE_INPUT),
        ]),
        
        html.Br(), html.Label("🏠 Huis Jaar", style=STYLE_LABEL), dcc.Slider(id='house-year', min=1, max=25, step=1, value=7, marks={1:'1', 25:'25'}, tooltip=TOOLTIP_CFG),
    ]),

    html.Div(style=STYLE_CONTENT, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '10px 0', 'marginBottom': '20px', 'backgroundColor': '#f4f7f6', 'borderBottom': '1px solid #ddd'}, children=[
            html.H1("DUO Planner Pro V34", style={'color': '#2c3e50', 'margin': 0}),
            html.Div(style={'display': 'flex', 'alignItems': 'center'}, children=[
                html.Button("🚀 Bereken Scenario", id="btn-run", style={'backgroundColor': '#27ae60', 'color': 'white', 'padding': '12px 20px', 'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'fontSize': '16px', 'fontWeight': 'bold', 'marginRight': '15px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),
                html.Button("❓ Uitleg", id="btn-howto", n_clicks=0, style=STYLE_BTN_SEC), 
                html.Button("📥 Download CSV", id="btn-download", style={**STYLE_BTN_SEC, 'backgroundColor': '#7f8c8d'})
            ])
        ]),
        
        html.Div(id='error-message', style={'color': 'red', 'fontWeight': 'bold', 'marginBottom': '10px'}),
        html.Div(id='howto-content', style={'display': 'none', 'backgroundColor': '#eaf2f8', 'padding': '20px', 'borderRadius': '8px', 'border': '1px solid #3498db', 'marginBottom':'20px'}, children=[
            html.H4("Welkom bij de DUO Planner V34", style={'marginTop':0}),
            html.P("Deze tool vergelijkt twee strategieën:", style={'fontWeight': 'bold'}),
            html.Ul([
                html.Li([html.B("Scenario B (Lenen/Niet Aflossen):", style={'color': '#e67e22'}), " Je betaalt alleen wat wettelijk moet (obv draagkracht). Restschuld wordt kwijtgescholden."]),
                html.Li([html.B("Scenario A (Aflossen):", style={'color': '#2980b9'}), " Je gebruikt je spaargeld/beleggingen om direct de schuld (deels) af te lossen. De maandlasten die je bespaart, beleg je."]),
            ]),
            html.P("Belangrijke aannames:", style={'fontWeight': 'bold'}),
            html.Ul([
                html.Li("Bij 'Fiscaal Partner' verdubbelen de vrijstellingen. Vul bij Inkomen het gezamenlijke inkomen in."),
                html.Li("Winst uit Discipline: Het extra vermogen dat je opbouwt door maandelijks te beleggen wat je anders aan DUO zou betalen (in Scenario A)."),
                html.Li("Benodigd Rendement: Wat je minimaal bruto moet halen om de leningrente te verslaan (na belasting).")
            ], style={'fontSize': '13px'})
        ]),
        dcc.Tabs([
            dcc.Tab(label='📊 Scenario Analyse', children=[
                html.Div(style={'padding': '20px'}, children=[
                    # 2x2 GRID FOR KPIs
                    html.Div(style=STYLE_KPI_GRID, children=[
                        # 1. Total Benefit
                        html.Div(style=STYLE_CARD, children=[
                            html.H4("⚖️ Netto Voordeel Lenen", style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'}), 
                            html.Div(id='kpi-end', style={'fontSize': '28px', 'fontWeight': 'bold', 'marginTop': '5px'}), 
                            html.Div("Extra winst t.o.v. Scenario A", style={'fontSize': '11px', 'color': '#2c3e50', 'fontWeight': 'bold'}), 
                            html.Div(id='kpi-end-real', style={'fontSize': '12px', 'color': '#95a5a6'})
                        ]),
                        # 2. Discipline Gain
                        html.Div(style=STYLE_CARD, children=[
                            html.H4("💪 Winst uit Discipline", style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'}), 
                            html.Div(id='kpi-salary', style={'fontSize': '28px', 'fontWeight': 'bold', 'marginTop': '5px', 'color': '#2980b9'}), 
                            html.Div("Winst uit beleggen vs sparen", style={'fontSize': '11px', 'color': '#2c3e50', 'fontWeight': 'bold'}), 
                            html.Div(id='kpi-monthly-payment', style={'fontSize': '12px', 'color': '#e67e22', 'marginTop': '5px'})
                        ]),
                        # 3. House Benefit
                        html.Div(style=STYLE_CARD, children=[
                            html.H4("🏠 Huis Voordeel", style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'}), 
                            html.Div(id='kpi-house', style={'fontSize': '28px', 'fontWeight': 'bold', 'marginTop': '5px'}), 
                            html.Div(id='kpi-house-real', style={'fontSize': '12px', 'color': '#95a5a6'})
                        ]),
                        # 4. Break-even
                        html.Div(style=STYLE_CARD, children=[
                            html.H4("🎯 Benodigd Rendement", style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'}), 
                            html.Div(id='kpi-breakeven', style={'fontSize': '28px', 'fontWeight': 'bold', 'marginTop': '5px', 'color': '#8e44ad'}), 
                            html.Div("Bruto rendement voor break-even", style={'fontSize': '11px', 'color': '#2c3e50', 'fontWeight': 'bold'}), 
                            html.Div(id='kpi-breakeven-sub', style={'fontSize': '12px', 'color': '#8e44ad', 'marginTop': '5px'})
                        ]),
                    ]),
                    
                    html.Div(style={**STYLE_CARD, 'marginTop': '0'}, children=[html.H3("Vermogensverloop & Schuld", style={'margin': '0 0 15px 0', 'color': '#34495e'}), dcc.Graph(id='main-graph', config={'displayModeBar': False}, style={'height': '500px'})]),
                    html.Div(style=STYLE_CARD, children=[html.H3("Netto Voordeel Scenario B (Lenen)", style={'margin': '0 0 15px 0', 'color': '#34495e'}), dcc.Graph(id='diff-graph', config={'displayModeBar': False}, style={'height': '350px'})]),
                    html.Div(style=STYLE_CARD, children=[html.H3("Tussenstanden & Cashflow", style={'margin': '0 0 15px 0', 'color': '#34495e'}), html.Div(id='table-container')]),
                ])
            ]),
            dcc.Tab(label='🎲 Risico Analyse (Monte Carlo)', children=[
                html.Div(style={'padding': '20px'}, children=[
                    html.Div(style={'backgroundColor': '#fff3cd', 'padding': '15px', 'borderRadius': '8px', 'borderLeft': '5px solid #f1c40f', 'marginBottom':'20px'}, children=[html.H4("⚠️ Risico Analyse", style={'marginTop':'0', 'color':'#856404'}), html.P("Simulatie van 500 mogelijke toekomsten (Geometrische Brownse Beweging). Inclusief outliers.", style={'color': '#856404'})]),
                    html.Div(style=STYLE_CARD, children=[html.H3("Waaier van Onzekerheid", style={'margin': '0 0 15px 0', 'color': '#34495e'}), dcc.Graph(id='mc-fan', style={'height': '400px'})]),
                    html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}, children=[
                        html.Div(style={**STYLE_CARD, 'flex': '1', 'minWidth': '300px'}, children=[html.H3("Kansverdeling", style={'margin': '0 0 15px 0', 'color': '#34495e'}), dcc.Graph(id='mc-hist', style={'height': '300px'})]),
                        html.Div(style={**STYLE_CARD, 'flex': '0.5', 'minWidth': '300px'}, children=[html.H3("Statistieken", style={'margin': '0 0 15px 0', 'color': '#34495e'}), html.Div(id='mc-stats', style={'lineHeight': '2'})])
                    ])
                ])
            ])
        ]),
        html.Div(style={'marginTop': '50px', 'padding': '20px', 'borderTop': '1px solid #ddd', 'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '12px'}, children=[html.P("Disclaimer: Geen financieel advies. Gemaakt met Python & Dash.")])
    ])
])

# ==========================================
# 4. CALLBACKS
# ==========================================

FIN_INPUTS = [
    'curr-invest', 'curr-savings', 'curr-debt', 'study-status-toggle',
    'loan-monthly', 'start-income', 'repay-years', 'months-borrow',
    'months-grace', 'emergency-buffer', 'house-year', 'income-growth'
]

@app.callback(Output('label-income', 'children'), Input('has-partner', 'value'))
def update_income_label(partner_list):
    has_partner = True if (partner_list and 'yes' in partner_list) else False
    if has_partner:
        return "Gezamenlijk Bruto Jaarinkomen (€)"
    return "Bruto Jaarinkomen (€)"

@app.callback(Output('reality-check-text', 'children'), [Input('duo-rate', 'value'), Input('inflation-slider', 'value')])
def check_economic_reality(duo, inf):
    if not duo or not inf: return ""
    if inf > (duo + 1.5):
        return "⚠️ Let op: Historisch gezien is de rente vaak hoger of gelijk aan inflatie. Een groot negatief verschil is zeldzaam."
    return ""

@app.callback(
    [Output('scenario-preset', 'value'), Output('preset-indicator', 'children')] +
    [Output(uid, 'value') for uid in FIN_INPUTS],
    [Input('scenario-preset', 'value'), Input('btn-reset-preset', 'n_clicks')] +
    [Input(uid, 'value') for uid in FIN_INPUTS],
    prevent_initial_call=False
)
def manage_profile_state(preset_val, reset_clicks, *args):
    current_inputs = list(args)
    ctx_id = ctx.triggered_id

    if not ctx_id or ctx_id == 'scenario-preset' or ctx_id == 'btn-reset-preset':
        target_preset = preset_val
        if ctx_id == 'btn-reset-preset' and target_preset == 'custom':
            target_preset = 'student'

        if target_preset in CONSTANTS['PRESETS']:
            p = CONSTANTS['PRESETS'][target_preset]
            desc = f"📋 {p['description']}"
            new_values = [
                p['invest'], p['savings'], p['debt'], p['status'],
                p['loan'], p['income'], p['years'], p['months_borrow'],
                p['months_grace'], p['buffer'], p['house_year'], p['income_growth']
            ]
            return [target_preset, desc] + new_values
        else:
            return [dash.no_update, "✏️ Je gebruikt aangepaste instellingen"] + [dash.no_update] * len(FIN_INPUTS)

    if ctx_id in FIN_INPUTS:
        if preset_val == 'custom':
            return [dash.no_update, dash.no_update] + [dash.no_update] * len(FIN_INPUTS)
        
        if preset_val in CONSTANTS['PRESETS']:
            p = CONSTANTS['PRESETS'][preset_val]
            (c_inv, c_sav, c_debt, c_stat, c_loan, c_inc, c_rep, c_bor, c_gra, c_buf, c_hou, c_gro) = current_inputs
            try:
                matches = (
                    c_inv == p['invest'] and c_sav == p['savings'] and c_debt == p['debt'] and
                    c_stat == p['status'] and c_loan == p['loan'] and c_inc == p['income'] and
                    c_rep == p['years'] and c_bor == p['months_borrow'] and 
                    c_gra == p['months_grace'] and c_buf == p['buffer'] and
                    c_hou == p['house_year'] and c_gro == p['income_growth']
                )
            except:
                matches = False
            
            if matches:
                return [dash.no_update] * (2 + len(FIN_INPUTS))
            else:
                return ['custom', "✏️ Je gebruikt aangepaste instellingen"] + [dash.no_update] * len(FIN_INPUTS)

    return [dash.no_update] * (2 + len(FIN_INPUTS))

@app.callback([Output('preset-selector', 'value'), Output('return-slider', 'value'), Output('vol-slider', 'value')],
              [Input('preset-selector', 'value'), Input('return-slider', 'value'), Input('vol-slider', 'value')])
def sync_presets_and_sliders(preset_val, ret_val, vol_val):
    trigger_id = ctx.triggered_id
    if trigger_id == 'preset-selector':
        if preset_val == 'vuaa': return 'vuaa', 10.5, 16.0
        elif preset_val == 'vwce': return 'vwce', 8.5, 14.0
        else: return 'custom', ret_val, vol_val
    else:
        if ret_val == 10.5 and vol_val == 16.0: return 'vuaa', 10.5, 16.0
        elif ret_val == 8.5 and vol_val == 14.0: return 'vwce', 8.5, 14.0
        else: return 'custom', ret_val, vol_val

@app.callback(Output('borrowing-inputs-container', 'style'), Input('study-status-toggle', 'value'))
def toggle_borrowing_inputs(status): return {'display': 'none'} if status == 'finished' else {'display': 'block'}

@app.callback(Output('howto-content', 'style'), Input('btn-howto', 'n_clicks'), State('howto-content', 'style'))
def toggle_howto(n, current_style):
    if current_style is None: current_style = {'display': 'none'}
    n = n or 0
    return {'display': 'block'} if n % 2 == 1 else {'display': 'none'}

@app.callback(
    [Output('simulation-results-store', 'data'), Output('error-message', 'children')],
    [Input('btn-run', 'n_clicks')],
    [State('sim-years', 'value'), 
     State('return-slider', 'value'), State('duo-rate', 'value'), State('inflation-slider', 'value'),
     State('loan-monthly', 'value'),
     State('curr-invest', 'value'), State('curr-savings', 'value'), State('curr-debt', 'value'),
     State('months-borrow', 'value'), State('months-grace', 'value'), State('repay-years', 'value'),
     State('house-year', 'value'),
     State('tax-rate-input', 'value'), State('tax-free-input', 'value'), State('tax-year-input', 'value'),
     State('fict-sav', 'value'), State('fict-inv', 'value'), State('fict-debt', 'value'),
     State('vol-slider', 'value'),
     State('study-status-toggle', 'value'),
     State('emergency-buffer', 'value'),
     State('use-draagkracht', 'value'), State('start-income', 'value'), State('income-growth', 'value'),
     State('savings-rate-slider', 'value'),
     State('has-partner', 'value')]
)
def run_calculations(n_clicks, sim_years, ret, duo, inf, loan, cur_inv, cur_sav, cur_debt, m_borrow, m_grace, repay_yrs, h_year, 
                     tax_rate, tax_free, tax_year, f_sav, f_inv, f_debt, vol, study_status, emerg_buffer,
                     use_draagkracht_list, start_inc, inc_growth, savings_rate, has_partner_list):
    
    if not n_clicks: pass 

    sim_years = sim_years or 35
    loan = loan or 0; cur_inv = cur_inv or 0; cur_sav = cur_sav or 0; cur_debt = cur_debt or 0
    duo = duo or 2.57; inf = inf or 2.0; vol = vol or 15
    f_sav = f_sav or 1.03; f_inv = f_inv or 6.04; f_debt = f_debt or 2.46
    repay_yrs = repay_yrs or 35
    emerg_buffer = emerg_buffer or 0
    start_inc = start_inc or 40000
    inc_growth = inc_growth or 2.5
    savings_rate = savings_rate or 2.0
    tax_rate = tax_rate or 36
    
    use_draagkracht = True if (use_draagkracht_list and 'yes' in use_draagkracht_list) else False
    has_partner = True if (has_partner_list and 'yes' in has_partner_list) else False
    
    if study_status == 'finished': m_borrow = 0; loan = 0

    valid, msg = validate_inputs(cur_inv, cur_sav, cur_debt, loan)
    if not valid: return None, msg

    df, house_diff, house_diff_real, salary_effect, statutory_payment_b = run_simulation(
        sim_years, market_return_decimal=ret/100, rate_debt_decimal=duo/100, inflation_decimal=inf/100, rate_savings_decimal=savings_rate/100,
        tax_rate_decimal=tax_rate/100, tax_free_income_val=tax_free, year_new_tax_system=tax_year, 
        fict_sav_decimal=f_sav/100, fict_inv_decimal=f_inv/100, fict_debt_decimal=f_debt/100,
        house_year=h_year, current_invest=cur_inv, current_savings=cur_sav, current_debt=cur_debt,
        loan_monthly=loan, months_left_to_borrow=m_borrow, months_grace_remaining=m_grace,
        years_repayment_total=repay_yrs, emergency_buffer=emerg_buffer,
        use_draagkracht=use_draagkracht, start_income=start_inc, income_growth_decimal=inc_growth/100,
        has_partner=has_partner
    )
    
    mc_years, p10, p50, p90, end_real_mc = run_monte_carlo(
        sim_years, 500, vol/100, ret/100, duo/100, inf/100, savings_rate/100,
        tax_rate/100, tax_free, tax_year, f_sav/100, f_inv/100, f_debt/100, 
        h_year, cur_inv, cur_sav, cur_debt, loan, m_borrow, m_grace, repay_yrs, emerg_buffer,
        use_draagkracht, start_inc, inc_growth/100, has_partner
    )
    
    gross_be_return = duo / (1 - (tax_rate/100))
    
    results = {
        'df': df.to_dict('records'),
        'house_diff': house_diff,
        'house_diff_real': house_diff_real,
        'salary_effect': salary_effect,
        'statutory_payment': statutory_payment_b,
        'mc': {'years': mc_years.tolist(), 'p10': p10, 'p50': p50, 'p90': p90, 'end_real_mc': end_real_mc.tolist()},
        'house_year': h_year,
        'sim_years': sim_years,
        'break_even_gross': gross_be_return
    }
    
    return results, ""

@app.callback(
    [Output('main-graph', 'figure'), Output('diff-graph', 'figure'),
     Output('kpi-end', 'children'), Output('kpi-end', 'style'), Output('kpi-end-real', 'children'),
     Output('kpi-salary', 'children'), Output('kpi-monthly-payment', 'children'),
     Output('kpi-house', 'children'), Output('kpi-house', 'style'), Output('kpi-house-real', 'children'),
     Output('kpi-breakeven', 'children'), Output('kpi-breakeven-sub', 'children'),
     Output('table-container', 'children'),
     Output('mc-fan', 'figure'), Output('mc-hist', 'figure'), Output('mc-stats', 'children')],
    [Input('simulation-results-store', 'data')]
)
def update_visuals(data):
    if not data: return go.Figure(), go.Figure(), "...", {}, "", "...", "...", "...", {}, "", "...", "...", "", go.Figure(), go.Figure(), ""
    
    df = pd.DataFrame(data['df'])
    h_year = data['house_year']
    sim_years = data['sim_years']
    
    fig_main = make_subplots(specs=[[{"secondary_y": True}]])
    ht = "<b>%{fullData.name}</b><br>€ %{y:,.0f}<extra></extra>"
    fig_main.add_trace(go.Scatter(x=df['Jaar'], y=df['Netto_B'], name='Netto B (Lenen)', line=dict(color='#e67e22', width=3), hovertemplate=ht), secondary_y=False)
    fig_main.add_trace(go.Scatter(x=df['Jaar'], y=df['Netto_A'], name='Netto A (Aflossen)', line=dict(color='#2980b9', width=3), hovertemplate=ht), secondary_y=False)
    fig_main.add_trace(go.Scatter(x=df['Jaar'], y=df['Netto_A_Base'], name='Netto A (Sparen)', line=dict(color='#aed6f1', width=2, dash='dot'), hovertemplate=ht), secondary_y=False)
    fig_main.add_trace(go.Scatter(x=df['Jaar'], y=df['Schuld_A'], name='Schuld A', line=dict(color='#800000', width=2, dash='dot'), hovertemplate=ht), secondary_y=True)
    fig_main.add_trace(go.Scatter(x=df['Jaar'], y=df['Schuld_B'], name='Schuld B', line=dict(color='#e74c3c', width=1), fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)', hovertemplate=ht), secondary_y=True)
    
    max_y = max(df['Netto_B'].max(), df['Netto_A'].max())
    fig_main.update_yaxes(range=[-10000, max_y * 1.1], secondary_y=False)
    
    fig_main.update_layout(template='plotly_white', hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0), margin=dict(l=40, r=40, t=20, b=40))
    fig_main.update_yaxes(title_text="Netto Vermogen (€)", tickformat=",.0f", secondary_y=False)
    fig_main.update_yaxes(title_text="Schuld (€)", tickformat=",.0f", secondary_y=True, showgrid=False, color='#c0392b')

    fig_diff = go.Figure()
    end_val = df['Voordeel_B'].iloc[-1]
    c_fill = '#27ae60' if end_val > 0 else '#c0392b'
    fig_diff.add_trace(go.Scatter(x=df['Jaar'], y=df['Voordeel_B'], name='Nominaal', line=dict(color=c_fill, width=2), fill='tozeroy', hovertemplate=ht))
    fig_diff.add_trace(go.Scatter(x=df['Jaar'], y=df['Voordeel_B_Real'], name='Reëel (Koopkracht)', line=dict(color='#34495e', width=2, dash='dot'), hovertemplate=ht))
    if (h_year*12)-1 < len(df):
        h_val = df.iloc[int(h_year*12)-1]['Voordeel_B']
        fig_diff.add_trace(go.Scatter(x=[h_year], y=[h_val], mode='markers+text', marker=dict(size=12, color='#2c3e50', symbol='diamond'), name='Huis Koop Moment', text=["🏠"], textposition="top center", hovertemplate=ht))
    fig_diff.update_layout(template='plotly_white', margin=dict(l=40, r=20, t=20, b=40), xaxis_title="Jaren", yaxis_title="Voordeel (€)")

    end_real = df['Voordeel_B_Real'].iloc[-1]
    kpi_end = f"€ {end_val:,.0f}"
    st_end = {'color': '#27ae60' if end_val > 0 else '#c0392b', 'fontSize': '28px', 'fontWeight': 'bold', 'marginTop': '5px'}
    kpi_h = f"€ {data['house_diff']:,.0f}"
    st_h = {'color': '#27ae60' if data['house_diff'] > 0 else '#c0392b', 'fontSize': '28px', 'fontWeight': 'bold', 'marginTop': '5px'}
    
    be_val = data.get('break_even_gross', 0)
    kpi_be = f"{be_val:.2f}%"
    kpi_be_sub = f"Bij 36% belasting & leningrente"
    
    txt_salary = f"€ {data['salary_effect']:,.0f}"
    txt_payment = f"Start Maandbedrag DUO: € {data['statutory_payment']:.0f}"

    years_to_show = list(range(5, sim_years + 1, 5))
    if sim_years not in years_to_show: years_to_show.append(sim_years)
    rows = []
    for y in years_to_show:
        idx = int(y*12)-1
        if idx < len(df):
            r = df.iloc[idx]
            rows.append({'Jaar': y, 'Netto A': r['Netto_A'], 'Netto B': r['Netto_B'], 'Aflos B': r['Cashflow_B_Pay'], 'Inleg A': r['Cashflow_A_Invest'], 'Voordeel': r['Voordeel_B']})
    
    money_fmt = FormatTemplate.money(0)
    columns = [{'name': i, 'id': i, 'type': 'numeric', 'format': money_fmt} for i in ['Netto A', 'Netto B', 'Aflos B', 'Inleg A', 'Voordeel']]
    columns.insert(0, {'name': 'Jaar', 'id': 'Jaar'})
    tbl = dash_table.DataTable(data=rows, columns=columns, style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}, style_cell={'textAlign': 'center', 'fontFamily': '"Segoe UI"'}, style_data_conditional=[{'if': {'column_id': 'Voordeel'}, 'color': '#27ae60', 'fontWeight': 'bold'}])

    mc = data['mc']
    fig_fan = go.Figure()
    fig_fan.add_trace(go.Scatter(x=mc['years'] + mc['years'][::-1], y=mc['p90'] + mc['p10'][::-1], fill='toself', fillcolor='rgba(39, 174, 96, 0.2)', line=dict(color='rgba(255,255,255,0)'), name='80% Interval', hovertemplate=ht))
    fig_fan.add_trace(go.Scatter(x=mc['years'], y=mc['p50'], name='Mediaan', line=dict(color='#27ae60', width=3), hovertemplate=ht))
    fig_fan.add_hline(y=0, line_color='black')
    fig_fan.update_layout(template='plotly_white', margin=dict(l=40, r=20, t=20, b=40), xaxis_title="Jaren", yaxis_title="Voordeel (€)", hovermode="x unified")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=mc['end_real_mc'], nbinsx=40, marker_color='#3498db', opacity=0.75, name='Simulaties'))
    fig_hist.add_vline(x=0, line_color='red', line_width=2)
    fig_hist.update_layout(template='plotly_white', margin=dict(l=40, r=20, t=20, b=40), xaxis_title="Eindvoordeel (Reëel, €)", yaxis_title="Simulaties")
    
    prob_win = np.mean(np.array(mc['end_real_mc']) > 0) * 100
    mc_txt = [html.P([html.B("Kans op Winst: "), f"{prob_win:.1f}%"], style={'color': '#27ae60' if prob_win>50 else '#c0392b', 'fontSize': '20px'}), html.P([html.B("Mediaan: "), f"€ {mc['p50'][-1]:,.0f}"]), html.P([html.B("Best case (95%): "), f"€ {np.percentile(mc['end_real_mc'], 95):,.0f}"]), html.P([html.B("Worst case (5%): "), f"€ {np.percentile(mc['end_real_mc'], 5):,.0f}"])]

    return fig_main, fig_diff, kpi_end, st_end, f"Koopkracht: € {end_real:,.0f}", txt_salary, txt_payment, kpi_h, st_h, f"Koopkracht: € {data['house_diff_real']:,.0f}", kpi_be, kpi_be_sub, tbl, fig_fan, fig_hist, mc_txt

@app.callback(Output("download-dataframe-csv", "data"), Input("btn-download", "n_clicks"), State("simulation-results-store", "data"), prevent_initial_call=True)
def download_csv(n_clicks, data):
    if not data: return None
    df = pd.DataFrame(data['df'])
    return dcc.send_data_frame(df.to_csv, "duo_scenario_export.csv")

if __name__ == '__main__':
    print("DASHBOARD V34 (Grid Layout + Scroll Fix): http://127.0.0.1:8051/")
    app.run_server(debug=True, port=8051)