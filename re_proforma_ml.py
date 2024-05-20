#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

console = Console()

RENTAL_PROFORMA_LOGO = """
[bold cyan]
 ____                    __     __   __      __         
|  _ \                   \ \    \ \ / /      \ \        
| |_) )__ _  _____ __  __ \ \    \ v / __  __ \ \__   __
|  __/ __) |/ (   )  \/ /  > \    > < /  \/ /  > \ \ / /
| |  > _)| / / | ( ()  <  / ^ \  / ^ ( ()  <  / ^ \ v / 
|_|  \___)__/   \_)__/\_\/_/ \_\/_/ \_\__/\_\/_/ \_> <  
                                                  / ^ \ 
                                                 /_/ \_\ 
[bold yellow]Proforma Financial Analysis[/bold yellow]
"""

def get_user_input():
    console.print("[bold cyan]Enter the required details for proforma calculation[/bold cyan]")
    
    property_name = Prompt.ask("Property Name")
    purchase_price = Prompt.ask("Purchase Price", default="0", convert=float)
    rental_income = Prompt.ask("Monthly Rental Income", default="0", convert=float)
    operating_expenses = Prompt.ask("Monthly Operating Expenses", default="0", convert=float)
    vacancy_rate = Prompt.ask("Vacancy Rate (in %)", default="0", convert=float) / 100
    loan_amount = Prompt.ask("Loan Amount", default="0", convert=float)
    interest_rate = Prompt.ask("Interest Rate (in %)", default="0", convert=float) / 100
    loan_term_years = Prompt.ask("Loan Term (years)", default="0", convert=int)
    
    return {
        "property_name": property_name,
        "purchase_price": purchase_price,
        "rental_income": rental_income,
        "operating_expenses": operating_expenses,
        "vacancy_rate": vacancy_rate,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "loan_term_years": loan_term_years
    }

def calculate_proforma(details):
    monthly_income = details["rental_income"]
    vacancy_loss = monthly_income * details["vacancy_rate"]
    net_rental_income = monthly_income - vacancy_loss
    annual_rental_income = net_rental_income * 12
    annual_operating_expenses = details["operating_expenses"] * 12
    noi = annual_rental_income - annual_operating_expenses
    
    monthly_interest_rate = details["interest_rate"] / 12
    num_payments = details["loan_term_years"] * 12
    if monthly_interest_rate > 0:
        monthly_mortgage_payment = (details["loan_amount"] * monthly_interest_rate) / \
                                   (1 - (1 + monthly_interest_rate) ** -num_payments)
    else:
        monthly_mortgage_payment = details["loan_amount"] / num_payments if num_payments > 0 else 0
    
    annual_debt_service = monthly_mortgage_payment * 12
    cash_flow = noi - annual_debt_service
    
    return {
        "noi": noi,
        "annual_debt_service": annual_debt_service,
        "cash_flow": cash_flow
    }

def display_proforma(details, results):
    table = Table(title="Proforma Analysis", box=box.ROUNDED)
    
    table.add_column("Description", style="cyan", no_wrap=True)
    table.add_column("Amount", style="magenta")
    
    table.add_row("Property Name", details["property_name"])
    table.add_row("Purchase Price", f"${details['purchase_price']:,.2f}")
    table.add_row("Monthly Rental Income", f"${details['rental_income']:,.2f}")
    table.add_row("Monthly Operating Expenses", f"${details['operating_expenses']:,.2f}")
    table.add_row("Vacancy Rate", f"{details['vacancy_rate'] * 100:.2f}%")
    table.add_row("Loan Amount", f"${details['loan_amount']:,.2f}")
    table.add_row("Interest Rate", f"{details['interest_rate'] * 100:.2f}%")
    table.add_row("Loan Term (years)", str(details["loan_term_years"]))
    table.add_row("Net Operating Income (NOI)", f"${results['noi']:,.2f}")
    table.add_row("Annual Debt Service", f"${results['annual_debt_service']:,.2f}")
    table.add_row("Annual Cash Flow", f"${results['cash_flow']:,.2f}", style="bold green" if results["cash_flow"] > 0 else "bold red")
    
    console.print(table)

def save_proforma(details, results):
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"proforma_{details['property_name']}_{current_date}.csv"
    
    data = {
        "Property Name": [details["property_name"]],
        "Purchase Price": [details["purchase_price"]],
        "Monthly Rental Income": [details["rental_income"]],
        "Monthly Operating Expenses": [details["operating_expenses"]],
        "Vacancy Rate (%)": [details["vacancy_rate"] * 100],
        "Loan Amount": [details["loan_amount"]],
        "Interest Rate (%)": [details["interest_rate"] * 100],
        "Loan Term (years)": [details["loan_term_years"]],
        "Net Operating Income (NOI)": [results["noi"]],
        "Annual Debt Service": [results["annual_debt_service"]],
        "Annual Cash Flow": [results["cash_flow"]]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
    console.print(f"Proforma saved as {filename}", style="bold green")

def train_rental_income_model():
    # Example historical data for rental income (replace with actual data)
    historical_data = {
        "year": [2015, 2016, 2017, 2018, 2019, 2020, 2021],
        "monthly_rental_income": [2000, 2100, 2200, 2300, 2400, 2500, 2600]
    }
    df = pd.DataFrame(historical_data)
    X = df["year"].values.reshape(-1, 1)
    y = df["monthly_rental_income"].values

    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_future_rental_income(model, current_year, years_ahead):
    future_years = np.array([current_year + i for i in range(1, years_ahead + 1)]).reshape(-1, 1)
    predictions = model.predict(future_years)
    return future_years.flatten(), predictions

def main():
    console.print(RENTAL_PROFORMA_LOGO)
    
    details = get_user_input()
    results = calculate_proforma(details)
    
    display_proforma(details, results)
    
    if Confirm.ask("[bold cyan]Do you want to save the proforma to a CSV file?[/bold cyan]"):
        save_proforma(details, results)
    
    if Confirm.ask("[bold cyan]Do you want to predict future rental income?[/bold cyan]"):
        model = train_rental_income_model()
        current_year = datetime.now().year
        future_years, predictions = predict_future_rental_income(model, current_year, 5)
        
        for year, prediction in zip(future_years, predictions):
            console.print(f"Predicted rental income for {year}: ${prediction:.2f}")
        
        # Plotting predictions
        plt.plot(future_years, predictions, marker='o', linestyle='-', color='blue')
        plt.title('Future Rental Income Predictions')
        plt.xlabel('Year')
        plt.ylabel('Monthly Rental Income')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
