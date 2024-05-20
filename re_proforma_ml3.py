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
import numpy_financial as npf

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
    
    while True:
        try:
            property_name = Prompt.ask("Property Name")
            purchase_price = float(Prompt.ask("Purchase Price", default="0"))
            rental_income = float(Prompt.ask("Monthly Rental Income", default="0"))
            operating_expenses = float(Prompt.ask("Monthly Operating Expenses", default="0"))
            vacancy_rate = float(Prompt.ask("Vacancy Rate (in %)", default="0")) / 100
            loan_amount = float(Prompt.ask("Loan Amount", default="0").replace("'", ""))
            interest_rate = float(Prompt.ask("Interest Rate (in %)", default="0")) / 100
            loan_term_years = int(Prompt.ask("Loan Term (years)", default="0"))
            break
        except ValueError:
            console.print("[bold red]Invalid input! Please enter numeric values for prices, rates, and amounts.[/bold red]")
    
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
    
    # Calculate Cap Rate
    cap_rate = noi / details["purchase_price"]
    
    # Calculate Cash-on-Cash Return
    total_cash_invested = details["purchase_price"] - details["loan_amount"]
    cash_on_cash_return = (cash_flow / total_cash_invested) * 100
    
    # Calculate IRR (simplified)
    initial_investment = -total_cash_invested
    cash_flows = [cash_flow] * details["loan_term_years"]
    irr = npf.irr([initial_investment] + cash_flows) * 100
    
    return {
        "noi": noi,
        "annual_debt_service": annual_debt_service,
        "cash_flow": cash_flow,
        "cap_rate": cap_rate,
        "cash_on_cash_return": cash_on_cash_return,
        "irr": irr
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
    table.add_row("Cap Rate", f"{results['cap_rate'] * 100:.2f}%")
    table.add_row("Cash-on-Cash Return", f"{results['cash_on_cash_return']:.2f}%")
    table.add_row("Internal Rate of Return (IRR)", f"{results['irr']:.2f}%")
    
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
        "Annual Cash Flow": [results["cash_flow"]],
        "Cap Rate (%)": [results["cap_rate"] * 100],
        "Cash-on-Cash Return (%)": [results["cash_on_cash_return"]],
        "Internal Rate of Return (IRR) (%)": [results["irr"]]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
    console.print(f"Proforma saved as {filename}", style="bold green")

def train_rental_income_model():
    df = load_historical_data()
    if df is None:
        return None

    X = df["Year"].values.reshape(-1, 1)
    y = df["Monthly Rental Income"].values

    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_future_rental_income(model, current_year, years_ahead):
    future_years = np.array([current_year + i for i in range(1, years_ahead + 1)]).reshape(-1, 1)
    predictions = model.predict(future_years)
    return future_years.flatten(), predictions

def store_historical_data():
    console.print("[bold cyan]Enter historical rental income data[/bold cyan]")
    years = []
    incomes = []
    while True:
        year = int(Prompt.ask("Year"))
        income = float(Prompt.ask("Monthly Rental Income"))
        years.append(year)
        incomes.append(income)
        if not Confirm.ask("[bold cyan]Do you want to add another record?[/bold cyan]"):
            break
    
    data = {"Year": years, "Monthly Rental Income": incomes}
    df = pd.DataFrame(data)
    filename = "historical_rental_income.csv"
    df.to_csv(filename, index=False)
    console.print(f"Historical data saved as {filename}", style="bold green")

def load_historical_data():
    filename = "historical_rental_income.csv"
    try:
        df = pd.read_csv(filename)
        console.print(f"Loaded historical data from {filename}", style="bold green")
        return df
    except FileNotFoundError:
        console.print("[bold red]Historical data file not found! Please store historical data first.[/bold red]")
        return None

def visualize_rental_income(model, current_year, years_ahead):
    df = load_historical_data()
    if df is None:
        return

    historical_years = df["Year"].values
    historical_income = df["Monthly Rental Income"].values

    future_years, future_income = predict_future_rental_income(model, current_year, years_ahead)

    plt.figure(figsize=(10, 5))
    plt.plot(historical_years, historical_income, marker="o", label="Historical Income")
    plt.plot(future_years, future_income, marker="x", linestyle="--", label="Predicted Income")
    plt.xlabel("Year")
    plt.ylabel("Monthly Rental Income ($)")
    plt.title("Historical and Predicted Monthly Rental Income")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    console.print(RENTAL_PROFORMA_LOGO)
    while True:
        action = Prompt.ask("Choose an action: [bold cyan]calculate[/bold cyan] proforma, [bold cyan]store[/bold cyan] historical data, [bold cyan]visualize[/bold cyan] rental income, or [bold cyan]exit[/bold cyan]", choices=["calculate", "store", "visualize", "exit"])
        
        if action == "calculate":
            details = get_user_input()
            results = calculate_proforma(details)
            display_proforma(details, results)
            if Confirm.ask("Do you want to save the proforma?"):
                save_proforma(details, results)
        
        elif action == "store":
            store_historical_data()
        
        elif action == "visualize":
            model = train_rental_income_model()
            if model:
                current_year = int(datetime.now().strftime("%Y"))
                years_ahead = int(Prompt.ask("Enter the number of years to predict"))
                visualize_rental_income(model, current_year, years_ahead)
        
        elif action == "exit":
            console.print("Goodbye!", style="bold cyan")
            break
