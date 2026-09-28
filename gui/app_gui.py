"""
Small Tkinter front-end for the income classifier.

Two-column form (numeric fields on the left, category drop-downs on the
right) instead of one long stacked list, so the whole thing fits without
scrolling on a small screen.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from inference import predict

CATEGORY_OPTIONS = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                  "Local-gov", "State-gov", "Without-pay", "Never-worked", "Unknown"],
    "education": ["No-HS", "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm",
                  "Bachelors", "Masters", "Prof-school", "Doctorate"],
    "marital_status": ["Married-civ-spouse", "Never-married", "Divorced", "Separated",
                        "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales",
                   "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                   "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                   "Transport-moving", "Priv-house-serv", "Protective-serv",
                   "Armed-Forces", "Unknown"],
    "relationship": ["Husband", "Wife", "Own-child", "Not-in-family",
                      "Other-relative", "Unmarried"],
    "race": ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
    "gender": ["Male", "Female"],
    "native_country": ["United-States", "Mexico", "Philippines", "Germany", "Canada",
                       "India", "England", "China", "Puerto-Rico", "Cuba", "Other",
                       "Unknown"],
}

NUMERIC_DEFAULTS = {
    "age": "35",
    "capital_gain": "0",
    "capital_loss": "0",
    "hours_per_week": "40",
}


class IncomeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Income Predictor")
        self.resizable(False, False)

        self.numeric_vars = {}
        self.category_vars = {}

        self._build_numeric_column()
        self._build_category_column()
        self._build_actions_row()

    def _build_numeric_column(self):
        frame = ttk.LabelFrame(self, text="Numeric fields")
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        for i, (field, default) in enumerate(NUMERIC_DEFAULTS.items()):
            ttk.Label(frame, text=field.replace("_", " ").title()).grid(
                row=i, column=0, sticky="w", padx=6, pady=4
            )
            var = tk.StringVar(value=default)
            ttk.Entry(frame, textvariable=var, width=15).grid(
                row=i, column=1, padx=6, pady=4
            )
            self.numeric_vars[field] = var

    def _build_category_column(self):
        frame = ttk.LabelFrame(self, text="Category fields")
        frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        for i, (field, options) in enumerate(CATEGORY_OPTIONS.items()):
            ttk.Label(frame, text=field.replace("_", " ").title()).grid(
                row=i, column=0, sticky="w", padx=6, pady=4
            )
            var = tk.StringVar(value=options[0])
            combo = ttk.Combobox(
                frame, textvariable=var, values=options, state="readonly", width=22
            )
            combo.grid(row=i, column=1, padx=6, pady=4)
            self.category_vars[field] = var

    def _build_actions_row(self):
        ttk.Button(self, text="Predict", command=self._on_predict).grid(
            row=1, column=0, columnspan=2, pady=(0, 6)
        )

        self.result_var = tk.StringVar(value="—")
        ttk.Label(
            self, textvariable=self.result_var, font=("Segoe UI", 11), justify="left"
        ).grid(row=2, column=0, columnspan=2, pady=(0, 12))

    def _on_predict(self):
        try:
            raw = {field: float(var.get()) for field, var in self.numeric_vars.items()}
        except ValueError:
            messagebox.showerror(
                "Invalid input", "Age, capital gain/loss and hours must all be numbers."
            )
            return

        raw.update({field: var.get() for field, var in self.category_vars.items()})

        try:
            label, probability = predict(raw)
        except Exception as exc:  # surfaces a loading/artifact problem clearly
            messagebox.showerror("Prediction failed", str(exc))
            return

        tag = ">50K" if label == 1 else "<=50K"
        self.result_var.set(f"Prediction: {tag}\nP(income > 50K): {probability:.1%}")


if __name__ == "__main__":
    IncomeApp().mainloop()
