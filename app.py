"""
Multi-Feature Indian Salary Prediction Desktop Application
---------------------------------------------------------
This application uses a pre-trained machine learning model to predict an employee's
salary in Indian Rupees (₹) based on 8 demographic and job-related inputs.

Displays:
1. Annual Salary (₹)
2. Monthly Salary (₹)
3. Salary Category (Entry, Mid, Senior, High-Earner)
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd


class SalaryPredictionApp:
    """
    Main GUI application class for Multi-Feature Salary Prediction.
    """

    def __init__(self, root: tk.Tk):
        """
        Initializes the application, loads pre-trained model/preprocessor,
        and constructs the 2-column input UI layout.
        """
        self.root = root
        self.root.title("Salary Prediction System")
        self.root.configure(bg="#F0F2F5")

        # Adjust window dimensions for 8 inputs and results
        window_width = 650
        window_height = 550
        self.center_window(window_width, window_height)
        self.root.resizable(False, False)

        # File paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "models", "model.pkl")
        self.preprocessor_path = os.path.join(base_dir, "models", "preprocessor.pkl")

        # Model variables
        self.model = None
        self.preprocessor = None
        self.model_loaded = False

        # Load models
        self.load_model_artifacts()

        # Build UI
        self.create_widgets()

    def center_window(self, width: int, height: int) -> None:
        """Centers the window on the screen."""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def load_model_artifacts(self) -> None:
        """Loads model and preprocessing transformer from disk."""
        try:
            if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError("Model or preprocessor missing. Run train_model.py first.")
            
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            self.model_loaded = True
            print("Model and Preprocessor loaded successfully.")
        except Exception as e:
            self.model_loaded = False
            print(f"Error loading model artifacts: {e}")

    def create_widgets(self) -> None:
        """Creates a clean, modern form layout with 2 columns of inputs."""
        # --- Header ---
        header_frame = tk.Frame(self.root, bg="#3B5998", height=55)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        header_label = tk.Label(
            header_frame,
            text="Indian Salary Prediction System (₹)",
            fg="#FFFFFF",
            bg="#3B5998",
            font=("Helvetica", 15, "bold")
        )
        header_label.pack(expand=True)

        # --- Main Layout Form Frame ---
        form_frame = tk.Frame(self.root, bg="#FFFFFF", bd=1, relief=tk.SOLID)
        form_frame.pack(padx=20, pady=15, fill=tk.BOTH, expand=True)

        # Input label layout style
        lbl_style = {"bg": "#FFFFFF", "fg": "#4A5568", "font": ("Helvetica", 10, "bold")}
        
        # Grid layout configurations for form fields
        grid_padding = {"padx": 15, "pady": 6}

        # --- Column 1 Inputs ---
        
        # 1. Age
        tk.Label(form_frame, text="Age:", **lbl_style).grid(row=0, column=0, sticky=tk.W, **grid_padding)
        self.age_entry = tk.Entry(form_frame, font=("Helvetica", 10), bg="#F7FAFC", bd=1, relief=tk.SOLID)
        self.age_entry.grid(row=0, column=1, sticky=tk.EW, **grid_padding)
        self.age_entry.focus_set()

        # 2. Education Level
        tk.Label(form_frame, text="Education Level:", **lbl_style).grid(row=1, column=0, sticky=tk.W, **grid_padding)
        self.edu_combo = ttk.Combobox(
            form_frame,
            values=["Bachelor's", "Master's", "PhD"],
            font=("Helvetica", 10),
            state="readonly"
        )
        self.edu_combo.grid(row=1, column=1, sticky=tk.EW, **grid_padding)

        # 3. Years of Experience
        tk.Label(form_frame, text="Years of Experience:", **lbl_style).grid(row=2, column=0, sticky=tk.W, **grid_padding)
        self.exp_entry = tk.Entry(form_frame, font=("Helvetica", 10), bg="#F7FAFC", bd=1, relief=tk.SOLID)
        self.exp_entry.grid(row=2, column=1, sticky=tk.EW, **grid_padding)

        # 4. Job Title
        tk.Label(form_frame, text="Job Title:", **lbl_style).grid(row=3, column=0, sticky=tk.W, **grid_padding)
        self.job_combo = ttk.Combobox(
            form_frame,
            values=["Software Engineer", "Data Scientist", "Manager", "Analyst", "HR Specialist"],
            font=("Helvetica", 10),
            state="readonly"
        )
        self.job_combo.grid(row=3, column=1, sticky=tk.EW, **grid_padding)

        # --- Column 2 Inputs ---
        
        # 5. City
        tk.Label(form_frame, text="City:", **lbl_style).grid(row=0, column=2, sticky=tk.W, **grid_padding)
        self.city_combo = ttk.Combobox(
            form_frame,
            values=["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai"],
            font=("Helvetica", 10),
            state="readonly"
        )
        self.city_combo.grid(row=0, column=3, sticky=tk.EW, **grid_padding)

        # 6. Company Size
        tk.Label(form_frame, text="Company Size:", **lbl_style).grid(row=1, column=2, sticky=tk.W, **grid_padding)
        self.size_combo = ttk.Combobox(
            form_frame,
            values=["Small", "Medium", "Large"],
            font=("Helvetica", 10),
            state="readonly"
        )
        self.size_combo.grid(row=1, column=3, sticky=tk.EW, **grid_padding)

        # 7. Industry
        tk.Label(form_frame, text="Industry:", **lbl_style).grid(row=2, column=2, sticky=tk.W, **grid_padding)
        self.industry_combo = ttk.Combobox(
            form_frame,
            values=["IT", "Finance", "Healthcare", "Education", "Manufacturing"],
            font=("Helvetica", 10),
            state="readonly"
        )
        self.industry_combo.grid(row=2, column=3, sticky=tk.EW, **grid_padding)

        # 8. Work Type
        tk.Label(form_frame, text="Work Type:", **lbl_style).grid(row=3, column=2, sticky=tk.W, **grid_padding)
        self.work_combo = ttk.Combobox(
            form_frame,
            values=["Remote", "Hybrid", "On-site"],
            font=("Helvetica", 10),
            state="readonly"
        )
        self.work_combo.grid(row=3, column=3, sticky=tk.EW, **grid_padding)

        # Ensure grid columns expand evenly
        form_frame.columnconfigure(1, weight=1)
        form_frame.columnconfigure(3, weight=1)

        # --- Display Prediction Results Frame ---
        result_frame = tk.Frame(form_frame, bg="#EDF2F7", bd=1, relief=tk.FLAT)
        result_frame.grid(row=4, column=0, columnspan=4, sticky=tk.NSEW, padx=15, pady=12, ipady=5)

        self.annual_lbl = tk.Label(
            result_frame,
            text="Annual Salary: ₹0.00",
            bg="#EDF2F7",
            fg="#2B6CB0",
            font=("Helvetica", 11, "bold")
        )
        self.annual_lbl.pack(pady=2)

        self.monthly_lbl = tk.Label(
            result_frame,
            text="Monthly Salary: ₹0.00",
            bg="#EDF2F7",
            fg="#2B6CB0",
            font=("Helvetica", 11, "bold")
        )
        self.monthly_lbl.pack(pady=2)

        self.category_lbl = tk.Label(
            result_frame,
            text="Salary Category: -",
            bg="#EDF2F7",
            fg="#4A5568",
            font=("Helvetica", 11, "bold")
        )
        self.category_lbl.pack(pady=2)

        # --- Action Buttons Layout ---
        btn_frame = tk.Frame(form_frame, bg="#FFFFFF")
        btn_frame.grid(row=5, column=0, columnspan=4, sticky=tk.EW, padx=15, pady=(5, 10))

        btn_style = {
            "font": ("Helvetica", 10, "bold"),
            "bd": 0,
            "cursor": "hand2",
            "activeforeground": "#FFFFFF"
        }

        # Predict Button
        self.predict_btn = tk.Button(
            btn_frame,
            text="Predict Salary",
            bg="#3182CE",
            fg="#FFFFFF",
            activebackground="#2B6CB0",
            command=self.predict_salary,
            **btn_style
        )
        self.predict_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5), ipady=8)

        # Clear Button
        self.clear_btn = tk.Button(
            btn_frame,
            text="Clear",
            bg="#E2E8F0",
            fg="#4A5568",
            activebackground="#CBD5E0",
            command=self.clear_fields,
            **btn_style
        )
        self.clear_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, ipady=8)

        # Exit Button
        self.exit_btn = tk.Button(
            btn_frame,
            text="Exit",
            bg="#E53E3E",
            fg="#FFFFFF",
            activebackground="#C53030",
            command=self.exit_app,
            **btn_style
        )
        self.exit_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0), ipady=8)

        # --- Status Bar ---
        self.status_label = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg="#E2E8F0",
            fg="#4A5568",
            font=("Helvetica", 9)
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Handle model loading failure
        if not self.model_loaded:
            self.show_status("Error: Trained model artifacts not found! Run train_model.py first.", "red")
            self.predict_btn.config(state=tk.DISABLED, bg="#CBD5E0", cursor="arrow")
            messagebox.showerror(
                "Model Loading Error",
                "Trained model artifacts ('model.pkl', 'preprocessor.pkl') were not found in 'models/' folder.\n\n"
                "Please run 'train_model.py' to compare models and save artifacts."
            )

    def show_status(self, message: str, color: str = "#4A5568") -> None:
        """Updates the text and foreground color of the status label."""
        self.status_label.config(text=message, fg=color)

    def validate_inputs(self) -> tuple[bool, dict | str]:
        """
        Validates all 8 input fields from entry boxes and dropdown comboboxes.
        
        Rules:
        - All fields are mandatory.
        - Numeric fields (Age, Years of Experience) must be valid numbers.
        - Age must be >= 18.
        - Experience must be >= 0.
        - Age must be >= Years of Experience + 18.
        
        Returns:
            tuple[bool, dict_of_values | error_string]
        """
        # Read text values
        age_str = self.age_entry.get().strip()
        exp_str = self.exp_entry.get().strip()
        
        edu = self.edu_combo.get().strip()
        job = self.job_combo.get().strip()
        city = self.city_combo.get().strip()
        size = self.size_combo.get().strip()
        ind = self.industry_combo.get().strip()
        work = self.work_combo.get().strip()

        # 1. Mandatory check
        if not all([age_str, exp_str, edu, job, city, size, ind, work]):
            return False, "Error: All fields are mandatory. Please fill in all entries."

        # 2. Age parsing check
        try:
            age = float(age_str)
        except ValueError:
            return False, "Error: Age must be a valid number."

        if age < 18:
            return False, "Error: Age must be 18 or above."

        # 3. Experience parsing check
        try:
            exp = float(exp_str)
        except ValueError:
            return False, "Error: Years of Experience must be a valid number."

        if exp < 0:
            return False, "Error: Years of Experience cannot be negative."

        # 4. Logical check: Age vs Experience
        if age < (exp + 18):
            return False, "Error: Age is too low for this amount of experience (Age >= Experience + 18)."

        # Return dict of parsed values
        return True, {
            "Age": age,
            "YearsExperience": exp,
            "EducationLevel": edu,
            "JobTitle": job,
            "City": city,
            "CompanySize": size,
            "Industry": ind,
            "WorkType": work,
        }

    def predict_salary(self) -> None:
        """Triggers input validation, scales features, runs prediction, and categorizes salary."""
        if not self.model_loaded:
            self.show_status("Error: No model loaded. Cannot predict.", "red")
            return

        is_valid, result = self.validate_inputs()

        if not is_valid:
            self.show_status(result, "red")
            self.reset_result_display(error=True)
            return

        try:
            # Create a 1-row Pandas DataFrame with exact column order
            input_df = pd.DataFrame([result])
            
            # The Columns must be in the exact order fitted by the ColumnTransformer
            columns_order = [
                "Age",
                "YearsExperience",
                "EducationLevel",
                "JobTitle",
                "City",
                "CompanySize",
                "Industry",
                "WorkType",
            ]
            input_df = input_df[columns_order]

            # Preprocess the input data
            input_processed = self.preprocessor.transform(input_df)

            # Predict annual salary
            pred_annual = self.model.predict(input_processed)[0]
            pred_annual = max(0, pred_annual)  # Enforce non-negative salary

            # Calculate monthly salary
            pred_monthly = pred_annual / 12.0

            # Determine category
            if pred_annual < 600000.0:
                category = "Entry Level 🆕"
            elif pred_annual < 1500000.0:
                category = "Mid Level 📈"
            elif pred_annual < 3000000.0:
                category = "Senior Level 🌟"
            else:
                category = "High Earner 👑"

            # Format outputs (Indian Rupee formatting or basic comma formatting)
            formatted_annual = f"₹{pred_annual:,.2f}"
            formatted_monthly = f"₹{pred_monthly:,.2f}"

            # Update Labels
            self.annual_lbl.config(text=f"Annual Salary: {formatted_annual}", fg="#2F855A")
            self.monthly_lbl.config(text=f"Monthly Salary: {formatted_monthly}", fg="#2F855A")
            self.category_lbl.config(text=f"Salary Category: {category}", fg="#2F855A")

            self.show_status("Success: Salary predicted successfully!", "green")

        except Exception as e:
            self.show_status(f"Prediction Error: {str(e)}", "red")
            self.reset_result_display(error=True)

    def reset_result_display(self, error: bool = False) -> None:
        """Helper to reset result labels."""
        fg_color = "#E53E3E" if error else "#2B6CB0"
        self.annual_lbl.config(text="Annual Salary: ₹0.00", fg=fg_color)
        self.monthly_lbl.config(text="Monthly Salary: ₹0.00", fg=fg_color)
        self.category_lbl.config(text="Salary Category: -", fg="#4A5568")

    def clear_fields(self) -> None:
        """Resets all entry boxes, drop-downs, result displays, and status text."""
        # Clear Entries
        self.age_entry.delete(0, tk.END)
        self.exp_entry.delete(0, tk.END)

        # Clear Comboboxes
        self.edu_combo.set("")
        self.job_combo.set("")
        self.city_combo.set("")
        self.size_combo.set("")
        self.industry_combo.set("")
        self.work_combo.set("")

        # Reset Results
        self.reset_result_display()

        # Reset Status
        if self.model_loaded:
            self.show_status("Ready", "#4A5568")
        else:
            self.show_status("Error: Trained model artifacts not found! Run train_model.py first.", "red")

        # Refocus on the first field (Age)
        self.age_entry.focus_set()

    def exit_app(self) -> None:
        """Closes the Tkinter application."""
        self.root.destroy()


if __name__ == "__main__":
    root_window = tk.Tk()
    app = SalaryPredictionApp(root_window)
    root_window.mainloop()
