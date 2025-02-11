import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import re
from fpdf import FPDF

def detect_delimiter(file_path):
    """Detects whether the CSV file is comma or semicolon separated."""
    with open(file_path, 'r', encoding='utf-8') as file:
        first_line = file.readline()
        return ';' if first_line.count(';') > first_line.count(',') else ','

def preprocess_data(df):
    """Preprocesses the data: converts numeric columns, replaces missing values, and filters out IDs."""
    for col in df.columns:
        df[col] = df[col].astype(str).str.lstrip("'")  # Remove leading quotes from Moodle export
        df[col] = df[col].replace("-", np.nan)  # Mark "-" as NaN (absent students)
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric where possible

    # Drop columns that are completely empty after removing non-numeric values
    df = df.dropna(axis=1, how='all')
    
    return df

def is_grade_column(series):
    """Determines if a column is a valid grade column by checking its value range and uniqueness."""
    if series.max() > 100:  # Values too large, likely an ID
        return False
    if len(series.unique()) == len(series):  # Likely an ID if all values are unique
        return False
    if series.nunique() == 1 and series.iloc[0] == 0:  # Ignore empty columns
        return False
    return True

def determine_max_score(series):
    """Determines the max score of the grade column (20 or 100)."""
    return 100 if series.max() > 20 else 20

def add_statistics_table(pdf, stats_inscrits, stats_presents, title, series):
    """Adds a formatted statistics table to the PDF in the lower half of the page, with enhanced min/max counts."""
    pdf.set_y(160)  # Move to the lower half of the page
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.set_font("Arial", size=10)

    max_score = 100 if stats_inscrits['max'] > 20 else 20
    threshold = 50 if max_score == 100 else 10

    # Count students who passed (>= threshold), ignoring NaN values
    count_above_threshold_inscrits = sum(1 for val in series if isinstance(val, (int, float)) and val >= threshold)
    count_above_threshold_presents = sum(1 for val in series.dropna() if isinstance(val, (int, float)) and val >= threshold)

    # Percentage calculations
    percentage_above_inscrits = (count_above_threshold_inscrits / stats_inscrits["count"]) * 100 if stats_inscrits["count"] > 0 else 0
    percentage_above_presents = (count_above_threshold_presents / stats_presents["count"]) * 100 if stats_presents["count"] > 0 else 0

    # Count occurrences of min and max values
    # Calculate correct min/max values
    min_value_inscrits = 0 if series.isna().sum() > 0 else stats_inscrits["min"]
    min_value_presents = stats_presents["min"]  # True minimum from actual scores
    max_value = stats_inscrits["max"]
    
    # Correct min/max counts
    min_count_inscrits = (series.fillna(0) == min_value_inscrits).sum()  # Includes absents as 0
    min_count_presents = (series.dropna() == min_value_presents).sum()  # Only counts actual 0s in presents
    max_count = (series == max_value).sum()
    
    # Format min/max to show occurrences
    min_display_inscrits = f"{min_value_inscrits:.2f} (*{min_count_inscrits})"
    min_display_presents = f"{min_value_presents:.2f} (*{min_count_presents})"
    max_display = f"{max_value:.2f} (*{max_count})"


    # Percentage of presents over inscrits
    present_percentage = (stats_presents["count"] / stats_inscrits["count"]) * 100 if stats_inscrits["count"] > 0 else 0
    present_count_display = f"{stats_presents['count']:.0f} ({present_percentage:.0f}%)"

    table_data = [
        ["", "Parmi les inscrits", "Parmi les présents"],
        ["Nombre de notes", f"{stats_inscrits['count']:.0f}", present_count_display],  # Add present % here
        ["Moyenne", f"{stats_inscrits['mean']:.2f}", f"{stats_presents['mean']:.2f}"],
        ["Écart type", f"{stats_inscrits['std']:.2f}", f"{stats_presents['std']:.2f}"],
        ["Min", min_display_inscrits, min_display_presents],  # Show min + count
        ["Q1 (25%)", f"{stats_inscrits['25%']:.2f}", f"{stats_presents['25%']:.2f}"],
        ["Médiane (50%)", f"{stats_inscrits['50%']:.2f}", f"{stats_presents['50%']:.2f}"],
        ["Q3 (75%)", f"{stats_inscrits['75%']:.2f}", f"{stats_presents['75%']:.2f}"],
        ["Max", max_display, max_display],  # Show max + count
        [f"Nombre >= {threshold}", f"{int(count_above_threshold_inscrits)}", f"{int(count_above_threshold_presents)}"],
        [f"% au-dessus de {threshold}", f"{percentage_above_inscrits:.2f}%", f"{percentage_above_presents:.2f}%"],
    ]

    col_width = 60
    row_height = 6
    for row in table_data:
        for item in row:
            pdf.cell(col_width, row_height, txt=item, border=1, align='C')
        pdf.ln(row_height)


def plot_distribution(series, column_name, pdf):
    """Generates a distribution plot for a given column and adds it to the PDF."""
    max_score = determine_max_score(series)
    plt.figure(figsize=(8, 5))
    
    ax = sns.histplot(series, bins=20, kde=True, color="lightblue", edgecolor="black", alpha=0.6)
    #ax = sns.histplot(series.dropna(), bins=bins, kde=True, color="lightblue", edgecolor="black", alpha=0.6)

    for line in ax.lines:
        line.set_color("darkorange")
        line.set_linestyle("-")
        line.set_linewidth(2)
    
    stats_inscrits = series.fillna(0).describe()  # Convert absents (NaN) to 0 for "inscrits"
    stats_presents = series.dropna().describe()  # Exclude absents but keep real zeros
    
    plt.axvline(stats_presents["25%"], color="blue", linestyle="--", label="Q1 (25%)")
    plt.axvline(stats_presents["50%"], color="red", linestyle="-", label="Médiane (50%)")
    plt.axvline(stats_presents["75%"], color="blue", linestyle="--", label="Q3 (75%)")
    plt.axvline(stats_presents["mean"], color="green", linestyle=":", label="Moyenne")
    
    plt.xlim(0, max_score + 1)
    plt.xlabel(column_name)
    plt.ylabel("Fréquence")
    plt.title(f"Statistiques sur {column_name} parmi les présents")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    safe_column_name = re.sub(r'[<>:"/\\|?*]', '_', column_name)  # Replace forbidden characters
    plot_filename = f"plot_{safe_column_name}.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    pdf.add_page()
    pdf.image(plot_filename, x=10, y=30, w=180)
    add_statistics_table(pdf, stats_inscrits, stats_presents, f"Statistiques sur {column_name}", series)
    os.remove(plot_filename)

def generate_pdf(file_path):
    delimiter = detect_delimiter(file_path)
    df = pd.read_csv(file_path, delimiter=delimiter)
    df = preprocess_data(df)
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    for column in df.columns:
        if df[column].dtype in [np.float64, np.int64] and is_grade_column(df[column]):
            plot_distribution(df[column], column, pdf)
    
    output_pdf = file_path.rsplit('.', 1)[0] + ".pdf"
    pdf.output(output_pdf)
    print(f"Generated {output_pdf}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python genstats.py <input_csv_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    generate_pdf(input_file)
