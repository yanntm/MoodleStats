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

def detect_id_column(df):
    """
    Detects a column likely to be a student ID by looking for a column name
    that matches the regex /.*ident.*/i.
    Returns the column name if found, else None.
    """
    for col in df.columns:
        if re.search(r'.*ident.*', col, re.IGNORECASE):
            print(f"Selected column '{col}' as ID column based on regex match.")
            return col
    return None

def alphanum_key(s):
    """
    Splits the string into (prefix, number, remainder) where:
      - prefix is the non-digit part before the first number (in lower case)
      - number is the first number as an integer (if present)
      - remainder is the rest of the string (in lower case)
    If no number is found, returns a one-element tuple.
    """
    match = re.match(r'([^0-9]*)([0-9]+)(.*)', s)
    if match:
        prefix, num, remainder = match.groups()
        return (prefix.lower(), int(num), remainder.lower())
    else:
        return (s.lower(),)


def process_id_column(df):
    """
    Detects the ID column using detect_id_column, renames it to "ID",
    filters out rows where the ID is NA, empty, or equals "nan" (case-insensitive),
    normalizes the IDs (removes trailing .0 for whole numbers), prints a debug trace,
    and returns the modified DataFrame.
    """
    df = df.copy()
    id_col = detect_id_column(df)
    if id_col:
        print(f"Detected ID column '{id_col}', renaming to 'ID' and cleaning data.")
        before = len(df)
        df.rename(columns={id_col: "ID"}, inplace=True)
        # Filter out rows where ID is NA or blank or equals "nan"
        df = df[df["ID"].notna()]
        df = df[df["ID"].astype(str).str.strip().str.lower() != "nan"]
        df = df[df["ID"].astype(str).str.strip() != ""]
        after = len(df)
        print(f"Removed {before - after} rows with empty/invalid ID (from {before} to {after}).")
        # Normalize: if the ID is numeric with a trailing .0, convert to int then string.
        df["ID"] = df["ID"].apply(
            lambda x: str(int(float(x))) if re.match(r"^-?\d+\.0$", str(x).strip()) else str(x).strip()
        )
    else:
        print("No ID column detected.")
    return df

def preprocess_data(df):
    df = df.copy()
    # Remove rows that contain the string "root@localhost" in any cell.
    df = df[~df.apply(lambda row: row.astype(str).str.contains("root@localhost", case=False, na=False).any(), axis=1)]

    """Preprocesses the data: converts numeric columns, replaces missing values, and filters out IDs."""
    for col in df.columns:
        if col == "ID":
          continue  # Skip the ID column.
        df[col] = df[col].astype(str).str.lstrip("'")  # Remove leading quotes from Moodle export
        df[col] = df[col].replace("-", np.nan)  # Mark "-" as NaN (absent students)
        df[col] = df[col].str.replace('%', '', regex=True).str.strip()  # Remove percentage signs if present
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric where possible

    # Drop columns that are completely empty after removing non-numeric values
    df = df.dropna(axis=1, how='all')
    
    return df

def detect_group_column(df):
    """
    Detects the column representing group information by matching its name against a regex.
    Returns the column name if found, otherwise None.
    """
    for col in df.columns:
        if re.search(r'.*group.*', col, re.IGNORECASE):
            return col
    return None


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
        [f"Nombre >= {threshold}", f"{int(count_above_threshold_inscrits)} ({percentage_above_inscrits:.0f}%)", f"{int(count_above_threshold_presents)} ({percentage_above_presents:.0f}%)"],
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


def compute_stats(series, fillna=False):
    """Compute descriptive statistics.
    If fillna is True, replace NaN with 0 (for inscrits); otherwise, only use present values.
    """
    s = series.fillna(0) if fillna else series.dropna()
    stats = s.describe().to_dict()
    max_score = 100 if stats['max'] > 20 else 20
    threshold = 50 if max_score == 100 else 10
    count_above = sum(1 for val in s if isinstance(val, (int, float)) and val >= threshold)
    percentage_above = (count_above / stats["count"]) * 100 if stats["count"] > 0 else 0
    # Determine min and max occurrences
    if fillna:
        min_value = 0 if series.isna().sum() > 0 else stats['min']
        min_count = (series.fillna(0) == min_value).sum()
    else:
        min_value = stats['min']
        min_count = (s == min_value).sum()
    max_count = (s == stats['max']).sum()
    
    stats["threshold"] = threshold
    stats["count_above"] = count_above
    stats["percentage_above"] = percentage_above
    stats["min_count"] = min_count
    stats["max_count"] = max_count
    return stats


def plot_small_distribution(ax, series, max_score, threshold, title):
    """Plot a small histogram on a given axis."""
    s = series.dropna()
    sns.histplot(s, bins=20, kde=True, color="lightblue",
                 edgecolor="black", alpha=0.6, ax=ax)
    stats = s.describe()
    ax.axvline(stats["25%"], color="blue", linestyle="--")
    ax.axvline(stats["50%"], color="red", linestyle="-")
    ax.axvline(stats["75%"], color="blue", linestyle="--")
    ax.axvline(stats["mean"], color="green", linestyle=":")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, max_score + 1)
    ax.grid(True, linestyle="--", alpha=0.6)

def plot_group_details(merged_df, column_name, pdf, group_sizes, total_inscrits):
    """
    Plots overall and per-group statistics for the given grade column.
    Assumes merged_df contains a 'group' column.
    """
    # Filter rows with a valid group (but don’t filter out NA grades yet)
    df_valid = merged_df[merged_df["group"].notna()]
    
    # For computing overall stats, consider only rows with a grade.
    overall = df_valid[df_valid[column_name].notna()][column_name]
    overall_stats = compute_stats(overall, fillna=False)
    overall_stats["total"] = total_inscrits  # Use the precomputed overall total.
    
    max_score = 100 if overall_stats['max'] > 20 else 20
    threshold = overall_stats['threshold']
    
    # Get group names sorted with natural/alphanum order.
    group_names = df_valid["group"].unique()
    group_names = sorted(group_names, key=alphanum_key)
    print(f"Found {len(group_names)} groups: {', '.join(group_names)}")
    
    # Process groups in chunks.
    chunk_size = 3  # Adjust as needed.
    for i in range(0, len(group_names), chunk_size):
        chunk = group_names[i:i+chunk_size]
        total_plots = 1 + len(chunk)  # One overall plot + one per group.
        ncols = 2
        nrows = (total_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        axes = axes.flatten()
        
        # Plot overall distribution.
        plot_small_distribution(axes[0], overall, max_score, threshold, "Tous les présents")
        stats_dict = {"Tous les présents": overall_stats}
        
        # Plot each group's distribution.
        for idx, grp in enumerate(chunk, start=1):
            grp_series = df_valid[(df_valid["group"] == grp) &
                                  (df_valid[column_name].notna())][column_name]
            if grp_series.empty:
                axes[idx].axis('off')
                continue
            plot_small_distribution(axes[idx], grp_series, max_score, threshold, grp)
            grp_stats = compute_stats(grp_series, fillna=False)
            # Use the precomputed group size.
            grp_stats["total"] = group_sizes.get(grp, 0)
            stats_dict[grp] = grp_stats
        
        # Hide unused axes.
        for j in range(total_plots, len(axes)):
            axes[j].axis('off')
        
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', column_name)
        temp_fig_file = f"temp_{safe_name}_{i}.png"
        plt.tight_layout()
        plt.savefig(temp_fig_file, bbox_inches='tight')
        plt.close(fig)
        pdf.add_page()
        pdf.image(temp_fig_file, x=10, y=30, w=180)
        os.remove(temp_fig_file)
        
        add_detailed_statistics_table(pdf, stats_dict, column_name)

def add_detailed_statistics_table(pdf, stats_dict, column_name):
    """Add a table comparing overall and per-group statistics."""
    pdf.set_y(220)  # Adjust vertical position as needed.
    pdf.set_font("Arial", size=12, style='B')
    header = ["Statistique"] + list(stats_dict.keys())
    col_width = 35
    row_height = 6
    
    # Header row.
    for item in header:
        pdf.cell(col_width, row_height, txt=item, border=1, align='C')
    pdf.ln(row_height)
    
    # Define the rows: each tuple is (stat key, label)
    stat_keys = [
        ("count", "Nombre de notes"),
        ("mean", "Moyenne"),
        ("std", "Écart type"),
        ("min", "Min"),
        ("25%", "Q1 (25%)"),
        ("50%", "Médiane (50%)"),
        ("75%", "Q3 (75%)"),
        ("max", "Max"),
        ("count_above", f"Nombre >= {stats_dict['Tous les présents']['threshold']}")
    ]
    
    # Data rows.
    for key, label in stat_keys:
        pdf.set_font("Arial", size=10)
        pdf.cell(col_width, row_height, txt=label, border=1, align='C')
        for stat in stats_dict.values():
            if key == "count":
                # stat["count"] is the number of present grades.
                total = stat.get("total", stat["count"])
                count_present = stat["count"]
                percent = (count_present / total * 100) if total > 0 else 0
                value = f"{int(count_present)}/{int(total)} ({int(round(percent))}%)"
            elif key == "min":
                value = f"{stat['min']:.2f} (*{stat['min_count']})"
            elif key == "max":
                value = f"{stat['max']:.2f} (*{stat['max_count']})"
            elif key == "count_above":
                value = f"{int(stat['count_above'])} ({int(round(stat['percentage_above']))}%)"
            else:
                value = f"{stat[key]:.2f}"
            pdf.cell(col_width, row_height, txt=value, border=1, align='C')
        pdf.ln(row_height)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python genstats.py <input_csv_file> [group_csv_file]")
        sys.exit(1)
    input_file = sys.argv[1]
    group_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    delimiter = detect_delimiter(input_file)
    df = pd.read_csv(input_file, delimiter=delimiter)
    df = process_id_column(df)
    df = preprocess_data(df)
    
    if "ID" in df.columns:
      df.set_index("ID", inplace=True)
      print("Main CSV: processed ID column and set index to 'ID'.")

    if group_file is not None:
      df_groups = pd.read_csv(group_file, delimiter=detect_delimiter(group_file))
      
      # Process the group CSV ID column
      df_groups = process_id_column(df_groups)
      
      # Detect and rename the group column using regex.
      group_col = detect_group_column(df_groups)
      if group_col:
          df_groups.rename(columns={group_col: "group"}, inplace=True)
          print(f"Group CSV: detected group column '{group_col}', renamed to 'group'.")
      else:
          print("Group CSV: no group column detected.")
      
      # Filter out rows with empty or missing group values.
      if "group" in df_groups.columns:
          before_filter = len(df_groups)
          df_groups = df_groups[df_groups["group"].notna() &
                                (df_groups["group"].astype(str).str.strip() != "")]
          print(f"Group CSV: filtered out empty group rows, from {before_filter} to {len(df_groups)} records.")
      
      else:
          df_groups = None

    if df_groups is not None:
        # Reset index (since main CSV has 'ID' as index) to merge on column 'ID'
        df.reset_index(inplace=True)
        # Merge group information; use a left join so all main CSV rows are kept.
        df = df.merge(df_groups[['ID', 'group']], on="ID", how="left")
        print(f"Merged main CSV and group CSV: {df.shape[0]} records.")
        
        # Now compute group sizes only if 'group' exists.
        if "group" in df.columns:
            group_sizes = df[df["group"].notna()].groupby("group").size().to_dict()
            total_inscrits = df[df["group"].notna()].shape[0]
            print("Computed group sizes:", group_sizes)
        else:
            group_sizes = {}
            total_inscrits = 0
    else:
        group_sizes = {}
        total_inscrits = 0

    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    for column in df.columns:
      if column in ["ID", "group"]:
          continue  # Skip non-grade columns.
      if df[column].dtype in [np.float64, np.int64] and is_grade_column(df[column]):
          # Generate the overall distribution plot.
          plot_distribution(df[column], column, pdf)
          # If group info is available, plot group details using the merged DataFrame.
          if "group" in df.columns:
              plot_group_details(df, column, pdf, group_sizes, total_inscrits)

    
    output_pdf = input_file.rsplit('.', 1)[0] + ".pdf"
    pdf.output(output_pdf)
    print(f"Generated {output_pdf}")
