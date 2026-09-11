import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load Bee Colony and Honey Production dataset

file_path = "honey.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

df.head()
print("Dataset Information")
print("=" * 50)

df.info()
def shape_audit(df):
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1]
    }

print(shape_audit(df))
def missing_value_audit(df):
    missing = df.isnull().sum()
    
    result = pd.DataFrame({
        "Column": missing.index,
        "Missing_Values": missing.values,
        "Missing_Percentage": (missing.values / len(df)) * 100
    })
    
    return result[result["Missing_Values"] > 0].sort_values(
        "Missing_Values", ascending=False
    )

missing_value_audit(df)
def duplicate_audit(df):
    duplicates = df.duplicated().sum()
    
    print("Total duplicate records:", duplicates)
    
    if duplicates == 0:
        print("No duplicate records found.")
    else:
        print("Duplicate records found.")

duplicate_audit(df)
print("Columns in the dataset:")
for column in df.columns:
    print("-", column)
    numeric_columns = df.select_dtypes(include=np.number).columns

df[numeric_columns].describe()
# Change the column name if your dataset uses a different name

if "Colony_Strength" in df.columns:
    print("Colony Strength Analysis")
    print("=" * 50)
    
    print(df["Colony_Strength"].describe())
if "Honey_Production_kg" in df.columns:
    print("Honey Production Analysis")
    print("=" * 50)
    
    print("Total Honey Production:",
          df["Honey_Production_kg"].sum(), "kg")
    
    print("Average Honey Production:",
          df["Honey_Production_kg"].mean(), "kg")
    
    print("Maximum Honey Production:",
          df["Honey_Production_kg"].max(), "kg")
    
    print("Minimum Honey Production:",
          df["Honey_Production_kg"].min(), "kg")
    if "Season" in df.columns and "Honey_Production_kg" in df.columns:
    
        seasonal_production = df.groupby("Season")[
            "Honey_Production_kg"
        ].agg(
            Total_Honey="sum",
            Average_Honey="mean",
            Maximum_Honey="max"
        )
    
    print(seasonal_production)
    if "Bee_Species" in df.columns and "Honey_Production_kg" in df.columns:
    
        species_production = df.groupby("Bee_Species")[
            "Honey_Production_kg"
        ].agg(
            Total_Honey="sum",
            Average_Honey="mean"
        ).sort_values(
            "Average_Honey", ascending=False
        )
    
    print(species_production)
    if "Location" in df.columns and "Honey_Production_kg" in df.columns:
    
        location_production = df.groupby("Location")[
            "Honey_Production_kg"
        ].agg(
            Total_Honey="sum",
            Average_Honey="mean"
        ).sort_values(
            "Average_Honey", ascending=False
        )
    
    print(location_production)
    if ("Colony_Strength" in df.columns and
    "Honey_Production_kg" in df.columns):
    
        correlation = df[
            ["Colony_Strength", "Honey_Production_kg"]
        ].corr()
    
    print("Correlation between Colony Strength and Honey Production:")
    print(correlation)
    if ("Temperature_C" in df.columns and
    "Honey_Production_kg" in df.columns):
    
        correlation = df[
            ["Temperature_C", "Honey_Production_kg"]
        ].corr()
    
    print("Temperature vs Honey Production:")
    print(correlation)
    if ("Rainfall_mm" in df.columns and
    "Honey_Production_kg" in df.columns):
    
        correlation = df[
            ["Rainfall_mm", "Honey_Production_kg"]
        ].corr()
    
    print("Rainfall vs Honey Production:")
    print(correlation)
    if ("Feeding" in df.columns and
    "Honey_Production_kg" in df.columns):
    
        feeding_analysis = df.groupby("Feeding")[
            "Honey_Production_kg"
        ].agg(
            Average_Honey="mean",
            Total_Honey="sum",
            Records="count"
        )
    
    print(feeding_analysis)
    if ("Disease_Status" in df.columns and
    "Honey_Production_kg" in df.columns):
    
        disease_analysis = df.groupby("Disease_Status")[
            "Honey_Production_kg"
        ].agg(
            Average_Honey="mean",
            Total_Honey="sum",
            Records="count"
        )
    
    print(disease_analysis)
    numeric_data = df.select_dtypes(include=np.number)

    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Bee Colony and Honey Production Correlation")
    plt.show()
if "Honey_Production_kg" in df.columns:
    
    plt.figure(figsize=(8, 5))
    
    plt.hist(
        df["Honey_Production_kg"].dropna(),
        bins=20
    )
    
    plt.title("Distribution of Honey Production")
    plt.xlabel("Honey Production (kg)")
    plt.ylabel("Number of Colonies")
    
    plt.show()
    if ("Colony_Strength" in df.columns and
    "Honey_Production_kg" in df.columns):
    
        plt.figure(figsize=(8, 5))
        
        plt.scatter(
            df["Colony_Strength"],
            df["Honey_Production_kg"]
        )
        
        plt.title("Colony Strength vs Honey Production")
        plt.xlabel("Colony Strength")
        plt.ylabel("Honey Production (kg)")
        
        plt.show()
    if ("Season" in df.columns and
    "Honey_Production_kg" in df.columns):
    
        seasonal = df.groupby("Season")[
            "Honey_Production_kg"
        ].mean()
        
        plt.figure(figsize=(8, 5))
        
        seasonal.plot(kind="bar")
        
        plt.title("Average Honey Production by Season")
        plt.xlabel("Season")
        plt.ylabel("Average Honey Production (kg)")
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    if ("Bee_Species" in df.columns and
    "Honey_Production_kg" in df.columns):
    
        species = df.groupby("Bee_Species")[
            "Honey_Production_kg"
        ].mean().sort_values(ascending=False)
        
        plt.figure(figsize=(8, 5))
        
        species.plot(kind="bar")
        
        plt.title("Average Honey Production by Bee Species")
        plt.xlabel("Bee Species")
        plt.ylabel("Average Honey Production (kg)")
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    if "Honey_Production_kg" in df.columns:
    
        Q1 = df["Honey_Production_kg"].quantile(0.25)
        Q3 = df["Honey_Production_kg"].quantile(0.75)
        
        IQR = Q3 - Q1
        
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        
        outliers = df[
            (df["Honey_Production_kg"] < lower_limit) |
            (df["Honey_Production_kg"] > upper_limit)
        ]
        
        print("Number of honey-production outliers:",
            len(outliers))
        
        outliers.head()
    def data_sanity_check(df):
    
        results = {}
        
        if "Honey_Production_kg" in df.columns:
            results["Negative honey production"] = int(
                (df["Honey_Production_kg"] < 0).sum()
            )
            
            results["Zero honey production"] = int(
                (df["Honey_Production_kg"] == 0).sum()
            )
        
        if "Colony_Strength" in df.columns:
            results["Negative colony strength"] = int(
                (df["Colony_Strength"] < 0).sum()
            )
        
        if "Rainfall_mm" in df.columns:
            results["Negative rainfall"] = int(
                (df["Rainfall_mm"] < 0).sum()
            )
        
        return results


    data_sanity_check(df)
def run_all_audits(df):
    
    print("=" * 60)
    print("BEE COLONY AND HONEY PRODUCTION - DATA AUDIT")
    print("=" * 60)
    
    print("\n1. DATASET SHAPE")
    print(df.shape)
    
    print("\n2. DUPLICATE RECORDS")
    print(df.duplicated().sum())
    
    print("\n3. MISSING VALUES")
    print(df.isnull().sum())
    
    print("\n4. NUMERIC SUMMARY")
    print(df.describe())
    
    # print("\n5. DATA SANITY CHECK")
    # print(data_sanity_check(df))
    
    if "Honey_Production_kg" in df.columns:
        print("\n6. HONEY PRODUCTION")
        print("Total:",
              df["Honey_Production_kg"].sum(), "kg")
        print("Average:",
              df["Honey_Production_kg"].mean(), "kg")
        print("Maximum:",
              df["Honey_Production_kg"].max(), "kg")
        print("Minimum:",
              df["Honey_Production_kg"].min(), "kg")
    
    if "Colony_Strength" in df.columns:
        print("\n7. COLONY STRENGTH")
        print(df["Colony_Strength"].describe())
    
    print("\nAudit completed successfully.")


run_all_audits(df)