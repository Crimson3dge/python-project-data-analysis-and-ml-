# Global Energy Consumption Analysis
# Main Objective:
# Analyze energy consumption, sustainability, and carbon emissions using data science techniques.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA


# ── Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv(r"global_energy_consumption.csv")
df.head()

# ── Cleaning ─────────────────────────────────────────────────────────────────
print(df.isnull().sum())
df.columns = df.columns.str.strip()
df.describe()
df.dtypes


# ── Objective 1: Global Energy Consumption Trends ────────────────────────────
# Analyze global energy consumption trends over time by aggregating yearly data
# to identify long-term growth patterns, fluctuations, and overall demand changes.

# Average energy consumption by country
avg_energy = df.groupby("Country")["Total Energy Consumption (TWh)"].mean()
avg_energy = avg_energy.sort_values(ascending=False)

print("\nTop Energy Consuming Countries:\n")
print(avg_energy)

plt.bar(avg_energy.index, avg_energy.values)
plt.xticks(rotation=45)
plt.xlabel("Country")
plt.ylabel("Average Energy Consumption (TWh)")
plt.title("Average Energy Consumption by Country")
plt.tight_layout()
plt.show()

# Global Energy Consumption Trend Over Time
global_trend = df.groupby("Year")["Total Energy Consumption (TWh)"].sum()

z = np.polyfit(global_trend.index, global_trend.values, 1)
p = np.poly1d(z)

plt.plot(global_trend.index, global_trend.values, marker='o', label="Actual")
plt.plot(global_trend.index, p(global_trend.index), '--', label="Trend Line")
plt.xlabel("Year")
plt.ylabel("Total Energy Consumption (TWh)")
plt.title("Trend Analysis of Global Energy Consumption")
plt.legend()
plt.grid(True)
plt.show()

# Energy Price Trend Over Time
price_trend = df.groupby("Year")["Energy Price Index (USD/kWh)"].mean()

plt.plot(price_trend.index, price_trend.values)
plt.xlabel("Year")
plt.ylabel("Energy Price Index(USD/kWh)")
plt.title("Energy Price Trend Over Time")
plt.show()

# Correlation Heatmap
cols = [
    "Total Energy Consumption (TWh)",
    "Renewable Energy Share (%)",
    "Fossil Fuel Dependency (%)",
    "Carbon Emissions (Million Tons)",
    "Energy Price Index (USD/kWh)"
]

corr = df[cols].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Energy Features")
plt.show()


# ── Objective 2: Country-wise Energy Analysis ────────────────────────────────

# Country-wise energy consumption trend
grouped = df.groupby(["Country", "Year"])["Total Energy Consumption (TWh)"].mean().reset_index()

countries = list(grouped["Country"].unique())

mid = len(countries) // 2
group1 = countries[:mid]
group2 = countries[mid:]


def plot_group(group, title):
    plt.figure(figsize=(10, 6))
    for country in group:
        subset = grouped[grouped["Country"] == country]
        plt.plot(
            subset["Year"],
            subset["Total Energy Consumption (TWh)"],
            label=country
        )
    plt.xlabel("Year")
    plt.ylabel("Energy Consumption (TWh)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_group(group1, "Energy Consumption Trend (Group 1)")
plot_group(group2, "Energy Consumption Trend (Group 2)")

# Energy Consumption Distribution: Industrial vs Household
industrial = df["Industrial Energy Use (%)"].mean()
household = df["Household Energy Use (%)"].mean()

labels = ["Industrial", "Household"]
values = [industrial, household]

plt.figure(figsize=(6, 6))
plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title("Energy Consumption Distribution: Industrial vs Household")
plt.show()


# ── Objective 3: Country Segmentation ───────────────────────────────────────
# Cluster countries based on energy behavior.

country_df = df.groupby("Country").mean()

features = [
    "Total Energy Consumption (TWh)",
    "Renewable Energy Share (%)",
    "Fossil Fuel Dependency (%)",
    "Carbon Emissions (Million Tons)"
]

X = country_df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
country_df["Cluster"] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

country_df["PCA1"] = X_pca[:, 0]
country_df["PCA2"] = X_pca[:, 1]

for cluster in range(3):
    subset = country_df[country_df["Cluster"] == cluster]
    plt.scatter(subset["PCA1"], subset["PCA2"], label=f"Cluster {cluster}")

for i in range(len(country_df)):
    plt.text(
        country_df["PCA1"].iloc[i],
        country_df["PCA2"].iloc[i],
        country_df.index[i],
        fontsize=9
    )

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Country Energy Clusters")
plt.legend()
plt.grid(True)
plt.show()

print("\nCountry Clusters:\n")
print(country_df["Cluster"])

# Insight:
# - Cluster 0 → Fossil-heavy countries
# - Cluster 1 → Renewable-focused
# - Cluster 2 → Transitional economies


# ── Objective 4: Carbon Emission Prediction ──────────────────────────────────
# Build and evaluate ML models to predict carbon emissions using energy features.

# Feature Engineering
df["Carbon Intensity"] = df["Carbon Emissions (Million Tons)"] / df["Total Energy Consumption (TWh)"]
df["Renewability Score"] = df["Renewable Energy Share (%)"] - df["Fossil Fuel Dependency (%)"]

features = [
    "Total Energy Consumption (TWh)",
    "Per Capita Energy Use (kWh)",
    "Renewable Energy Share (%)",
    "Fossil Fuel Dependency (%)",
    "Industrial Energy Use (%)",
    "Household Energy Use (%)",
    "Energy Price Index (USD/kWh)",
    "Carbon Intensity",
    "Renewability Score"
]

target = "Carbon Emissions (Million Tons)"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Evaluation
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("\nModel Performance:\n")
print("Linear Regression")
print("R2  :", round(lr_r2, 4))
print("RMSE:", round(lr_rmse, 2))
print("\nRandom Forest")
print("R2  :", round(rf_r2, 4))
print("RMSE:", round(rf_rmse, 2))

# Model Comparison Graph
models = ["Linear Regression", "Random Forest"]
r2_scores = [lr_r2, rf_r2]

plt.figure(figsize=(6, 4))
bars = plt.bar(models, r2_scores, color=["#4fc3f7", "#81c784"], edgecolor="black")
plt.ylabel("R2 Score")
plt.title("Model Comparison — R2 Score")
plt.ylim(0, 1.1)
plt.grid(axis="y", linestyle="--", alpha=0.6)
for bar, val in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
             f"{val:.4f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=130)
plt.show()

# Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, rf_pred, alpha=0.3, color="#4fc3f7", label="Predictions")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red", linewidth=1.5, label="Perfect fit"
)
plt.xlabel("Actual Emissions")
plt.ylabel("Predicted Emissions")
plt.title("Random Forest: Actual vs Predicted")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=130)
plt.show()

# Global Avg Renewable Share vs Carbon Emissions
renew_avg = df.groupby("Year")["Renewable Energy Share (%)"].mean()
carbon_avg = df.groupby("Year")["Carbon Emissions (Million Tons)"].mean()

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(renew_avg.index, renew_avg.values, marker='o', label="Renewable Energy (%)")
ax1.set_xlabel("Year")
ax1.set_ylabel("Renewable Energy (%)")

ax2 = ax1.twinx()
ax2.plot(carbon_avg.index, carbon_avg.values, marker='s', linestyle='--', label="Carbon Emissions (MT)")
ax2.set_ylabel("Carbon Emissions (MT)")

plt.title("Global Avg Renewable Share vs Carbon Emissions")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)
ax1.grid(True)

plt.show()


# ── Objective 5: Renewable Energy Forecast ───────────────────────────────────
# Forecast renewable adoption (2025–2029)

# Aggregate: mean Renewable Energy Share per Country per Year
df_agg = (
    df.groupby(["Country", "Year"])["Renewable Energy Share (%)"]
    .mean()
    .reset_index()
)

countries = sorted(df_agg["Country"].unique().tolist())
future_years = list(range(2025, 2030))

# Build Forecast Results
forecast_results = {}

for country in countries:
    subset = df_agg[df_agg["Country"] == country].sort_values("Year")

    years = subset["Year"].values
    actual = subset["Renewable Energy Share (%)"].values

    X = years.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, actual)

    slope = model.coef_[0]
    future_X = np.array(future_years).reshape(-1, 1)
    future = model.predict(future_X)
    future = np.clip(future, 0, 100)

    forecast_results[country] = {
        "years": years.tolist(),
        "actual": actual.tolist(),
        "future": future.tolist(),
        "slope": slope,
    }

# Split into Two Groups
mid = len(countries) // 2
group1 = countries[:mid]
group2 = countries[mid:]


def plot_group(group, title):
    plt.figure(figsize=(12, 6))
    for country in group:
        data = forecast_results[country]
        years = data["years"]
        actual = data["actual"]
        future = data["future"]

        line, = plt.plot(years, actual, marker='o', markersize=3, linewidth=1.5,
                         label=f"{country} (Actual)")
        color = line.get_color()

        plt.plot([years[-1], future_years[0]], [actual[-1], future[0]],
                 linestyle='--', color=color, linewidth=1.2)

        plt.plot(future_years, future, linestyle='--', color=color, linewidth=1.8,
                 label=f"{country} (Forecast)")

    plt.axvline(x=2024.5, linestyle=':', color='gray', linewidth=1.2,
                label='Forecast Start (2025)')
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Renewable Energy Share (%)", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=8, loc='upper left')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


plot_group(group1, "Renewable Forecast (Group 1)")
plot_group(group2, "Renewable Forecast (Group 2)")

# Forecast Summary (2029)
print("\n" + "=" * 55)
print("  Forecast Summary — Renewable Energy Share (2029)")
print("=" * 55)
print(f"  {'Country':<15} {'2029 Forecast':>14}  {'Trend'}")
print("-" * 55)

for country in countries:
    future_val = forecast_results[country]["future"][-1]
    slope = forecast_results[country]["slope"]
    trend = "Increasing" if slope > 0 else "Decreasing"
    print(f"  {country:<15} {future_val:>12.2f}%  {trend}")

print("=" * 55)

# ── Conclusion ───────────────────────────────────────────────────────────────
# - Global energy consumption is steadily increasing, driven by economic growth and industrialization.
# - Renewable energy adoption shows a moderate but limited impact on reducing carbon emissions.
# - Clustering reveals distinct energy profiles among countries, highlighting sustainability differences.
# - Random Forest outperforms Linear Regression, indicating non-linear relationships.
# - Forecasting suggests a slow but positive transition toward renewable energy.
