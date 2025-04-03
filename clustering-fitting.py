import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

# For clustering & fitting:
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression


def plot_relational_plot(df):
    """
    Plots a relational plot between 'TV Ad Budget ($)' (x) and 'Sales ($)' (y).
    Saves the figure as 'relational_plot.png'.
    """
    fig, ax = plt.subplots()
    sns.scatterplot(x=df["TV Ad Budget ($)"], y=df["Sales ($)"], ax=ax)
    plt.xlabel("TV Ad Budget ($)")
    plt.ylabel("Sales ($)")
    plt.title("TV Ad Budget vs Sales (Relational Plot)")
    plt.savefig('relational_plot.png')
    plt.close()


def plot_categorical_plot(df):
    """
    Plots a categorical plot (boxplot) showing how 'Sales ($)' vary
    based on 'HighTVAd' (0 or 1).
    Saves as 'categorical_plot.png'.
    'HighTVAd' is created in preprocessing as:
        1 if 'TV Ad Budget ($)' > 150 else 0
    """
    fig, ax = plt.subplots()
    sns.boxplot(x=df["HighTVAd"], y=df["Sales ($)"], ax=ax)
    plt.xlabel("HighTVAd (0=No, 1=Yes)")
    plt.ylabel("Sales ($)")
    plt.title("Sales by HighTVAd (Categorical Plot)")
    plt.savefig('categorical_plot.png')
    plt.close()


def plot_statistical_plot(df):
    """
    Plots a statistical plot (histogram + KDE) for 'TV Ad Budget ($)'.
    Saves the figure as 'statistical_plot.png'.
    """
    fig, ax = plt.subplots()
    sns.histplot(df["TV Ad Budget ($)"], kde=True, bins=20, ax=ax)
    plt.xlabel("TV Ad Budget ($)")
    plt.ylabel("Frequency")
    plt.title("Distribution of TV Ad Budget ($) (Statistical Plot)")
    plt.savefig('statistical_plot.png')
    plt.close()


def statistical_analysis(df, col: str):
    """
    Computes mean, standard deviation, skewness, and excess kurtosis
    for the chosen column 'col'.
    Returns (mean, std, skew, kurt).
    """
    mean_val = df[col].mean()
    std_val = df[col].std()
    skew_val = ss.skew(df[col])
    excess_kurt_val = ss.kurtosis(df[col])
    return mean_val, std_val, skew_val, excess_kurt_val


def preprocessing(df):
    """
    Preprocesses the 'Advertising Budget and Sales.csv' dataset by:
      1) Printing head, tail, describe
      2) Filling numeric missing data with median
      3) Dropping any remaining NaNs
      4) Creating 'HighTVAd' (0/1) if 'TV Ad Budget ($)' > 150
      5) Printing correlation matrix for numeric columns only
    Returns the cleaned df.
    """
    # Remove any extra whitespace from column names
    df.columns = df.columns.str.strip()

    print("Data Overview (head):")
    print(df.head())
    print("\nData Overview (tail):")
    print(df.tail())
    print("\nData Statistics:")
    print(df.describe())

    # Fill numeric missing values with median
    df.fillna(df.median(numeric_only=True), inplace=True)
    # Drop any remaining NaNs
    df.dropna(inplace=True)

    # Create 'HighTVAd': 1 if 'TV Ad Budget ($)' > 150 else 0
    df["HighTVAd"] = (df["TV Ad Budget ($)"] > 150).astype(int)

    # Print correlation matrix for numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    print("\nCorrelation Matrix (Numeric Columns Only):")
    print(numeric_df.corr())

    return df


def writing(moments, col):
    """
    Prints out the statistical analysis results for the given column.
    moments = (mean, std, skew, kurt).
    """
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    # Simple interpretation of skew & kurt
    skew_type = "not skewed"
    if moments[2] > 2:
        skew_type = "right skewed"
    elif moments[2] < -2:
        skew_type = "left skewed"

    kurt_type = "mesokurtic"
    if moments[3] > 2:
        kurt_type = "leptokurtic"
    elif moments[3] < -2:
        kurt_type = "platykurtic"

    print(f'The data was {skew_type} and {kurt_type}.')


def perform_clustering(df, col1, col2):
    """
    Performs K-Means clustering on two numeric columns (col1, col2).
    Demonstrates:
      - Elbow Plot (k=1..10)
      - Scaling
      - Silhouette Score
      - Inverse transform of centers
    Returns: labels, scaled_data, xkmeans, ykmeans, cluster_names
    """

    def plot_elbow_method(scaled_data):
        inertia_values = []
        K_range = range(1, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(scaled_data)
            inertia_values.append(km.inertia_)

        fig, ax = plt.subplots()
        plt.plot(K_range, inertia_values, marker="o")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.title("Elbow Plot")
        plt.savefig('elbow_plot.png')
        plt.close()

    def one_silhouette_inertia(scaled_data, n_clusters=3):
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labs = km.fit_predict(scaled_data)
        silhouette_val = silhouette_score(scaled_data, labs)
        centers_scaled = km.cluster_centers_
        return labs, centers_scaled, silhouette_val

    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[[col1, col2]])

    # Elbow method
    plot_elbow_method(data_scaled)

    # Default clusters = 3
    labels, centers_scaled, silhouette_val = one_silhouette_inertia(data_scaled)

    # Inverse transform cluster centers
    centers_unscaled = scaler.inverse_transform(centers_scaled)
    xkmeans, ykmeans = centers_unscaled[:, 0], centers_unscaled[:, 1]

    cluster_names = [f"Cluster {i + 1}" for i in range(3)]
    print(f"Clustering Silhouette Score: {silhouette_val:.2f}")

    return labels, data_scaled, xkmeans, ykmeans, cluster_names


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """
    Plots the clustered data (scaled domain) plus the unscaled centers if you prefer,
    or treat xkmeans, ykmeans as scaled. Saves figure as 'clustering.png'.
    """
    fig, ax = plt.subplots()
    plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.6, cmap='viridis')
    plt.scatter(xkmeans, ykmeans, color='red', marker='X', s=200, label="Centroids")
    plt.xlabel("Scaled TV Ad Budget ($)")
    plt.ylabel("Scaled Radio Ad Budget ($)")
    plt.title("K-Means Clustering (Scaled Data)")
    plt.legend()
    plt.savefig('clustering.png')
    plt.close()


def perform_fitting(df, col1, col2):
    """
    Fits a linear regression for col1 -> col2.
    Demonstrates scaling of col1 and inverse transform for x-range.
    Returns data, x_pred_unscaled, y_pred, original_feature_data
    """
    df_local = df.copy()

    # Save original feature data for optional plotting
    original_feature_data = df_local[col1].values.copy()

    scaler = StandardScaler()
    df_local[[col1]] = scaler.fit_transform(df_local[[col1]])

    X = df_local[[col1]].values
    y = df_local[col2].values

    model = LinearRegression()
    model.fit(X, y)

    x_min, x_max = X.min(), X.max()
    x_range_scaled = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_pred = model.predict(x_range_scaled)

    # Inverse transform for interpretability
    x_range_unscaled = scaler.inverse_transform(x_range_scaled)

    data = np.column_stack([X, y])
    return data, x_range_unscaled, y_pred, original_feature_data


def plot_fitted_data(data, x, y, original_feature_data, col_x, col_y):
    """
    Plots the linear regression results with unscaled x-range.
    'data' = scaled col1, col2.
    'x' = unscaled x-range for the fitted line.
    'y' = predicted y values.
    'original_feature_data' = original unscaled col1 data.
    Saves the figure as 'fitting.png'.
    """
    fig, ax = plt.subplots()
    # Plot original unscaled feature vs. actual target
    plt.scatter(original_feature_data, data[:, 1], alpha=0.6, label="Data (unscaled X)")
    # Plot fitted regression line
    plt.plot(x, y, color="red", label="Fitted Line (Unscaled X)")
    plt.xlabel(f"{col_x} (unscaled)")
    plt.ylabel(f"{col_y}")
    plt.title(f"Linear Regression: {col_x} vs {col_y}")
    plt.legend()
    plt.savefig('fitting.png')


def main():
    """
    Main function to orchestrate reading data,
    preprocessing, plotting, analysis, clustering, and fitting.
    """
    # 1) Load dataset
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    # 2) We choose 'TV Ad Budget ($)' for statistical analysis
    col = 'TV Ad Budget ($)'

    # 3) Generate Plots
    #   Relational: TV Ad Budget ($) vs Sales ($)
    #   Categorical: HighTVAd vs Sales ($)
    #   Statistical: Distribution of TV Ad Budget ($)
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    # 4) Stats Analysis
    moments = statistical_analysis(df, col)
    writing(moments, col)

    # 5) Clustering: 'TV Ad Budget ($)' & 'Radio Ad Budget ($)'
    cluster_res = perform_clustering(df, 'TV Ad Budget ($)', 'Radio Ad Budget ($)')
    plot_clustered_data(*cluster_res)

    # 6) Fitting: TV Ad Budget ($) -> Sales ($)
    fit_res = perform_fitting(df, 'TV Ad Budget ($)', 'Sales ($)')
    plot_fitted_data(*fit_res, col_x='TV Ad Budget ($)', col_y='Sales ($)')


if __name__ == '__main__':
    main()
