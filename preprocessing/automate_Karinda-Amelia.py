import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def preprocess_data(file_path, save_path=None):
    """
    Preprocess the dataset with standard cleaning steps:
    - Drop completely empty columns
    - Handling missing values
    - Remove duplicates
    - Handling outlier
    - Feature engineering
    - Encoding categorical variables
    - Standardization

    Args:
        file_path (str): Path to the raw CSV file.
        save_path (str, optional): Path to save preprocessed CSV. Default is None.

    Returns:
        pd.DataFrame: Cleaned and ready-to-train data.
    """

    # Load the dataset
    data = pd.read_csv(file_path, sep=";")

    # 1. Drop completely empty columns ('Unnamed: 15', 'Unnamed: 16')
    data = data.dropna(axis=1, how="all")

    # 2. Handling missing values
    data = data.dropna()

    # 3. Remove duplicates
    data = data.drop_duplicates()

    # 4. Handling outliers using the IQR (Interquartile Range) method
    numeric_columns = data.select_dtypes(include=["number"]).columns
    lower_bounds, upper_bounds = {}, {}

    for col in numeric_columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bounds[col] = Q1 - 1.5 * IQR
        upper_bounds[col] = Q3 + 1.5 * IQR

    data[numeric_columns] = data[numeric_columns].clip(
        lower=pd.Series(lower_bounds),
        upper=pd.Series(upper_bounds),
        axis=1
    )

    # 5. Feature Engineering - Time-based features from "Date"
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
        data["year"] = data["Date"].dt.year
        data["month"] = data["Date"].dt.month
        data["day"] = data["Date"].dt.day
        data["dayofweek"] = data["Date"].dt.dayofweek
        data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)
        data = data.drop(columns=["Date"])

    # 6. Encoding categorical variables
    le = LabelEncoder()
    categorical_cols = data.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    # 7. Standardization
    numeric_columns = data.select_dtypes(include="number").columns
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Save the preprocessed data to CSV
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if save_path:
        data.to_csv(save_path, index=False)
        print(f"File successfully saved to: {os.path.abspath(save_path)}")

    return data

def main():
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define multiple possible paths to locate the raw dataset
    possible_paths = [
        os.path.join(base_dir, "air_quality_raw.csv"),
        os.path.join(base_dir, "../air_quality_raw.csv"),
        os.path.join(base_dir, "../../air_quality_raw.csv")
    ]
    
    # Select the first valid path that exists
    raw_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    # Display an error message if the dataset is not found
    if raw_path is None:
        raise FileNotFoundError("Dataset not found in expected locations.")
    
    # Set the path where the preprocessed data will be saved
    save_path = os.path.join(base_dir, "air_quality_preprocessing.csv")

    print("Starting preprocessing...")
    
    # Call the preprocessing function
    data = preprocess_data(raw_path, save_path)
    
    # Display a sample of the preprocessed data
    print("Preprocessing complete. Sample of preprocessed data:")
    print(data.head())

# Run main function when this script is executed directly
if __name__ == "__main__":
    main()
