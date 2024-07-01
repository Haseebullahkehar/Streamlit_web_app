import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.experimental import enable_iterative_imputer  # Enable the experimental feature
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pickle

# Function to identify if the target column is regression or classification
def identify_problem(target_column):
    if pd.api.types.is_numeric_dtype(target_column):
        return "regression"
    else:
        return "classification"

# Function to preprocess the data
# Function to preprocess the data
def preprocess_data(df, features, target, problem_type):
    X = df[features]
    y = df[target]

    # Handle missing values
    imputer = IterativeImputer()
    X = pd.DataFrame(imputer.fit_transform(X), columns=features)

    # Encode categorical variables (highlighted addition)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Scale features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=features)

    # Encode categorical variables if classification problem (remove this redundant part)
    if problem_type == "classification":
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    return X, y


# Function to evaluate models
def evaluate_model(model, X_test, y_test, problem_type):
    y_pred = model.predict(X_test)
    if problem_type == "regression":
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        try:
            auroc = roc_auc_score(y_test, y_pred)
        except ValueError:
            auroc = "Not applicable"
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "AUROC": auroc}
    else:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1, "Confusion Matrix": cm}

# Main function
def main():
    st.title("Machine Learning Application")

    st.sidebar.header("Upload Data or Use Example Dataset")
    data_choice = st.sidebar.selectbox("Choose data source", ["Upload Data", "Use Example Dataset"])

    if data_choice == "Upload Data":
        uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx", "tsv"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith("csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith("xlsx"):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith("tsv"):
                df = pd.read_csv(uploaded_file, sep='\t')
            else:
                st.error("Unsupported file format")
                return
        else:
            st.warning("Please upload a dataset.")
            return
    else:
        example_dataset = st.sidebar.selectbox("Select Example Dataset", ["titanic", "tips", "iris"])
        df = sns.load_dataset(example_dataset)

    st.write("### Dataset Overview")
    st.write("**Data Head:**")
    st.write(df.head())
    st.write("**Data Shape:**", df.shape)
    st.write("**Data Description:**")
    st.write(df.describe())
    st.write("**Data Info:**")
    st.write(df.info())
    st.write("**Column Names:**", df.columns)

    st.sidebar.header("Select Features and Target")
    features = st.sidebar.multiselect("Select Features", df.columns)
    target = st.sidebar.selectbox("Select Target", df.columns)

    if not features or not target:
        st.warning("Please select features and target.")
        return

    problem_type = identify_problem(df[target])
    st.write(f"### Identified Problem Type: {problem_type.capitalize()}")

    X, y = preprocess_data(df, features, target, problem_type)
    st.sidebar.header("Train/Test Split")
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    st.sidebar.header("Select Model")
    if problem_type == "regression":
        model_choice = st.sidebar.selectbox("Model", ["Linear Regression", "Decision Tree", "Random Forest", "Support Vector Machine"])
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Decision Tree":
            model = DecisionTreeRegressor()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor()
        elif model_choice == "Support Vector Machine":
            model = SVR()
    else:
        model_choice = st.sidebar.selectbox("Model", ["Decision Tree", "Random Forest", "Support Vector Machine"])
        if model_choice == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "Support Vector Machine":
            model = SVC()

    best_model_name = None
    best_metric = float('inf') if problem_type == 'regression' else 0

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor() if problem_type == 'regression' else DecisionTreeClassifier(),
        "Random Forest": RandomForestRegressor() if problem_type == 'regression' else RandomForestClassifier(),
        "Support Vector Machine": SVR() if problem_type == 'regression' else SVC()
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, problem_type)
        st.write(f"### {model_name} Evaluation Metrics")
        for metric, value in metrics.items():
            st.write(f"{metric}: {value}")

        if problem_type == 'regression':
            if metrics["RMSE"] < best_metric:
                best_metric = metrics["RMSE"]
                best_model_name = model_name
        else:
            if metrics["F1 Score"] > best_metric:
                best_metric = metrics["F1 Score"]
                best_model_name = model_name

    st.write(f"### Best Model: {best_model_name}")

    st.sidebar.header("Download Model")
    if st.sidebar.button("Download Model"):
        with open("best_model.pkl", "wb") as f:
            pickle.dump(model, f)
        st.success("Model downloaded as best_model.pkl")

    st.sidebar.header("Make Predictions")
    prediction_choice = st.sidebar.selectbox("Choose Prediction Input Method", ["Manual Input", "Upload File"])
    if prediction_choice == "Manual Input":
        input_data = {}
        for feature in features:
            input_data[feature] = st.sidebar.number_input(f"Input {feature}")
        input_df = pd.DataFrame([input_data])
        if st.sidebar.button("Predict"):
            prediction = model.predict(input_df)
            st.write(f"Prediction: {prediction}")
    else:
        pred_file = st.sidebar.file_uploader("Upload Prediction Data", type=["csv", "xlsx", "tsv"])
        if pred_file is not None:
            if pred_file.name.endswith("csv"):
                pred_df = pd.read_csv(pred_file)
            elif pred_file.name.endswith("xlsx"):
                pred_df = pd.read_excel(pred_file)
            elif pred_file.name.endswith("tsv"):
                pred_df = pd.read_csv(pred_file, sep='\t')
            else:
                st.error("Unsupported file format")
                return
            if st.sidebar.button("Predict"):
                predictions = model.predict(pred_df)
                st.write("Predictions:", predictions)

if __name__ == "__main__":
    main()
