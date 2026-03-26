import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from preprocess import load_data, preprocess

# Set experiment
mlflow.set_experiment("house-price")

df = load_data("data/house.csv")
X, y = preprocess(df)

X_train, X_test, y_train, y_test = train_test_split(X, y)

with mlflow.start_run():

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # Logging
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)

    # Save model
    mlflow.sklearn.log_model(model, "model")

    print("Model trained & logged!")
