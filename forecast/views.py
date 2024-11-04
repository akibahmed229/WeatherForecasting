# Section-1: Import Libraries
from django.http.response import os
from django.shortcuts import render
from django.http import HttpResponse

import requests  # This library helps us to fetch data from API
import pandas as pd  # For handling and analysing data
import numpy as np  # For numerical operations
import pytz
from sklearn.model_selection import (
    train_test_split,
)  # To split data into training and testing sets
from sklearn.preprocessing import (
    LabelEncoder,
)  # TO convert catogerical data into numericals values
from sklearn.ensemble import RandomForestRegressor  # Models for regression tasks
from sklearn.linear_model import LinearRegression  # Model for linear regression
from sklearn.metrics import (
    mean_squared_error,
)  # To measure the  accuracy of out prediction
from datetime import datetime, timedelta  # To handle date & time

API_KEY = "7ec3ca4280bb09b11fab0b2153721c10"
BASE_URL = "https://api.openweathermap.org/data/2.5/"  # for making api request


# 1.Fetch Weather Data
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"  # construct the API request URL
    response = requests.get(url)  # send the get request to API
    data = response.json()  # parse the response JSON

    # Check for errors in the API response
    if response.status_code != 200 or "name" not in data:
        # Log the issue or return an error message to the user
        print(f"Error fetching weather data: {data.get('message', 'Unknown error')}")
        return None

    # If everything is fine, extract the relevant data
    return {
        "city": data["name"],
        "current_temp": round(data["main"]["temp"]),
        "feels_like": round(data["main"]["feels_like"]),
        "temp_min": round(data["main"]["temp_min"]),
        "temp_max": round(data["main"]["temp_max"]),
        "humidity": round(data["main"]["humidity"]),
        "description": data["weather"][0]["description"],
        "country": data["sys"]["country"],
        "wind_gust_dir": data["wind"]["deg"],
        "pressure": data["main"]["pressure"],
        "wind_gust_speed": data["wind"]["speed"],
        "clouds": data["clouds"]["all"],
        "visibility": data["visibility"],
    }


# 2.Read Historical data
def read_historical_data(filename):
    df = pd.read_csv(filename)  # load csv file into dataFrame
    df = df.dropna()  # remove rows with missing values
    df = df.drop_duplicates()
    return df


# 3.Prepare Data For Training
def prepare_data(data):
    le = LabelEncoder()  # initialize LabelEncoder instance

    # Convert categorical data to numerical
    data["WindGustDir"] = le.fit_transform(data["WindGustDir"])
    data["RainTomorrow"] = le.fit_transform(data["RainTomorrow"])

    # Define the feature variables and target variable
    X = data.iloc[:, :-1]  # feature variables
    y = data.iloc[:, -1]  # target variable

    return X, y, le


# 4.Train Rain Prediction Model
def train_rain_model(X, y):
    """
    This function train our model using classification to predict data

    * random_state: reproducibility making sure results are consistent
    * n_estimators: to use 100 decision tree
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = (model_lr.predict(X_test) > 0.5).astype(
        int
    )  # Convert to binary predictions

    print("Mean Square Error for Rain Model")
    print(mean_squared_error(y_test, y_pred_lr))

    return model_lr


# 5.Prepare regression data
def prepare_regression_data(dataFrame, feature):
    """
    To predict future values of a feature (like temperature & humidity), we need to use regression.
    Regression helps us predict continuous values, such as numerical measurements.
    """

    # Initialize lists to store feature (X) and target (y) values
    X, y = [], []

    # Iterate over the data frame rows, excluding the last one
    for i in range(len(dataFrame) - 1):
        # Append the current value of the specified feature to X
        X.append(dataFrame[feature].iloc[i])
        # Append the next value of the specified feature to y, creating a target value for each input
        y.append(dataFrame[feature].iloc[i + 1])

    # Convert X to a 2D numpy array and reshape it to be compatible with scikit-learn (each value is an array)
    X = np.array(X).reshape(-1, 1)
    # Convert y to a numpy array for consistency in data format
    y = np.array(y)

    # Return the prepared feature and target arrays
    return X, y


# 6.Train Regression Model
def train_regression_model(X, y):
    # Initialize a RandomForestRegressor model with 100 decision trees
    # and a fixed random seed for reproducibility.
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the regression model on the feature set X and target y
    model.fit(X, y)

    # Return the trained model for later use in making predictions
    return model


# 7.Predict Future Weather
def predict_future(model, current_value):
    """
    Predict future temperature values.

    model: The trained RandomForestRegressor model
    current_value: The latest temperature value we have
    """

    # Start predictions list with the most recent temperature
    predictions = [current_value]

    # Predict the next 5 values (for the next 5 hours in this example)
    for i in range(5):
        # Predict the next temperature based on the latest predicted value
        next_value = model.predict([[predictions[-1]]])  # Use last prediction as input

        # Add this predicted temperature to the list
        predictions.append(next_value[0])

    # Return only the future predictions (skip the first element)
    return predictions[1:]


def weather_view(request):
    # Set default city to "Dhaka" if it's a GET request
    if request.method == "POST":
        city = request.POST.get("city")
    else:
        city = "Dhaka"  # Default city if no POST request

    # Fetch the current weather for the city
    current_weather = get_current_weather(city)

    # Check if the current_weather data is None due to an API error
    if current_weather is None:
        return HttpResponse("Error: Unable to fetch weather data. Please try again.")

    # Load historical weather data from a CSV file
    csv_path = os.path.join(os.path.dirname(__file__), "../data/weather.csv")
    historical_data = read_historical_data(csv_path)

    # Prepare data and train the rain prediction model
    X, y, le = prepare_data(
        historical_data
    )  # X, y are features and labels, le is the label encoder
    rain_model = train_rain_model(X, y)  # Train model for rain prediction

    # Map wind direction (in degrees) to compass points (N, NE, E, etc.)
    wind_deg = (
        current_weather["wind_gust_dir"] % 360
    )  # Normalize wind degree to a 0-360 range
    compass_points = [
        ("N", 0, 11.25),
        ("NNE", 11.25, 33.75),
        ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75),
        ("E", 78.75, 101.25),
        ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25),
        ("SSE", 146.25, 168.75),
        ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75),
        ("SW", 213.75, 236.25),
        ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25),
        ("WNW", 281.25, 303.75),
        ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75),
    ]

    # Find the compass direction based on the wind degree
    compass_direction = next(
        point for point, start, end in compass_points if start <= wind_deg < end
    )

    # Encode compass direction using the label encoder (le), if it exists in the classes
    compass_direction_encoded = (
        le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1
    )

    # Prepare the data for the current weather to make predictions
    current_data = {
        "MinTemp": current_weather["temp_min"],
        "MaxTemp": current_weather["temp_max"],
        "WindGustDir": compass_direction_encoded,
        "WindGustSpeed": current_weather["wind_gust_speed"],
        "Humidity": current_weather["humidity"],
        "Pressure": current_weather["pressure"],
        "Temp": current_weather["current_temp"],
    }

    # Convert current data into a DataFrame for model prediction
    current_df = pd.DataFrame([current_data])

    # Predict rain based on the current data
    rain_prediction = rain_model.predict(current_df)[0]

    # Prepare regression data for temperature and humidity
    X_temp, y_temp = prepare_regression_data(
        historical_data, "Temp"
    )  # Features and target for temperature
    X_hum, y_hum = prepare_regression_data(
        historical_data, "Humidity"
    )  # Features and target for humidity

    # Train regression models to predict temperature and humidity
    temp_model = train_regression_model(X_temp, y_temp)
    humidity_model = train_regression_model(X_hum, y_hum)

    # Predict future temperature and humidity values
    future_temp = predict_future(temp_model, current_weather["temp_min"])
    future_humidity = predict_future(humidity_model, current_weather["humidity"])

    # Prepare timestamps for the next 5 hours for future predictions
    timezone = pytz.timezone("Asia/Dhaka")  # Set the timezone
    now = datetime.now(timezone)  # Get the current time
    next_hour = now + timedelta(hours=1)  # Move to the next hour
    next_hour = next_hour.replace(
        minute=0, second=0, microsecond=0
    )  # Round to the hour

    # Generate a list of time labels for the next 5 hours
    future_times = [
        (next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)
    ]

    # store each value  seperately
    time1, time2, time3, time4, time5 = future_times
    temp1, temp2, temp3, temp4, temp5 = future_temp
    hum1, hum2, hum3, hum4, hum5 = future_humidity

    # pass the data to the template
    context = {
        "location": city,
        "current_temp": current_weather["current_temp"],
        "MinTemp": current_weather["temp_min"],
        "MaxTemp": current_weather["temp_max"],
        "feels_like": current_weather["feels_like"],
        "humidity": current_weather["humidity"],
        "clouds": current_weather["clouds"],
        "description": current_weather["description"],
        "city": current_weather["city"],
        "country": current_weather["country"],
        "time": datetime.now(),
        "date": datetime.now().strftime("%B %d, %Y"),
        "wind": current_weather["wind_gust_speed"],
        "pressure": current_weather["pressure"],
        "visibility": current_weather["visibility"],
        "rain_prediction": "Yes" if rain_prediction else "No",
        # predicted values
        "time1": time1,
        "time2": time2,
        "time3": time3,
        "time4": time4,
        "time5": time5,
        "temp1": f"{round(temp1, 1)}",
        "temp2": f"{round(temp2, 1)}",
        "temp3": f"{round(temp3, 1)}",
        "temp4": f"{round(temp4, 1)}",
        "temp5": f"{round(temp5, 1)}",
        "hum1": f"{round(hum1, 1)}",
        "hum2": f"{round(hum2, 1)}",
        "hum3": f"{round(hum3, 1)}",
        "hum4": f"{round(hum4, 1)}",
        "hum5": f"{round(hum5, 1)}",
    }

    return render(request, "weather.html", context)
