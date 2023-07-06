
import csv
import numpy as np
import tkinter as tk
from tkinter import messagebox
from LogisticRegression import LogisticRegression


def load_data(filename):
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        data = []
        for row in csv_reader:
            data.append([float(x) for x in row])
    return np.array(data)

def preprocess_data(data):
    X = data[:, :-1]  # Features (all columns except the last one)
    y = data[:, -1]   # Labels (last column)
    return X, y

# Step 1: Load and preprocess the data
data = load_data('diabetes2.csv')
np.random.seed(1)
np.random.shuffle(data)  # Shuffle the data randomly
X, y = preprocess_data(data)

# Step 2: Split the dataset into training and testing sets
num_samples = X.shape[0]
num_train = int(0.75 * num_samples)

X_train = X[:num_train]
y_train = y[:num_train]
X_test = X[num_train:]
y_test = y[num_train:]

# Step 3: Create an instance of LogisticRegression
model = LogisticRegression()

# Step 4: Fit the model using the training data
model.fit(X_train, y_train)

# Step 5: Evaluate the model's performance on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
print("Accuracy:", accuracy)


##phase 3

# Function to handle button click event
def on_predict():
    # Retrieve the user input from entry fields
    pregnancies = float(entry_pregnancies.get())
    glucose = float(entry_glucose.get())
    blood_pressure = float(entry_blood_pressure.get())
    skin_thickness = float(entry_skin_thickness.get())
    insulin = float(entry_insulin.get())
    bmi = float(entry_bmi.get())
    diabetes_pedigree = float(entry_diabetes_pedigree.get())
    age = float(entry_age.get())

    # Create a feature vector from the user input
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, diabetes_pedigree, age]])

    # Make predictions using the logistic regression model
    predictions = model.predict(features)

    # Show the prediction in a message box
    messagebox.showinfo("Prediction", f"The predicted class is: {predictions[0]}")

# Create the main application window
window = tk.Tk()
window.title("Diabetes Prediction")
window.geometry("300x400")

# Create entry fields for user input
label_pregnancies = tk.Label(window, text="Pregnancies:")
label_pregnancies.pack()
entry_pregnancies = tk.Entry(window)
entry_pregnancies.pack()

label_glucose = tk.Label(window, text="Glucose:")
label_glucose.pack()
entry_glucose = tk.Entry(window)
entry_glucose.pack()

label_blood_pressure = tk.Label(window, text="Blood Pressure:")
label_blood_pressure.pack()
entry_blood_pressure = tk.Entry(window)
entry_blood_pressure.pack()

label_skin_thickness = tk.Label(window, text="Skin Thickness:")
label_skin_thickness.pack()
entry_skin_thickness = tk.Entry(window)
entry_skin_thickness.pack()

label_insulin = tk.Label(window, text="Insulin:")
label_insulin.pack()
entry_insulin = tk.Entry(window)
entry_insulin.pack()

label_bmi = tk.Label(window, text="BMI:")
label_bmi.pack()
entry_bmi = tk.Entry(window)
entry_bmi.pack()

label_diabetes_pedigree = tk.Label(window, text="Diabetes Pedigree Function:")
label_diabetes_pedigree.pack()
entry_diabetes_pedigree = tk.Entry(window)
entry_diabetes_pedigree.pack()

label_age = tk.Label(window, text="Age:")
label_age.pack()
entry_age = tk.Entry(window)
entry_age.pack()

# Create a button for prediction
button_predict = tk.Button(window, text="Predict", command=on_predict)
button_predict.pack()

# Start the GUI event loop
window.mainloop()