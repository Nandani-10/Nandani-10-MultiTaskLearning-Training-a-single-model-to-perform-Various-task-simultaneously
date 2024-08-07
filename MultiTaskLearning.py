import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import cv2
import logging

# Initialize global variables
df = None
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC()
}
selected_model_name = 'Logistic Regression'
accuracy_scores = []

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Titanic dataset
def load_dataset():
    global df
    try:
        df = pd.read_csv('titanic.csv')
        logging.info("Dataset loaded successfully.")
        messagebox.showinfo("Info", "Dataset loaded successfully!")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        messagebox.showerror("Error", f"Error loading dataset: {e}")

# Data Visualization Task
def visualize_data():
    if df is not None:
        try:
            plt.figure(figsize=(10, 6))
            df['Survived'].value_counts().plot(kind='bar', color=['blue', 'orange'])
            plt.title('Survival Counts')
            plt.xlabel('Survived')
            plt.ylabel('Count')
            plt.show()
            logging.info("Data visualization completed.")
        except Exception as e:
            logging.error(f"Error visualizing data: {e}")
            messagebox.showerror("Error", f"Error visualizing data: {e}")
    else:
        messagebox.showwarning("Warning", "Load the dataset first!")

# Machine Learning Task
def train_model():
    global accuracy_scores
    if df is not None:
        try:
            X = df[['Pclass', 'Sex', 'Age', 'Fare']].copy()
            X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
            X.fillna(X.mean(), inplace=True)
            y = df['Survived']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = models[selected_model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)
            logging.info(f"Model trained ({selected_model_name}). Accuracy: {accuracy * 100:.2f}%")
            messagebox.showinfo("Model Accuracy", f"Model Accuracy: {accuracy * 100:.2f}%")
            plot_accuracy_graph()
            save_model()
        except Exception as e:
            logging.error(f"Error training model: {e}")
            messagebox.showerror("Error", f"Error training model: {e}")
    else:
        messagebox.showwarning("Warning", "Load the dataset first!")

# Save Model
def save_model():
    try:
        model = models[selected_model_name]
        joblib.dump(model, f'{selected_model_name}.pkl')
        logging.info(f"Model {selected_model_name} saved successfully.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        messagebox.showerror("Error", f"Error saving model: {e}")

# Load Model
def load_model():
    global selected_model_name
    try:
        model_file = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if model_file:
            model_name = model_file.split('/')[-1].replace('.pkl', '')
            if model_name in models:
                model = joblib.load(model_file)
                models[model_name] = model
                selected_model_name = model_name
                messagebox.showinfo("Model Loaded", f"Model {model_name} loaded successfully.")
                logging.info(f"Model {model_name} loaded successfully.")
            else:
                messagebox.showerror("Error", "Invalid model file.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        messagebox.showerror("Error", f"Error loading model: {e}")

# Cross-Validation Task
def cross_validate_model():
    if df is not None:
        try:
            X = df[['Pclass', 'Sex', 'Age', 'Fare']].copy()
            X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
            X.fillna(X.mean(), inplace=True)
            y = df['Survived']

            model = models[selected_model_name]
            scores = cross_val_score(model, X, y, cv=5)
            logging.info(f"Cross-validation scores for {selected_model_name}: {scores}")
            messagebox.showinfo("Cross-Validation", f"Cross-Validation Scores: {scores.mean() * 100:.2f}%")
        except Exception as e:
            logging.error(f"Error in cross-validation: {e}")
            messagebox.showerror("Error", f"Error in cross-validation: {e}")
    else:
        messagebox.showwarning("Warning", "Load the dataset first!")

# Plot accuracy graph
def plot_accuracy_graph():
    if accuracy_scores:
        try:
            fig, ax = plt.subplots()
            ax.plot(accuracy_scores, marker='o', linestyle='-', color='b')
            ax.set_title('Model Accuracy over Runs')
            ax.set_xlabel('Run')
            ax.set_ylabel('Accuracy')

            # Clear previous canvas if exists
            for widget in canvas_frame.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            logging.info("Accuracy graph updated.")
        except Exception as e:
            logging.error(f"Error plotting accuracy graph: {e}")
            messagebox.showerror("Error", f"Error plotting accuracy graph: {e}")
    else:
        messagebox.showwarning("Warning", "Train the model first!")

# Image Processing Task
def process_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            img = cv2.imread(file_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('gray_image.png', gray_img)
            gray_img_pil = Image.open('gray_image.png')
            gray_img_tk = ImageTk.PhotoImage(gray_img_pil)
            
            # Create a new window to display the image
            img_window = tk.Toplevel(root)
            img_window.title("Grayscale Image")
            panel = tk.Label(img_window, image=gray_img_tk)
            panel.image = gray_img_tk
            panel.pack()
            logging.info("Image processed and displayed successfully.")
            messagebox.showinfo("Info", "Image processed and displayed successfully!")
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            messagebox.showerror("Error", f"Error processing image: {e}")

# GUI Setup
root = tk.Tk()
root.title("Multi-Task Learning Application")
root.geometry("800x600")
root.configure(bg='#C75B7A')  # Set background color

# Welcome Label
welcome_label = tk.Label(root, text="WELCOME TO THE MULTI-TASK LEARNING", font=('Arial', 16, 'bold'), bg='#C75B7A', fg='white')
welcome_label.pack(pady=10)

# Frame for buttons
button_frame = tk.Frame(root, bg='#C75B7A')
button_frame.pack(pady=20)

# Button styles
button_styles = {
    'fg': 'white',
    'font': ('Arial', 12, 'bold'),
    'width': 20,
    'height': 2
}

# Model selection dropdown
model_var = tk.StringVar(value='Logistic Regression')
model_menu = ttk.Combobox(button_frame, textvariable=model_var, values=list(models.keys()), state='readonly')
model_menu.grid(row=0, column=0, padx=10, pady=10)

# Buttons for tasks
load_button = tk.Button(button_frame, text="Load Titanic Dataset", command=load_dataset, **button_styles, bg='#FF8C9E')
load_button.grid(row=0, column=1, padx=10, pady=10)

visualize_button = tk.Button(button_frame, text="Visualize Data", command=visualize_data, **button_styles, bg='#FFDA76')
visualize_button.grid(row=0, column=2, padx=10, pady=10)

train_button = tk.Button(button_frame, text="Train Model", command=lambda: train_model(), **button_styles, bg='#F0A8D0')
train_button.grid(row=0, column=3, padx=10, pady=10)

cross_val_button = tk.Button(button_frame, text="Cross-Validate Model", command=cross_validate_model, **button_styles, bg='#FFDA76')
cross_val_button.grid(row=1, column=0, padx=10, pady=10)

save_model_button = tk.Button(button_frame, text="Save Model", command=save_model, **button_styles, bg='#F0A8D0')
save_model_button.grid(row=1, column=1, padx=10, pady=10)

load_model_button = tk.Button(button_frame, text="Load Model", command=load_model, **button_styles, bg='#FF8C9E')
load_model_button.grid(row=1, column=2, padx=10, pady=10)

process_button = tk.Button(button_frame, text="Process Image", command=process_image, **button_styles, bg='#FF8C9E')
process_button.grid(row=1, column=3, padx=10, pady=10)

# Status bar
status_var = tk.StringVar()
status_var.set("Welcome to the Multi-Task Learning Application!")
status_bar = tk.Label(root, textvariable=status_var, bg='#C75B7A', fg='white', relief=tk.SUNKEN, anchor='w')
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Frame for the accuracy graph
canvas_frame = tk.Frame(root, bg='#C75B7A')
canvas_frame.pack(pady=20, fill=tk.BOTH, expand=True)

root.mainloop()
