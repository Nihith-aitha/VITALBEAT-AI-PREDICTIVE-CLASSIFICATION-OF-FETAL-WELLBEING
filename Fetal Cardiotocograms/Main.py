
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
from tkinter import simpledialog
from tkinter import filedialog
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pymysql
import tkinter as tk
from tkinter import messagebox
import pymysql
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


labels = ['Normal', 'Suspect', 'Pathological']

def uploadDataset():
    global filename, dataset, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))


def DatasetPreprocessing():
    text.delete('1.0', END)
    global X, Y, dataset, label_encoder

    dataset.fillna(0, inplace = True)
    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values


    for i in range(len(types)):
        name = types[i]
        if name == 'object':  # finding column with object type
            le = LabelEncoder()
            dataset[columns[i]] = le.fit_transform(dataset[columns[i]].astype(str))
            label_encoder[columns[i]] = le   # save encoder with column name

    joblib.dump(label_encoder, "model/LabelEncoders.pkl") 
    
    X = dataset.drop(['fetal_health'], axis = 1)
    Y = dataset['fetal_health']
    labels, label_count = np.unique(dataset['fetal_health'], return_counts=True)

    text.insert(END,"Dataset Normalization & Preprocessing Task Completed\n\n")
    text.insert(END,str(dataset)+"\n\n")
    #dataset preprocessing such as replacing missing values, normalization and splitting dataset into train and test
    smote = SMOTE(random_state=42)
    X, Y = smote.fit_resample(X, Y)
# Count after SMOTE
    labels_resampled, label_count_resampled = np.unique(Y, return_counts=True)
# Plotting
    plt.figure(figsize=(10, 5))
# Before SMOTE
    plt.subplot(1, 2, 1)
    plt.bar(labels, label_count, color='skyblue', alpha=0.8)
    plt.xlabel("Output Type")
    plt.ylabel("Count")
    plt.title("Before SMOTE")

    # After SMOTE
    plt.subplot(1, 2, 2)
    plt.bar(labels_resampled, label_count_resampled, color='lightgreen', alpha=0.8)
    plt.xlabel("Output Type")
    plt.ylabel("Count")
    plt.title("After SMOTE")
    plt.tight_layout()
    plt.show()


def Train_test_splitting():
    text.delete('1.0', END)
    global X, Y, dataset, label_encoder
    global X_train, X_test, y_train, y_test, scaler

 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3) #split dataset into train and test
    text.insert(END,"Dataset Train & Test Splits\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"70% dataset used for training  : "+str(X_train.shape[0])+"\n")
    text.insert(END,"30% dataset user for testing   : "+str(X_test.shape[0])+"\n")


def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show() 




def Existing_DTC():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore
    global X_train, y_train, X_test, y_test
    
    accuracy = []
    precision = []
    recall = []
    fscore = []
    
    #check whether the model exists or not
    if os.path.exists('model/DTC_weights.pkl'):
        classifier = joblib.load('model/DTC_weights.pkl')
    else:
        classifier = DecisionTreeClassifier(random_state=4)
        classifier.fit(X_train, y_train)
        joblib.dump(classifier, 'model/DTC_weights.pkl')

    y_pred_dtc = classifier.predict(X_test)
    calculateMetrics("Existing DTC", y_test, y_pred_dtc)


def Existing_LRC():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore
    global X_train, y_train, X_test, y_test
    
    accuracy = []
    precision = []
    recall = []
    fscore = []
    
    if os.path.exists('model/LRC_weights.pkl'):
        classifier = joblib.load('model/LRC_weights.pkl')
    else:
        classifier = LogisticRegression(max_iter=1000, solver="lbfgs")
        classifier.fit(X_train, y_train)
        joblib.dump(classifier, 'model/LRC_weights.pkl')

    y_pred_lrc = classifier.predict(X_test)
    calculateMetrics("Existing LRC", y_test, y_pred_lrc)




def Existing_NBC():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore
    global X_train, y_train, X_test, y_test
    
    accuracy = []
    precision = []
    recall = []
    fscore = []
    
    if os.path.exists('model/NBC_weights.pkl'):
        classifier = joblib.load('model/NBC_weights.pkl')
    else:
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        joblib.dump(classifier, 'model/NBC_weights.pkl')

    y_pred_nbc = classifier.predict(X_test)
    calculateMetrics("Existing NBC", y_test, y_pred_nbc)


def Proposed_Classifier():
    global classifier
    text.delete('1.0', END)
    global X_train, y_train, X_test, y_test
    if os.path.exists('model/LightGBM_weights.pkl'):
        # Load the model from the pkl file
        classifier = joblib.load('model/LightGBM_weights.pkl')
    else:
        # Train the classifier on the training data
        classifier = LGBMClassifier(random_state=42)
        classifier.fit(X_train, y_train)
        # Save the model weights to a pkl file
        joblib.dump(classifier, 'model/LightGBM_weights.pkl')
    
    y_pred = classifier.predict(X_test)
    calculateMetrics("Proposed Light GBM", y_test, y_pred)

     


def predict():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")#upload test data
    dataset = pd.read_csv(filename)#read data from uploaded file
    dataset.fillna(0, inplace = True)#removing missing values
    classifier = joblib.load('model/LightGBM_weights.pkl')

    labels = ['Normal', 'Suspect', 'Pathological']
    predict = classifier.predict(dataset)   
    predict_labels = [labels[int(p) - 1] for p in predict]


    df_predictions = pd.DataFrame( predict_labels, columns=['Predicted_Outcome'] )
    predictions = pd.concat(
        [dataset.reset_index(drop=True), df_predictions],
        axis=1
    )

    for index, row in predictions.iterrows():
        row_dict = row.to_dict()
        text.insert(tk.END, f"Row {index + 1}: {row_dict}\n\n")


def connect_db():
    return pymysql.connect(host='localhost', user='root', password='root', database='fetal_db')

# Signup Functionality
def signup(role):
    def register_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_db()
                cursor = conn.cursor()
                query = "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)"
                cursor.execute(query, (username, password, role))
                conn.commit()
                conn.close()
                messagebox.showinfo("Success", f"{role} Signup Successful!")
                signup_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Database Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x300")
    signup_window.title(f"{role} Signup")

    tk.Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)

    tk.Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)

# Login Functionality
def login(role):
    def verify_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_db()
                cursor = conn.cursor()
                query = "SELECT * FROM users WHERE username=%s AND password=%s AND role=%s"
                cursor.execute(query, (username, password, role))
                result = cursor.fetchone()
                conn.close()
                if result:
                    messagebox.showinfo("Success", f"{role} Login Successful!")
                    login_window.destroy()
                    if role == "Admin":
                        show_admin_buttons()
                    elif role == "User":
                        show_user_buttons()
                else:
                    messagebox.showerror("Error", "Invalid Credentials!")
            except Exception as e:
                messagebox.showerror("Error", f"Database Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    tk.Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)

# Admin Button Functions
def show_admin_buttons():
    clear_buttons()
    tk.Button(main, text="Upload Dataset", command=uploadDataset, font=font1).place(x=50, y=200)
    tk.Button(main, text="Preprocess and SMOTE", command=DatasetPreprocessing, font=font1).place(x=220, y=200)
    tk.Button(main, text="Train Test Splitting", command=Train_test_splitting, font=font1).place(x=450, y=200)
    tk.Button(main, text="DTC Model", command=Existing_DTC, font=font1).place(x=650, y=200)
    tk.Button(main, text="LRC Model", command=Existing_LRC, font=font1).place(x=800, y=200)
    tk.Button(main, text="NBC Model", command=Existing_NBC, font=font1).place(x=950, y=200)
    tk.Button(main, text="LGBM Model", command=Proposed_Classifier, font=font1).place(x=1100, y=200)

# User Button Functions
def show_user_buttons():
    clear_buttons()
    tk.Button(main, text="Prediction", command=predict, font=font1).place(x=550, y=200)
    #tk.Button(main, text="Comparison Graph", command=comparison_graph, font=font1).place(x=400, y=400)

# Clear buttons before adding new ones
def clear_buttons():
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button, user_button]:
            widget.destroy()
    

# Main tkinter window
main = tk.Tk()
main.geometry("1300x1200")
main.title("Machine Learning Approach for Fetal Health Classification in Intensive Care Unit")

# Title
font = ('times', 18, 'bold')
title = tk.Label(main, text="Machine Learning Approach for Fetal Health Classification in Intensive Care Unit", bg='white', fg='black', font=font, height=2, width=100)
title.pack()

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=170)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=100,y=250)
text.config(font=font1)


# Admin and User Buttons
font1 = ('times', 14, 'bold')


tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=20, height=2, bg='LightBlue').place(x=100, y=100)

tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=20, height=2, bg='LightGreen').place(x=400, y=100)


admin_button = tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=20, height=2, bg='LightBlue')
admin_button.place(x=700, y=100)

user_button = tk.Button(main, text="User Login", command=lambda: login("User"), font=font1, width=20, height=2, bg='LightGreen')
user_button.place(x=1000, y=100)

main.config(bg='OliveDrab2')
main.mainloop()