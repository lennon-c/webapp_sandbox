
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    # add text using markdown on the main page
    "***Are your mushrooms edible or poisonous?*** 🍄"

    # add button to display  
    if st.sidebar.button("About this App"):
        """ 
        This code is adapted with minor modifications from the fantastic project on Coursera: [Build a Machine Learning Web App with Streamlit and Python](https://www.coursera.org/projects/machine-learning-streamlit-python) by [Snehan Kekre](https://www.coursera.org/instructor/snehan-kekre).

        I updated the code to ensure compatibility with Python 3.11 and the latest versions of key libraries:

        - **Streamlit**: 1.40.2
        - **Pandas**: 2.2.3
        - **NumPy**: 2.1.3
        - **scikit-learn**: 1.5.2 
        - **Matplotlib**: 3.9.2

        ### Main Changes:
        1. Transitioned from the older `plot_confusion_matrix`, `plot_roc_curve`, and `plot_precision_recall_curve` functions to their modern counterparts (`ConfusionMatrixDisplay`, `RocCurveDisplay`, and `PrecisionRecallDisplay`) from `scikit-learn`.
        2. Explicitly passed `Axes` and `Figures` objects to `Streamlit` for plot rendering.
        3. `@st.cache(persist=True)` (deprecated) was replaced **with** `@st.cache_data`.
        4. Grouped some common functionalities.

        Modified code on [GitHub - webapp_sandbox](https://github.com/lennon-c/webapp_sandbox). Code : [1_Classifiers.py](https://github.com/lennon-c/webapp_sandbox/blob/900b3c457e74b672be5d7aa18cdaeb3dc82683b3/webapp_sandbox/pages/1_Classifiers.py)
        """

    # @st.cache(persist=True) # deprecated 
    @st.cache_data
    def load_data():
        """Loads the data and returns it.

        Transform labels to encoded values
        
        Data source: https://archive.ics.uci.edu/dataset/73/mushroom
        """
        data = pd.read_csv(r"webapp_sandbox/mushrooms.csv")
        label=LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache_data
    def split(df):
        """Splits the data into train and test sets"""
        y = df.type # first column
        x = df.drop(columns=['type']) # the rest except the first
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(model, metrics_list):
        """Plots the metrics
        
        Plots from sklearn metrics, they all use matplotlib under the hood.
        """
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names, ax=ax)
            st.pyplot(fig=fig)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig=fig)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig=fig)
        
    def calculate_metrics(model):
        """Calculates accuracy, precision, recall and prints them on the web app."""
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy = model.score(x_test, y_test)
        st.write("Accuracy: ", f'{accuracy:.2f}')

        precision = precision_score(y_test, y_pred, labels=class_names)
        st.write("Precision: ", f'{precision:.2f}')

        recall = recall_score(y_test, y_pred, labels=class_names)
        st.write("Recall: ", f'{recall:.2f}')


    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']
    classifiers = ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest")
    metrics_labels = ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve')

    
 
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",  classifiers, label_visibility='collapsed')
    st.sidebar.subheader("What metrics to plot?")
    metrics = st.sidebar.multiselect("What metrics to plot?", metrics_labels,label_visibility='collapsed')
    
    if classifier== 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("***C (Regularization parameter)***"
                                    , 0.01, 10.0
                                    , step=0.01 # increment by 0.01 on the slider
                                    , key='C_SVM' # unique key for the slider
                                    )
    
        kernel = st.sidebar.radio("***Kernel***", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("***Gamma (Kernel Coefficient)***", ("scale", "auto"), key='gamma')

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            calculate_metrics(model)
            plot_metrics(model, metrics)
            
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("***C (Regularization parameter)***"
                                    , 0.01, 10.0
                                    , step=0.01
                                    , key='C_LR')
 
        max_iter = st.sidebar.slider("***Maximum number of iterations***"
                                     , 100, 500
                                     , key='max_iter')

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            calculate_metrics(model)
            plot_metrics(model, metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("***The number of trees in the forest***", 100, 5000, step=100, key='n_estimators')
        max_depth = st.sidebar.number_input("***The maximum depth of the tree***", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("***Bootstrap samples when building trees***", ('True', 'False'), key='bootstrap')
 
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators
                                           , max_depth=max_depth, bootstrap= True if bootstrap == 'True' else False	
                                           , n_jobs=-1) # use all processors
            calculate_metrics(model)
            plot_metrics(model, metrics)


    # Showing the raw data
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)
        st.markdown("This [data set](https://archive.ics.uci.edu/ml/datasets/Mushroom) includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms "
        "in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, "
        "or of unknown edibility and not recommended. This latter class was combined with the poisonous one.")

if __name__ == '__main__':
    main()
