# Iris Classification Project

## Overview
This project demonstrates a complete machine learning workflow using the classic Iris dataset. It includes:

- **Data Loading & Exploration**: Using scikit-learn and Pandas.
- **Data Visualization**: Leveraging Matplotlib and Seaborn.
- **Data Splitting**: Dividing the data into training and testing sets.
- **Model Training**: Using a RandomForestClassifier.
- **Model Evaluation**: Assessing performance with accuracy score and a confusion matrix.

## Data Description
The Iris dataset contains 150 samples, each with 4 features:

- **Sepal Length (cm)**
- **Sepal Width (cm)**
- **Petal Length (cm)**
- **Petal Width (cm)**

The target variable represents three iris species:
- Setosa
- Versicolor
- Virginica

## Technologies Used
- **Python**
- **Pandas** – Data manipulation.
- **Matplotlib** and **Seaborn** – Data visualization.
- **Scikit-learn** – Machine learning (data loading, model building, evaluation).

## Project Structure

```
Iris_Classification/
├── iris_classification.py      # Main script for data analysis, visualization, training, and evaluation
├── README.md                   # Project documentation (this file)
├── requirements.txt            # List of required Python libraries
└── screenshots/                # Folder containing visualization screenshots
    ├── histogram.png           # Histogram of feature distributions
    ├── pairplot.png            # Pairplot showing feature relationships by species
    └── confusion_matrix.png    # Confusion matrix of model predictions
```

## How to Run
1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/Iris_Classification.git
   ```

2. **Install the required libraries:**
   ```
   cd Iris_Classification
   pip install -r requirements.txt
   ```

3. **Run the main script:**
   ```
   python iris_classification.py
   ```

During runtime, the script prints the model accuracy (e.g., "Model Accuracy: X.XX") and displays various visualizations including:
- Histogram of Features
- Pairplot of Features
- Confusion Matrix

## Conclusion
This project showcases a complete machine learning pipeline on the Iris dataset by:
- Performing effective data exploration and visualization.
- Building and training a RandomForestClassifier.
- Evaluating model performance using standard metrics.

It demonstrates the ability to manage data, apply machine learning techniques, and present results in a professional manner.

## Contact
For further information or collaboration, please contact: [znewmich@gmail.com](mailto:znewmich@gmail.com)
