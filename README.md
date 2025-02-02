# Heart Disease Prediction Analysis

## Project Overview
This project focuses on analyzing and predicting heart disease using machine learning techniques. The dataset used includes key medical parameters that help identify the risk of heart disease. The analysis involves data preprocessing, exploratory data analysis (EDA), forward and back propagation. 

## Project Structure
- **Heartdisease.ipynb** – Jupyter Notebook containing data analysis, visualization, and model-building steps.
- **README.md** – Documentation for the project.

## Technologies Used
- Python
- Jupyter Notebook
- Pandas & NumPy for data manipulation
- Matplotlib & Seaborn for data visualization
- Scikit-learn for machine learning model training

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-analysis.git
   ```
2. Navigate to the project folder:
   ```bash
   cd heart-disease-analysis
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Heartdisease.ipynb
   ```

## Results & Insights
- Exploratory Data Analysis (EDA) provides insights into the dataset.
- Machine learning models are trained to predict heart disease risk.
- Model evaluation metrics such as accuracy, precision, and recall are analyzed.

How Backpropagation Works in This Project
The model follows these key steps:
1. Forward Propagation (Prediction Phase)
Input features (e.g., Age, Cholesterol, Blood Pressure) are passed through the neural network.
Each neuron applies a weighted sum function followed by an activation function (e.g., ReLU, Sigmoid).
The network makes an initial prediction (ŷ = output).

2. Compute Loss (Error Measurement)
The difference between the predicted output (ŷ) and the actual label (y) is measured using a loss function.

3. Backward Propagation (Error Correction)
The model computes the gradient of the loss function with respect to each weight using partial derivatives (via the chain rule in calculus).
Weight updates occur layer by layer, moving from the output layer backward to the input layer.

4. Weight Update (Learning Process)
The gradients are used to adjust weights via Gradient Descent.

5. Repeat Until Convergence
The forward and backward propagation steps repeat for multiple epochs until the model reaches optimal accuracy.

## Contributions
Contributions are welcome! Feel free to fork this repository and submit pull requests.


---
