"""
@ Authors: Noam Amar along with me.
Robot Anomaly Detection - Data Preparation
This module loads robot sensor data and creates differential features for anomaly detection.
"""

# TODO: Style and add more concise titles

# TODO: Add titels for graphs
# TODO: Consider continous circumstances
# TODO: How many robots are there, for many clean how many with a fault
# TODO: What percentage of robots
# TODO: Explicit split between each robot (each robot represents a row)
# TODO: Slides representing the model, new slides.


# Generally required imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joblib

# Machine learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def create_diff_features(sub_features, df, features):
    """
    Create differential features between pairs of sensor measurements for each motor.

    This function computes the difference between each pair of features in sub_features
    for each motor and adds these differential features to the features dictionary.

    Args:
        sub_features (list): List of base feature names to create differences from
        df (DataFrame): DataFrame containing the original sensor data
        features (dict): Dictionary to store the created differential features

    Returns:
        None: Features are added to the provided features dictionary
    """
    motors = list(range(1, 7))  # Motors 1 through 6

    for i, f1 in enumerate(sub_features):
        for j in range(i + 1, len(sub_features)):
            f2 = sub_features[j]
            for motor in motors:
                motor_str = str(motor)
                feature_name = f"{f1}_diff_{f2}_{motor_str}"

                # Calculate difference between two measurements for this motor
                diff = np.subtract(
                    df[f"{f1}_{motor_str}"],
                    df[f"{f2}_{motor_str}"]
                )

                features[feature_name] = diff


# Load the robot sensor data
print("Loading robot sensor data...")
df = pd.read_parquet("./data.parquet")

# Display column information
print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
print("Available measurements:", ", ".join(sorted(list(set([col.split('_')[0] for col in df.columns if '_' in col])))))

# Initialize features dictionary
features = {}

# Create differential features for different measurement types
print("Generating differential features...")

# Velocity-related differential features
create_diff_features(["motor_velocity", "joint_velocity", "target_velocity"], df, features)

# Position-related differential features
create_diff_features(["motor_position", "joint_position", "target_position"], df, features)

# Torque-related differential features
create_diff_features(["motor_torque", "torque_sensor_a", "torque_sensor_b", "computed_torque"], df, features)

# Copy categorical and system-level measurements directly
print("Adding categorical and system measurements...")
metadata_cols = ["category", "anomaly", "sample", "time"]
system_cols = ["system_current", "robot_voltage", "robot_current"]


for col in metadata_cols + system_cols:
    features[col] = df[col]

# Copy motor current measurements
print("Adding motor current measurements...")
for i in range(1, 7):  # For motors 1-6
    features[f"motor_iq_{i}"] = df[f"motor_iq_{i}"]
    features[f"motor_id_{i}"] = df[f"motor_id_{i}"]

print(f"Feature generation complete. Total features created: {len(features)}")

# Convert features dictionary to DataFrame
our_data = pd.DataFrame(features)


def organize_samples_by_category(dataframe):
    """
    Organize time series samples into clean and anomaly categories.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the features with sample and anomaly columns

    Returns:
        tuple: (samples, clean_samples, anomaly_samples) where:
               - samples is a list of all sample time series
               - clean_samples is a list of normal operation samples
               - anomaly_samples is a list of lists (by category) of anomalous samples
    """
    samples = []
    clean_samples = []
    anomaly_samples = [[] for _ in range(12)]  # 12 anomaly categories

    # Group data by sample ID
    for _, group in dataframe.groupby('sample'):
        samples.append(group)

    # Categorize samples as clean or by anomaly type
    for sample in samples:
        first_point = sample.iloc[0]
        if first_point["anomaly"]:
            anomaly_samples[first_point["category"]].append(sample)
        else:
            clean_samples.append(sample)

    return samples, clean_samples, anomaly_samples


# Organize samples by their categories
samples, clean_samples, anomaly_samples = organize_samples_by_category(our_data)

# Print summary of sample distribution
print(f"Clean samples: {len(clean_samples)}")
for i in range(len(anomaly_samples)):
    if len(anomaly_samples[i]) > 0:
        print(f"Anomaly category {i}: {len(anomaly_samples[i])} samples")

# Select representative samples for visualization
clean_sample = clean_samples[3]
extra_friction_sample = anomaly_samples[0][3]  # Category 0 is "extra friction"

# Save all samples for later use
all_samples = {"clean": clean_samples, "anomalies": anomaly_samples}
with open('data.pkl', 'wb') as file:
    pickle.dump(all_samples, file)
    print("Sample data saved to data.pkl")


def plot_feature_comparison(clean_sample, anomaly_sample, motor_num, features_to_plot):
    """
    Plots a comparison of selected features between clean and anomalous samples.

    Args:
        clean_sample (pd.DataFrame): A time series of a normal operation
        anomaly_sample (pd.DataFrame): A time series of an anomalous operation
        motor_num (int): Motor number to examine
        features_to_plot (list): List of feature patterns to plot (motor number will be appended)

    Returns:
        plt.Figure: The matplotlib figure containing the plots
    """
    # Format feature names with motor number
    variables = [f"{feature}_{motor_num}" for feature in features_to_plot]
    num_vars = len(variables)

    # Create plot grid
    fig, axes = plt.subplots(num_vars, 2, figsize=(10, 2.5 * num_vars))

    # Create comparison plots
    for i, variable in enumerate(variables):
        # Clean sample plot
        axes[i, 0].plot(clean_sample["time"], clean_sample[variable])
        axes[i, 0].set_title(f"Normal Operation: {variable}")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_ylabel("Value")

        # Anomaly sample plot
        axes[i, 1].plot(anomaly_sample["time"], anomaly_sample[variable])
        axes[i, 1].set_title(f"Extra Friction: {variable}")
        axes[i, 1].set_xlabel("Time (s)")
        axes[i, 1].set_ylabel("Value")

    plt.tight_layout()
    return fig


# Plot differential features for each motor
print("Generating feature comparison plots...")
for motor_num in range(1, 7):
    # Features to compare between normal and anomalous operation
    feature_patterns = [
        "motor_velocity_diff_joint_velocity",
        "motor_velocity_diff_target_velocity"
    ]

    # Create comparison plots
    plot_feature_comparison(
        clean_sample,
        extra_friction_sample,
        motor_num,
        feature_patterns
    )
    # Uncomment to display plots
    # plt.show()

# Prepare data for machine learning
print("Preparing data for model training...")

# Define metadata columns that will be excluded from feature set
metadata_columns = ["category", "anomaly", "sample", "time"]
numeric_cols = [col for col in our_data.columns if col not in metadata_columns]

# Standardize numeric features
print("Standardizing numeric features...")
scaler = StandardScaler()
our_data[numeric_cols] = scaler.fit_transform(our_data[numeric_cols])
print("Standardization complete.")

# Model configuration
MODEL_CONFIG = {
    "HIDDEN_LAYERS": 2,
    "NODES_PER_LAYER": 64
}

# Prepare features (X) and target labels (y)
X = []
y = []

# Add normal samples (label 0)
print("Preparing normal samples...")
for sample in clean_samples:
    X.append(sample.drop(columns=metadata_columns))
    y.append(np.zeros(len(sample)))

# Add anomaly samples - focusing on extra friction category (label 1)
print("Preparing anomaly samples (extra friction)...")
for sample in anomaly_samples[0]:  # Category 0: extra friction
    X.append(sample.drop(columns=metadata_columns))
    y.append(np.ones(len(sample)))

# Combine all samples into single datasets
X = pd.concat(X, ignore_index=True)
y = np.concatenate(y)

print(f"Dataset prepared: {X.shape[0]} samples with {X.shape[1]} features")
print(f"Class distribution: {np.sum(y == 0)} normal, {np.sum(y == 1)} anomalous")


def train_anomaly_detection_model(X, y, model_params=None, save_path=None):
    """
    Train an anomaly detection model using a neural network.

    Args:
        X (pd.DataFrame): Feature dataset
        y (np.ndarray): Target labels
        model_params (dict, optional): Neural network hyperparameters
        save_path (str, optional): Path to save the trained model

    Returns:
        tuple: (trained_model, evaluation_metrics) containing the trained model and
               dictionary of evaluation metrics
    """
    # Set default parameters if none provided
    if model_params is None:
        model_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate_init': 0.001,
            'max_iter': 200,  # Increased from 1 to ensure convergence
            'random_state': 42,
            'verbose': True
        }

    # 1. Split the data with stratification to maintain class balance
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # 2. Create and train the neural network
    print("\nTraining neural network model...")
    print(f"Model architecture: {model_params['hidden_layer_sizes']} hidden layers")

    mlp = MLPClassifier(**model_params)
    mlp.fit(X_train, y_train)

    # 3. Evaluate the model
    print("\nEvaluating model performance...")
    y_pred = mlp.predict(X_test)

    # Calculate various metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'])

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Collect metrics
    evaluation_metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred
    }

    # 4. Save the model if a path is specified
    if save_path:
        joblib.dump(mlp, save_path)
        print(f"Model saved to {save_path}")

    return mlp, evaluation_metrics


def plot_confusion_matrix(confusion_matrix_data, class_names=['Normal', 'Anomaly']):
    """
    Plot and format a confusion matrix visualization.

    Args:
        confusion_matrix_data (numpy.ndarray): The confusion matrix
        class_names (list): Names of the classes
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_data,
                                  display_labels=class_names)

    disp.plot(cmap='Blues', values_format='d', ax=ax)

    # Add title and labels
    plt.title('Confusion Matrix', fontsize=15)
    plt.grid(False)

    # Add text with overall accuracy
    total = np.sum(confusion_matrix_data)
    correct = np.trace(confusion_matrix_data)
    accuracy = correct / total

    plt.figtext(0.5, 0.01, f'Overall Accuracy: {accuracy:.4f}',
                ha='center', fontsize=12)

    plt.tight_layout()
    return fig


def plot_model_learning_curve(model, X, y, cv=5):
    """
    Plot learning curves for the trained model to diagnose overfitting/underfitting.

    Args:
        model: Trained model
        X: Features
        y: Target labels
        cv: Number of cross-validation folds
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot learning curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid()
    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_mean - test_std,
                    test_mean + test_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

    ax.set_title("Learning Curve", fontsize=14)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="best")

    return fig


# Main execution
if __name__ == "__main__":
    # If you haven't defined X and y earlier, you would need to prepare them here

    # Define model hyperparameters - adjust as needed
    model_config = {
        'hidden_layer_sizes': (100, 50),  # Two hidden layers with 100 and 50 neurons
        'activation': 'relu',  # ReLU activation function
        'solver': 'adam',  # Adam optimizer
        'learning_rate_init': 0.001,  # Initial learning rate
        'max_iter': 200,  # Maximum number of iterations (epochs)
        'random_state': 42,  # Random seed for reproducibility
        'verbose': True,  # Print progress during training
        'early_stopping': True,  # Stop training when validation score doesn't improve
        'validation_fraction': 0.1  # Use 10% of training data for validation
    }

    # Train the model
    model_save_path = "robot_anomaly_detection_model.pkl"
    model, metrics = train_anomaly_detection_model(
        X, y,
        model_params=model_config,
        save_path=model_save_path
    )

    # Print evaluation results
    print("\n----- Model Evaluation Results -----")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nDetailed Classification Report:")
    print(metrics['classification_report'])

    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

    # Create and display evaluation-visualizations
    # 1. Confusion Matrix
    cm_fig = plot_confusion_matrix(metrics['confusion_matrix'])
    plt.figure(cm_fig.number)
    plt.savefig('confusion_matrix.png')

    # 2. Learning Curve
    lc_fig = plot_model_learning_curve(model, X, y)
    plt.figure(lc_fig.number)
    plt.savefig('learning_curve.png')

    print("\nVisualizations saved as 'confusion_matrix.png' and 'learning_curve.png'")

    # Verify the saved model works correctly
    # loaded_model = joblib.load(model_save_path)
    # loaded_accuracy = accuracy_score(metrics['y_test'], loaded_model.predict(X_test))
    # print(f"\nVerification - Loaded model accuracy: {loaded_accuracy:.4f}")

    plt.show()  # Display all figures
