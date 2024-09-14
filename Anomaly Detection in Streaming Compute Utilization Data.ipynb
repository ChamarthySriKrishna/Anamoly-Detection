# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from google.colab import files
import io

# Upload files from local machine when prompted. (17 CSV files present in the computing-usage-dataset)
print("Please upload the CSV files:")
uploaded_files = files.upload()

# Create a dictionary to hold dataframes
data = {}

# Load data from uploaded files into the dictionary
for file_name in uploaded_files.keys():
    data[file_name] = pd.read_csv(io.BytesIO(uploaded_files[file_name]))

# Function to process and analyze each file
def process_file(df, metric_name):
    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Normalize data
    scaler = StandardScaler()
    df['value_normalized'] = scaler.fit_transform(df[['value']])

    # Anomaly Detection using Isolation Forest
    model = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly'] = model.fit_predict(df[['value_normalized']])
    
    # Map -1 to anomaly, 1 to normal
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
    
    # Categorize anomalies
    df['anomaly_type'] = np.nan
    df.loc[df['anomaly'] == 1, 'anomaly_type'] = 'Anomaly'
    df['anomaly_type'] = df['anomaly_type'].fillna('Normal')

    # Calculate severity scores (example: z-score of the normalized value)
    df['severity_score'] = np.where(df['anomaly'] == 1, np.abs(df['value_normalized']), 0)
    
    return df

# Process each file and collect results
results = {}
for file_name, df in data.items():
    metric_name = file_name.split('_')[1]  # Extract metric name from file name
    results[file_name] = process_file(df, metric_name)

# Save detected anomalies, categories, and severity scores to CSV
for file_name, df in results.items():
    output_file = f'anomalies_{file_name}'
    df.to_csv(output_file)
    print(f'Anomaly detection report saved to {output_file}')

# Create a visualization dashboard
def plot_dashboard(df, title):
    fig = go.Figure()
    
    # Plot normal data
    fig.add_trace(go.Scatter(x=df.index, y=df['value'], mode='lines', name='Value'))
    
    # Plot anomalies
    anomalies = df[df['anomaly'] == 1]
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['value'], mode='markers', name='Anomalies', marker=dict(color='red')))
    
    # Plot severity scores (if needed)
    if 'severity_score' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['severity_score'], mode='lines', name='Severity Score', line=dict(dash='dash')))
    
    fig.update_layout(title=title, xaxis_title='Timestamp', yaxis_title='Value')
    fig.show()

# Display dashboards for each metric
for file_name, df in results.items():
    plot_dashboard(df, f'{file_name} - Anomaly Detection')

# Example of scoring and summarizing results
def summarize_anomalies(df):
    total_anomalies = df['anomaly'].sum()
    critical_anomalies = df[df['severity_score'] > 1]  # Example threshold for critical anomalies
    print(f'Total anomalies detected: {total_anomalies}')
    print(f'Critical anomalies detected: {len(critical_anomalies)}')
    return total_anomalies, critical_anomalies

# Summary report
summary_report = {}
for file_name, df in results.items():
    total_anomalies, critical_anomalies = summarize_anomalies(df)
    summary_report[file_name] = {
        'Total Anomalies': total_anomalies,
        'Critical Anomalies': len(critical_anomalies),
        'Critical Anomalies Details': critical_anomalies[['value', 'anomaly_type', 'severity_score']].head()
    }

# Save insights report to a text file
insights_report_file = 'insights_report.txt'
with open(insights_report_file, 'w') as file:
    for file_name, summary in summary_report.items():
        file.write(f'File: {file_name}\n')
        file.write(f"Total Anomalies: {summary['Total Anomalies']}\n")
        file.write(f"Critical Anomalies: {summary['Critical Anomalies']}\n")
        file.write(f"Details of Critical Anomalies:\n{summary['Critical Anomalies Details']}\n")
        file.write('\n' + '-'*40 + '\n')

print(f'Insights report saved to {insights_report_file}')
