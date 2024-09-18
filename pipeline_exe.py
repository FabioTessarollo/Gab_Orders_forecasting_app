import subprocess

# List of your script file paths
scripts = ['data_source_and_smooth.py', 'data_features_extraction.py', 'data_fine_tuning_1.py', 'data_fine_tuning_2.py', 'data_model_evaluation.py']



# Execute each script in order
for script in scripts:
    subprocess.run(["python", script])