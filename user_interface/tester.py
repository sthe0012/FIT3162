import gradio as gr
import pandas as pd
import time
import os

def slowly_reverse(word, progress=gr.Progress()):
    progress(0, desc="Starting")
    time.sleep(1)
    progress(0.05)
    new_string = ""
    for letter in progress.tqdm(word, desc="Reversing"):
        time.sleep(0.25)
        new_string = letter + new_string
    return new_string

# demo = gr.Interface(slowly_reverse, gr.Text(), gr.Text())

def old_predict_inp(model, gaze_path, mexp_path, max_columns=576):
    
    # read csv
    csv_gaze, csv_mexp = pd.read_csv(gaze_path), pd.read_csv(mexp_path)
    # filter csv attributes
    gaze_data_clean, mexp_data_clean = pre_processing(csv_gaze), pre_processing(csv_mexp)
    # resample consistent samples
    gaze_data_resampled,mexp_data_resampled = resample(gaze_data_clean, 300),resample(mexp_data_clean, 300)
    # multimodal features (gaze, mexp)
    combined_features = np.hstack([gaze_data_resampled, mexp_data_resampled])

    adjusted_combined_data = {}
    
    for key, data in combined_data.items():
        current_columns = data.shape[1]
        if current_columns < max_columns:
            # Calculate how many columns to add
            additional_columns = max_columns - current_columns
            
            # Create an array of NaNs to add
            empty_columns = np.zeros((combined_features.shape[0], additional_columns))  # Change from np.nan to np.zeros
            
            # Concatenate the original data with the new empty columns
            new_data = np.hstack([data, empty_columns])
        else:
            new_data = data

        # Store the adjusted data back into the dictionary
        adjusted_combined_data[key] = new_data

    # Flatten the features into a single vector
    new_data_vector = combined_features.flatten().reshape(1, -1)

    # Check for NaN values and ensure the input data is valid
    valid_indices = ~np.isnan(new_data_vector).any(axis=1)
    new_data_vector_clean = new_data_vector[valid_indices]

    # Make a prediction using the trained model pipeline
    prediction = model.predict(new_data_vector_clean)

    # Output the prediction
    return 1 if prediction == 0 else 0  

def old_fx(video):
                result = video_identify(video)
                
                # Create the bar graph
                plt.figure()
                plt.bar(['Video'], [probability_of_authenticity], color='blue')
                plt.ylim(0, 1)
                plt.ylabel('Probability')
                plt.title('Action Unit Trigger')
                plt.grid(True)
                bar_graph = plt.gcf()  # Get the current figure to return to Gradio
                
                # Create the line graph
                plt.figure()
                plt.plot(data_for_line_graph, marker='o', linestyle='-', color='red')
                plt.title('Gaze Prediction')
                plt.xlabel('Time')
                plt.ylabel('Metric')
                plt.grid(True)
                line_graph = plt.gcf()  # Get the current figure to return to Gradio

def pre_processing(input):
    cols_to_drop = ["frame", "Unnamed: 0", "label", "face_id", "timestamp", "confidence", "success"]
    processed_file = input.drop([col for col in cols_to_drop if col in input.columns], axis=1).drop_duplicates()
    processed_file = np.array(processed_file)
    return processed_file

def extract_gaze_features(video_path):
    #openface_cmd ="/Users/jingweiong/openFace/OpenFace/build/bin/"         # onji
    openface_cmd ="D:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"        # jk

    # Extract the directory and filename from the video path
    directory, filename = os.path.split(video_path)
    base_filename = os.path.splitext(filename)[0]
    
    # Define the output path for the CSV
    output_csv = os.path.join(directory, f"Gaze_{base_filename}.csv")

    # Construct the command to run feature extraction with gaze tracking
    cmd = f"{openface_cmd} -f \"{video_path}\" -out_dir \"{os.path.dirname(output_csv)}\" -of \"{output_csv}\" -gaze"

    # Execute the command
    result = os.system(cmd)
    
    if result == 0: print(f"Gaze data extracted successfully for {filename}, saved to {output_csv}")
    else: print(f"Failed to extract gaze data for {filename}")
    return output_csv


# extract micro expression features into csv file
def extract_micro_features(video_path):
    #openface_cmd ="/Users/jingweiong/openFace/OpenFace/build/bin/"                             # onji
    openface_cmd ="D:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"      # jk

    directory, filename = os.path.split(video_path)
    base_filename = os.path.splitext(filename)[0]    
    output_csv = os.path.join(directory, f"Mexp_reallifedeception_{base_filename}.csv")
    cmd = f"{openface_cmd} -f \"{video_path}\" -out_dir \"{os.path.dirname(output_csv)}\" -of \"{output_csv}\" -pose -aus"
    result = os.system(cmd)
    
    if result == 0: print(f"Mexp data extracted successfully for {filename}, saved to {output_csv}")
    else: print(f"Failed to extract mexp data for {filename}")
    return output_csv

# Example usage
# video_path = "D:\\fit3162\dataset\Real-life_Deception_Detection_2016\Clips\Deceptive\\trial_lie_005.mp4"
# print(extract_micro_features(video_path))



def main():
    doggo_path = "dataset_file\doggo_output.csv"
    df = pd.read_csv(doggo_path)
    # Print the type of the 'confidence' column
    print(type(df['confidence']))

    # Check if all values in the 'confidence' column are equal to 0 or 0.0
    if (df['confidence'] == 0).all():
        print("All elements in the 'confidence' column are equal to 0")
    else:
        print("Not all elements in the 'confidence' column are equal to 0")

if __name__ == '__main__':
    main()