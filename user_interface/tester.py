import gradio as gr
import time

def slowly_reverse(word, progress=gr.Progress()):
    progress(0, desc="Starting")
    time.sleep(1)
    progress(0.05)
    new_string = ""
    for letter in progress.tqdm(word, desc="Reversing"):
        time.sleep(0.25)
        new_string = letter + new_string
    return new_string

demo = gr.Interface(slowly_reverse, gr.Text(), gr.Text())

demo.launch()

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

# above code allows to display progress bar

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
