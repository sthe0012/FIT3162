import io
import os
import sys
import glob
import csv
import cv2
import time
import joblib
import gradio as gr
import pandas as pd
import numpy as np
from os import walk
from os.path import join
from os.path import splitext
import matplotlib.pyplot as plt
from transformers import pipeline
from scipy.signal import resample

def compute_accuracy():
    pass

def count_training_distribution():
    
    csv_list = {
        r"dataset_file\\Annotation_mexp_features.csv",
        r"dataset_file\\Annotation_gaze_features.csv"    
    }
    all_labels = []
    
    for file in csv_list:
        current_file = pd.read_csv(file)
        all_labels.append(current_file['label'])
    
    combined_labels = pd.concat(all_labels, ignore_index=True)
    label_counts = combined_labels.value_counts()
    
    # Plotting the pie chart
    fig, ax = plt.subplots()
    ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen'])
    ax.axis('equal')
    ax.set_title('Label Distribution')
    return fig
        
def count_gender_distribution():
    csv_path = r"dataset_file\\gender_bias_trial_data_results.csv" 
    gender_bias_file = pd.read_csv(csv_path)
    
    # Extract gender and video count data
    gender = gender_bias_file['Gender']
    video_count = gender_bias_file['Video_Count']
    
    # Plotting the pie chart
    fig, ax = plt.subplots()
    ax.pie(video_count, labels=gender, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.
    
    return fig

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

def predict_inp(model, gaze_path, mexp_path, max_columns=576):
    
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

def count_dataset()->int:
    total = 0
    total += 121 
    return total

def get_model():
    model = joblib.load('multimodal_mexp_and_gaze.pkl')
    return model

# Start building the interface with Blocks
with gr.Blocks() as mcs4ui:

    gr.Markdown(""" # MCS4 - Securing Face ID """)
    
    with gr.Tab("Home"):
        with gr.Row():
            with gr.Column(): 
                gr.Markdown("# Start Detecting Deceptions With Uploading A Video")
                gr.Markdown("""
                            Deception prevails in many forms in society and can be broadly
                            categorized as deception in high-risk environments and incidental
                            deception.
                            """)
            with gr.Column():
                gr.Image("user_interface/images/t1.png", width=800, height=500)

    with gr.Tab("About"):
        gr.Markdown("""
        ## About the Deception System
        Deception detection refers to the process of identifying and distinguishing between truthfulness and deception. 
        It involves assessing and analysing various verbal and non-verbal cues, macro and micro facial expressions, and 
        body language. In this project, we will mainly be dealing with facial expressions and using features such as a 
        person’s eyebrow movement, lips, eyes, nose, cheeks, and chin. These features can be extracted and classified 
        into two groups, upper face action units and lower face action units (Torre et al., 2015). These facial action 
        units when analysed correctly, provide great detail in determining a person’s true emotion.
        """)

        gr.Image("project_goal.png", width=500, height=330)
        
        gr.Markdown("""
                    
        ### How to Use
        To use this system, select the Tab "Deception" and follow the instructions shown within it.
        
        ### Ethical Considerations
        Please use this system responsibly. It is designed for educational purposes and should not be used to create misleading content that could harm individuals or entities.

        ### Technical Details
        The system uses SVM machine learning models to generate and analyze deceptive patterns in digital media.

        ### About Us
        We are a team of undergraduate Computer Science students, MCS4 working on this project as part of our final year projects. 
        Our goal is to explore the complexities and applications of deception systems within digital environments, pushing the 
        boundaries of what's possible with current technology and contributing to academic understanding in this field.
        
        """)

    with gr.Tab("Deception"):
        
        with gr.Tab("Video Analysis"):
            
            def video_identify(video):
                
                # validation
                # if video is None: return gr.Error("Input is empty")
                # if not isinstance(video, str) or not video.lower().endswith('.mp4'): return gr.Error("Input must be an .mp4 file")

                # generate csv file (mexp & gaze)
                gaze_file = r"D:\\fit3162\\dataset\\output_gaze\\Gaze_reallifedeception_trial_lie_005.csv"
                mexp_file = r"D:\\fit3162\\dataset\\output_micro_expression\\Mexp_reallifedeception_trial_lie_005.csv"
                
                trained_model = get_model()
                result = predict_inp(trained_model, gaze_file, mexp_file)
                return "Truthful" if result == 0 else "Deceptive"
        
            def ui(video):
                result = video_identify(video)
                return f"Probability of Authenticity: {result:.2f}"

            combined_ui = gr.Interface(
                    fn=ui,
                    inputs=gr.Video(),
                    outputs=["text"],  # Output both a plot and text
                    title="Deception Detection System",
                    description="Displays the result of the input video to identify authenticity."
            )
        
        gr.Markdown("""       
            Steps: 
            
            1. Select different tab of Real-time Camera Analysis or Video Analysis.
            2. Select a video or upload a video at the video input.
            3. Click Submit button.
            4. Wait for the result.
            5. The result will appear in output section indicating truthfulness.
            """)

        with gr.Tab("Real-time Camera Analysis"):            
            gr.Markdown("""
                        Unable to apply real-time camera at the moment.
                        """)

    with gr.Tab("Dataset"):
                    
        gr.Interface(
            fn = count_gender_distribution,
            inputs=None,
            outputs="plot",
            title="Gender Bias Distribution"
        )
        
        gr.Markdown("""
            The graph has showed the overall percentages of gender bias to train the model.
            It has showed the gender bias distribution to allow users to visualise the 
            gender distribution (Latest updated: 13/5/2024)
                    """)

        gr.Interface(
            fn=count_training_distribution,
            inputs=None,
            outputs="plot",
            title="Training Data Distribution",
        )
        
        gr.Markdown("""
            The graph has showed the overall percentages of deceptive and truthful data to train the model.
            It has showed the label distribution to allow users to visualise the 
            percentages of real and deceptive distribution (Latest update: 13/5/2024)
                    """)

        gr.Interface(
            fn=count_dataset,
            inputs=None,
            outputs="text",
            title="Total data trained to build the model"
        )
        
        gr.Interface(
            fn=compute_accuracy,
            inputs=None,
            outputs="text",
            title="Overall Deceptive Detection System Accuracy "
        )
        
    
    with gr.Tab("Contact"):
        
        gr.Markdown("# The Team")
        
        gr.Image("user_interface/images/t2.jpg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Name:** Shannon Theng\n"
                    "- **Role:** Product Manager\n"
                    "- **Email:** ..@student.monash.edu")
        
        gr.Image("user_interface/images/t2.jpg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Name:** Jiahui\n"
                    "- **Role:** Quality Assurance\n"
                    "- **Email:** ..@student.monash.edu")
        
        gr.Image("user_interface/images/t2.jpg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Name:** Jasper\n"
                    "- **Role:** Quality Assurance\n"
                    "- **Email:** kaww0003@student.monash.edu")
    
        gr.Image("user_interface/images/t2.jpg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Name:** Jing Wei\n"
                    "- **Role:** Technical Lead\n"
                    "- **Email:** jong0074@student.monash.edu")
        
        gr.Image("user_interface/images/t5.jpg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Name:** Jessie\n"
                    "- **Role:** Supervisor\n"
                    "- **Email:** leong.shumin@monash.edu\n"
                    "- **Phone:** +603-5516 1892")

        
    
if __name__ == '__main__':
    mcs4ui.launch()