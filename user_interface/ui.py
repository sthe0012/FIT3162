import io
import os
import sys
import glob
import csv
import cv2
import time
import pandas as pd
import gradio as gr
import numpy as np
from os import walk
from os.path import join
from os.path import splitext
import matplotlib.pyplot as plt
from transformers import pipeline

theme_code = {
    "base": "dark",
    "primary": "#ff6347",  # Tomato red for primary elements like buttons
    "secondary": "#ff6347",  # Same color for secondary elements
    "background": "#000000",  # Black background
    "text_color": "#FFFFFF",  # Ensuring text is visible against the dark background
    "font_family": "Arial"
}

def count_dataset()->int:
    total = 0
    total += count_trained_data_real_life()    
    return total

def count_trained_data_real_life():
        total = 0
        path = 'D:\fit3162\dataset\Real-life_Deception_Detection_2016\Annotation\All_Gestures_Deceptive_and_Truthful.csv'
        data = pd.read_csv(path)
        total += len(data)        
        return total

# Start building the interface with Blocks
with gr.Blocks(theme=theme_code) as mcs4ui:

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
        
        with gr.Tab("Real-time Camera Analysis"):
            
            def process_video(input):
                return input
            
            gr.Markdown("""
                        Unable to apply real-time camera at the moment.
                        """)

        with gr.Tab("Video Analysis"):
            
            def video_identify(video):
                if video is None: gr.Error("Input is empty")
                return np.random.random(),np.random.random()
        
            def ui(video):
                probability_of_authenticity, data_for_line_graph = video_identify(video)
                
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

                return bar_graph, line_graph, f"Probability of Authenticity: {probability_of_authenticity:.2f}"

            combined_ui = gr.Interface(
                    fn=ui,
                    inputs=gr.Video(),
                    outputs=["plot", "plot", "text"],  # Output both a plot and text
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

    with gr.Tab("Dataset"):
                    
        gr.Interface(
            fn = count_dataset,
            inputs=None,
            outputs="text",
            title="Datasets",
            description="The total of dataset that has trained"
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