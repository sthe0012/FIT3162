import gradio as gr
import pandas as pd

theme_code = {
    "base": "dark",
    "primary": "#ff6347",  # Tomato red for primary elements like buttons
    "secondary": "#ff6347",  # Same color for secondary elements
    "background": "#000000",  # Black background
    "text_color": "#FFFFFF",  # Ensuring text is visible against the dark background
    "font_family": "Arial"
}

# Start building the interface with Blocks
with gr.Blocks(theme=theme_code) as demo:

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
                gr.Image("t1.png", width=800, height=500)

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
            
            def ui(video):
                result = video_identity(video)
                if result: return "Authenticated"
                return result

            combined_ui = gr.Interface(
                    fn=ui,
                    inputs=gr.Video(),
                    outputs="text",
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
        
        def count_trained_data():
            total = 0
            path = 'dataset_1\Real-life_Deception_Detection_2016\Annotation\All_Gestures_Deceptive_and_Truthful.csv'
            data = pd.read_csv(path)
            total += len(data)
            
            return total
            
        gr.Interface(
            fn = count_trained_data,
            inputs=None,
            outputs="text",
            title="Datasets",
            description="The total of dataset that has trained"
        )
    
    with gr.Tab("Contact"):
        
        gr.Markdown("# The Team")
        
        gr.Image("t2.jpg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Name:** Shannon Theng\n"
                    "- **Role:** Product Manager\n"
                    "- **Email:** ..@student.monash.edu")
        
        gr.Image("t2.jpg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Name:** Jiahui\n"
                    "- **Role:** Quality Assurance\n"
                    "- **Email:** ..@student.monash.edu")
        
        gr.Image("t2.jpg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Name:** Jasper\n"
                    "- **Role:** Quality Assurance\n"
                    "- **Email:** kaww0003@student.monash.edu")
    
        gr.Image("t2.jpg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Name:** Jing Wei\n"
                    "- **Role:** Technical Lead\n"
                    "- **Email:** jong0074@student.monash.edu")
        
        gr.Image("t5.jpg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Name:** Jessie\n"
                    "- **Role:** Supervisor\n"
                    "- **Email:** leong.shumin@monash.edu\n"
                    "- **Phone:** +603-5516 1892")

        
        
# Launch the demo interface
demo.launch()
