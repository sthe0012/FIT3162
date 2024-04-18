import gradio as gr

# Start building the interface with Blocks
with gr.Blocks() as demo:
    
    gr.Markdown(""" # MCS4 - Securing Face ID """)
    
    with gr.Tab("Home"):
        gr.Markdown(""" init """)
    
    with gr.Tab("About"):
        gr.Markdown(""" init """)
    
    with gr.Tab("Deception"):
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

    with gr.Tab("Dataset"):
        gr.Markdown(""" init """)
      
    with gr.Tab("Contact"):
        gr.Markdown("# The Team")
        
        gr.Image("t2.jpg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Role:** Quality Assurance\n"
                    "- **Email:** example@email.com\n"
                    "- **Phone:** +123456789")
        
        gr.Image("t5.jpg", width=150, height=200)
        gr.Markdown("### Contact Information\n"
                    "Please contact us at:\n"
                    "- **Role:** Supervisor\n"
                    "- **Email:** leong.shumin@monash.edu\n"
                    "- **Phone:** +603-5516 1892")


demo.launch()