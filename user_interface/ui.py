import gradio as gr

# Start building the interface with Blocks
with gr.Blocks() as demo:
    
    gr.Markdown(""" # MCS4 - Securing Face ID """)
    
    with gr.Tab("Home"):
        gr.Markdown(""" init """)
    
    with gr.Tab("About"):
        gr.Markdown(""" init """)
    
    with gr.Tab("Deception"):
        gr.Markdown(""" init """)

    with gr.Tab("Dataset"):
        gr.Markdown(""" init """)
      
    with gr.Tab("Contact"):
        gr.Markdown(""" init """)


demo.launch()