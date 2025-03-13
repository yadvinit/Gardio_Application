import pandas as pd
import ollama
import gradio as gr
import matplotlib.pyplot as plt
import io
from PIL import Image
import itertools

# Load and process CSV
def load_and_process_csv(file_path):
    df = pd.read_csv(file_path)
    df = df.fillna("")
    formatted_text = "\n".join([", ".join(f"{col}: {row[col]}" for col in df.columns) for _, row in df.iterrows()])
    return df, formatted_text

# Upload and process QA
def upload_and_process_qa(file):
    upload_and_process_graph(file)
    global csv_data, dataframe_qa
    dataframe_qa, csv_data = load_and_process_csv(file)
    return "QA File uploaded successfully! You can now ask questions."

# Upload and process graph
def upload_and_process_graph(file):
    global dataframe_graph
    dataframe_graph = pd.read_csv(file)
    return "Graph File uploaded successfully! You can now generate graphs."

# Query Ollama API
def query_ollama(context, question):
    prompt = f"""
    You are an AI assistant analyzing the following dataset:
    {context}
    
    Answer the following question based on the dataset:
    {question}
    """
    response = ollama.chat(model="llama3.1:latest", messages=[
        {"role": "system", "content": "You are an expert data analyst."},
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']

# Chatbot interface
def chatbot_interface(question):
    global csv_data
    if not csv_data:
        return "Please upload a CSV file first."
    return query_ollama(csv_data, question)

# Generate scatter plot graphs
def generate_graphs():
    global dataframe_graph
    if dataframe_graph is None:
        return "Please upload a CSV file first for graph generation."
    
    plots = []
    numerical_columns = dataframe_graph.select_dtypes(include=['number']).columns
    if len(numerical_columns) < 2:
        return "At least two numerical columns are required for graph generation."
    
    for col_x, col_y in itertools.combinations(numerical_columns, 2):
        fig, ax = plt.subplots()
        ax.scatter(dataframe_graph[col_x], dataframe_graph[col_y])
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_title(f'Scatter Plot: {col_x} vs {col_y}')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        plots.append(img)
    
    return plots

# Global variables
csv_data = ""
dataframe_qa = None
dataframe_graph = None

# Gradio Interface with Theme Selection
def set_theme(theme):
    themes = {
        "Soft": gr.themes.Soft(),
        "Base": gr.themes.Base(),
        "Glass": gr.themes.Glass()
    }
    return themes.get(theme, gr.themes.Soft())

demo = gr.Blocks(theme=set_theme("Soft"))

with demo:
    gr.Markdown("# 🧑‍💻 Data Analysis & Visualization Tool")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📁 Upload CSV File")
            file_upload = gr.File(label="Upload CSV", type="filepath")
            upload_button = gr.Button("Upload")
            progress_bar = gr.Markdown("")
            upload_button.click(upload_and_process_qa, inputs=file_upload, outputs=progress_bar)
        
        with gr.Column():
            gr.Markdown("### 💬 Ask Questions")
            chatbot_input = gr.Textbox(label="Ask a question", lines=2)
            submit_button = gr.Button("Submit")
            output_box = gr.Textbox(label="Chat Output", interactive=True, lines=9)
            submit_button.click(chatbot_interface, inputs=chatbot_input, outputs=output_box)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Generate Graphs")
            graph_output = gr.Gallery()
            generate_button = gr.Button("Generate Graphs")
            generate_button.click(generate_graphs, outputs=graph_output)
    
    with gr.Row():
        gr.Markdown("### 🎨 Select Theme")
        theme_selector = gr.Dropdown(label="Choose a Theme", choices=["Soft", "Base", "Glass"], value="Soft")
        apply_theme_button = gr.Button("Apply Theme")
        apply_theme_button.click(lambda theme: demo.set_theme(set_theme(theme)), inputs=theme_selector)

demo.launch()
