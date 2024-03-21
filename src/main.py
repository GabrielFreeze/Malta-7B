import os
import gradio as gr
from time import sleep
from utils import color
from backend import SessionBackend, GlobalBackend


def main():

    def update_range(s_bk,from_date,to_date):
        s_bk.update_range(from_date,to_date)
        return s_bk   
    def update_sources(history,s_bk,thresh):
        print(f"\n\n CHAT HISTORY: {history}")
        return s_bk, s_bk.update_sources(history,thresh,g_bk.retrieval_model)  
    def update_sources_UI(s_bk,thresh):
        return s_bk.display_sources(thresh)

    def format_prompt(user_msg,history):
        return history + [[user_msg, None]]       
    def prepare_streamer(history, s_bk):
        s_bk.stop_generation = False
        s_bk.prepare_streamer(history,g_bk)
        return s_bk
    def stream_response(history,s_bk):
        yield from s_bk.stream_response(history,g_bk)
    
    def reset(s_bk:SessionBackend):
        s_bk.stop_generation = True
        sleep(0.6)
        s_bk.reset()
        return None,None,s_bk

    model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    revision = "main"

    if "HF_TOKEN" in os.environ:
        hf_token = os.environ["HF_TOKEN"]
    else:
        print(f'{color.YELLOW}Aborting.. HuggingFace token not set!\n{color.ESC} Powershell:{color.BLUE} $env:HF_TOKEN="<YOUR_TOKEN>"{color.ESC}')
        return
    
    g_bk = GlobalBackend(model_id,hf_token=hf_token,exllama=True, revision=revision)

    how_to_use:str = ""
    with open("how_to_use.txt", "r", encoding='utf-8') as f:
        how_to_use = f.read()
    
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        s_bk = gr.State(SessionBackend())

        gr.Markdown("""<div style="text-align: center; font-size: 24px;">Malta-7B</div>
                       <div style="text-align: center; font-size: 12px;"><i>L-Università ta' Malta</i></div>""")
        with gr.Row():
            with gr.Column():
                    # start_yr = gr.Slider(2018,2024,
                    #                      value=g_bk.default_from_yr, interactive=True,
                    #                      step=1,label="From", info="1st January of...")
                    # end_yr   = gr.Slider(2018,2024,
                    #                      value=g_bk.default_to_yr, interactive=True,
                    #                      step=1,label="To"  , info="31st December of...")               
                how_to_use = gr.Markdown(value=how_to_use)
                src_slider = gr.Slider(0.55,0.85, value=g_bk.default_src_thresh, interactive=True,
                                       step=0.01,label="Source Filter", info="Filter sources according to a relevancy threshold")
                sourceBox  = gr.Markdown(value="""<div style="text-align: center; font-size: 24px;">Sources</div>""")
            with gr.Column():
                single_yr   = gr.Slider(2018,2024,
                        value=g_bk.default_to_yr, interactive=True,
                        step=1,label="Sena", info="Choose year which the model will search in")   
                chatbot   = gr.Chatbot(height=500)
                msg       = gr.Textbox(label="User")
                clear     = gr.Button("Start New Topic")
            

                   
        single_yr .release(update_range     , [s_bk,single_yr,single_yr], s_bk     )
        src_slider.release(update_sources_UI, [s_bk,src_slider]         , sourceBox)


        (#===_===_===_FUNCTION_===_===_===_INPUT_===_===_OUTPUT_===_===_===_===_===
        msg.submit(  format_prompt , [msg,chatbot,           ], [chatbot       ], queue=False)
           .then  (  lambda: None  ,  None                    , [msg           ],            )
           .then  (prepare_streamer, [chatbot,s_bk,          ], [s_bk          ],            )
           .then  ( stream_response, [chatbot,s_bk,          ], [chatbot       ],            ) 
           .then  ( update_sources , [chatbot,s_bk,src_slider], [s_bk,sourceBox],            )
        )#===_===_===_===_===_===_===_===_===_===_===_===_===_===_===_===_===_===
        
        #Remove all texts & reset SessionBackend
        clear.click(reset,[s_bk], [chatbot,sourceBox,s_bk], queue=False)
        

        demo.queue(concurrency_count=1)
        demo.launch(share=True)


    return


if __name__ == '__main__':
    main()

    
