import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    system_message = "Welcome to Trip Planning Buddy✈️! I’m here to help you plan your perfect trip. Whether you need recommendations for destinations, help with booking flights and accommodations, or tips on local attractions and activities, I've got you covered. Just let me know how I can assist you today, and we'll get started on planning your amazing journey!"
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value = " I’m here to help you plan your perfect trip. Whether you need recommendations for destinations, help with booking flights and accommodations, or tips on local attractions and activities, I've got you covered. Just let me know how I can assist you today, and we'll get started on planning your amazing journey.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],

    examples = [ 
        ["I want to plan a trip to Europe, but I'm not sure where to start."],
        ["Can you recommend some budget-friendly destinations for a solo traveler?"],
        ["What are the best things to do in Paris for a first-time visitor?"],
        ["How do I find cheap flights for my trip to Japan?"]
    ],
    title = 'Trip Planning Buddy✈️'
)


if __name__ == "__main__":
    demo.launch()