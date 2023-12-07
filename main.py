import chainlit as cl
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from openai import OpenAI

client = OpenAI()
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer_2 = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M", src_lang="aka_Latn"
)
model_2 = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

def generate_response(input):
    completion = client.chat.completions.create(
      model="gpt-4",
      messages=[
        {"role": "user", "content": input}
      ]
    )
    return completion.choices[0].message.content

def eng_to_target(article):
    article = article
    inputs = tokenizer(article, return_tensors="pt")

    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["aka_Latn"], max_length=3000
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


def target_to_eng(article):
    article = article
    inputs = tokenizer_2(article, return_tensors="pt")

    translated_tokens = model_2.generate(
        **inputs, forced_bos_token_id=tokenizer_2.lang_code_to_id["eng_Latn"], max_length=3000
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...

    # Send a response back to the user
    message_in = target_to_eng(message.content)
    print(message_in)
    chat_response = generate_response(message_in)
    print(chat_response)
    message_out = eng_to_target(chat_response)
    print(message_out)

    await cl.Message(
        content=message_out
    ).send()
