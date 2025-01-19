
def make_three_words(input_text: dict):

    model_name = input_text['model']
    text = input_text['search']

    text_list = text.split(' ')

    if len(text_list) < 3:
        text_list = ["Harry", "Potter", text_list[-1]]
    else:
        text_list = text_list[-3:]

    return model_name, text_list


def deserialize(output_text: list, model_name: str, input_text: str):
    body = "\n".join([f"{text[0]:<25}{str(text[1])[:4]:>25}" for text in output_text])

    header = f"of {model_name} for '{input_text[-1]}' with their similarity scores:\n{body}"
    
    return header
