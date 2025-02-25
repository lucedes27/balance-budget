CHATML_TEMPLATE = """{% for message in messages %}\
    {% if message['role'] == 'user' %}\
        {{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n'}}\
    {% elif message['role'] == 'assistant' %}\
        {{'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}\
    {% else %}\
        {{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}\
    {% endif %}\
{% endfor %}\
{% if add_generation_prompt %}\
    {{ '<|im_start|>assistant\n' }}\
{% endif %}"""