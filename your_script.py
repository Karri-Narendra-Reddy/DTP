import json
import pandas as pd
import os
from openai import AzureOpenAI

def run_scripts(data):
    grouped_data = group_data(data)
    combined_data = combine_data(grouped_data)
    json_data = combined_data.to_json(orient="records")
    return json_data

def group_data(data):
    data['Has_Statement'] = data['Statement'].notnull()
    grouped_data = data.groupby(['Question Number', data['Has_Statement'].cumsum()]).agg({
        'Question Number': 'first',
        'Question Text': 'first',
        'Statement': lambda x: x.iloc[0] if x.notnull().any() else None,
        'Question Instructions': 'first',
        'Answer': lambda x: ', '.join(x),
        'Answer Number': lambda x: ', '.join(map(str, x)),
        'Answer Type': 'first',
        'Question Type': 'first'
    }).reset_index(drop=True)
    grouped_data.drop(columns=['Has_Statement'], inplace=True, errors='ignore')
    return grouped_data

def combine_data(data):
    data['Answer Number'] = data['Answer Number'].astype(str)
    merged_data = data[data['Statement'].isnull()].groupby('Question Number').agg({
        'Question Text': lambda x: ', '.join(x),
        'Question Instructions': lambda x: ', '.join(x),
        'Answer': lambda x: ', '.join(x),
        'Answer Number': lambda x: ', '.join(x),
        'Answer Type': 'first',
        'Question Type': 'first'
    }).reset_index()
    statement_data = data[data['Statement'].notnull()].groupby('Statement').agg({
        'Question Number': lambda x: ', '.join(map(str, x)),
        'Question Text': lambda x: ', '.join(x),
        'Question Instructions': lambda x: ', '.join(x),
        'Answer': lambda x: ', '.join(x),
        'Answer Number': lambda x: ', '.join(x),
        'Answer Type': 'first',
        'Question Type': 'first'
    }).reset_index()
    final_data = pd.concat([merged_data, statement_data]).reset_index(drop=True)
    return final_data

def get_product_questions_and_answers(training_data,product):
    prompt_message = (
        "So I have JSON which contains questions and answers for shampoo category. "
        "Consider you are a survey taker. Can you please come up with generating questions and answers for " f"{product}\n" " buyers? "
        "Provide the output in JSON format only and make sure answers must be in array format. Give at least 6 different questions and I want the questions must be in MCSS, MCMS, and GRID."
        "Here are the existing questions and answers:\n"
        f"{json.dumps(training_data, indent=2)}\n"
    )
    os.environ["AZURE_OPENAI_API_VERSION"] = "2023-12-01-preview"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://analyst-augmentation-non-us.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_KEY"] = "73141bf1b4eb4f6bba5216cb6a8e0a33"
    os.environ["OPENAI_API_TYPE"] = "azure_ad"
    os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4-32k"
    client = AzureOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    completion = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        messages=[{"role": "user","content": prompt_message}],
        max_tokens=4096,
    )
    return completion.choices[0].message.content

def get_Answers_from_persona(questions_and_answers, personas):


    prompts = (
        "So I have a few personas listed below:\n"
        f"{personas}\n"
        "You have to take the below survey of questions with the respective individual persona listed above and return me survey responses as an output in a JSON format, mentioning the persona name and persona opted answer with reason in key-value pair."
        "Here are the existing questions and answers:\n"
        f"{questions_and_answers}\n"
        "Make sure all personas take the survey and respond to all questions from the above questions. \n"
        "and also give me the summary of all persona survey responses in a paragraph highlighting strengths and watchouts"
    )


    os.environ["AZURE_OPENAI_API_VERSION"] = "2023-12-01-preview"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://analyst-augmentation-non-us.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_KEY"] = "73141bf1b4eb4f6bba5216cb6a8e0a33"
    os.environ["OPENAI_API_TYPE"] = "azure_ad"
    os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4-32k"
    client = AzureOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    completion = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        messages=[{"role": "user", "content": prompts}],
        max_tokens=4096,
    )
    return completion.choices[0].message.content


# //persona type: role
# //role: //generic/software profession
def createpersona(persona_type,role, product_name):
    persona_type = persona_type.upper()
    if(persona_type == "ROLE"):
        prompt = (
            "Write me 5 different" f"{persona_type} "
            "based personas with the role of " f"{role} " " interested in buying" f"{product_name} "
            ".Include a short biography about their age, goals, needs and wants, pain points, motivations and what influences them most in 150 words. "
            "Give me a table of all the personas which have the probability of buying the "f"{product_name}? Also list their buying capability and their preferences to the "f"{product_name} ?"
        )
    elif(persona_type == "GOAL"):
        prompt = (
            "Write me 5 different" f"{persona_type} "
            "based personas with the Goal of " f"{role} " " interested in buying" f"{product_name} "
            ".Include a short biography about their age, goals, needs and wants, pain points, motivations and what influences them most in 150 words. "
            "Give me a table of all the personas which have the probability of buying the "f"{product_name}? Also list their buying capability and their preferences to the "f"{product_name} ?"
        )
    else:
        prompt = (
            "IGNORE EVERYTHING BEFORE THIS PROMPT."
            "Write me 5 different" f"{persona_type} "
            "based personas with the Scenario of " f"{role} " " interested in buying" f"{product_name} "
            ".Include a short biography about their age, goals, needs and wants, pain points, motivations and what influences them most in 150 words. "
            "Give me a table of all the personas which have the probability of buying the "f"{product_name}? Also list their buying capability and their preferences to the "f"{product_name} ?"
        )

    os.environ["AZURE_OPENAI_API_VERSION"] = "2023-12-01-preview"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://analyst-augmentation-non-us.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_KEY"] = "73141bf1b4eb4f6bba5216cb6a8e0a33"
    os.environ["OPENAI_API_TYPE"] = "azure_ad"
    os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4-32k"
    client = AzureOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    completion = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    return completion.choices[0].message.content

