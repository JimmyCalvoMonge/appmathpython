from openai import OpenAI
import yaml
import os

env_path = './env/env.yaml'
with open(env_path, 'r') as file:
    data = yaml.safe_load(file)
    OPENAI_API_KEY = data['environment']['api_key']

# Write a sample markdown notebook with the topics and sections laid out for each chapter in 'index.yaml'
# These notebooks are just a basis and later can be used to create our content and perfect it.

def obtain_llm_content(title, sections):

    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """
            You must write contents in an engaging, polite and textbook-like manner.
            Making good, concise and clear explanations and important remarks on the content provided
            Content should be laid out with clarity and following the proposed structure.
            All code snippets should be enclosed in ``` symbols.
            """},
            {
                "role": "user",
                "content": f"""
                Write a markdown notebook for the topic {title} with sections given by 
                {sections}
                For each section provide clear and complete mathematical explanations, code examples with Python and real applications.
                The contents of the notebook should be laid out like this:
                # Title
                ## Introduction
                ## Section 1
                ### Theory
                ### Examples
                ## Section 2
                ### Theory
                ### Examples
                ...
                """
            }
        ]
    )

    return completion.choices[0].message.content

def write_response_file(short_title, response):

    os.makedirs(f"./src/{short_title}", exist_ok=True)
    file_path = f"./src/{short_title}/sample_notebook.md"

    # Write content to the file
    with open(file_path, "w+", encoding="utf-8") as file:
        file.write(str(response))

def main():

    file_path = "index.yaml"

    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    for chapter in data['chapters']:

        title = chapter['title']
        short_title= chapter['short_title']
        sections = chapter['sections']

        print(f'Started for {title} ...')
        # Call LLM to create notebook
        response = obtain_llm_content(title, sections)
        # Save file
        write_response_file(short_title, response)
        print(f'Sample notebook for {title} created')

if __name__ == '__main__':
    main()