import json
import pandas as pd
import os
from flask import Flask, request, render_template
from your_script import run_scripts, get_product_questions_and_answers, createpersona, get_Answers_from_persona

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    product_name = request.form['product_name']  #product name
    persona_type = request.form['persona_type']  #role/goal/scenario
    role = request.form['role']    #Software engineer/ dairy farmer
    data = pd.read_csv("reference-files/sample_data.csv")  # Adjust this line if the input data changes
    training_data = run_scripts(data)

    questions_and_answers = get_product_questions_and_answers(training_data,product_name)
    personas = createpersona(persona_type,role,product_name)
    survey_results = get_Answers_from_persona(questions_and_answers, personas)

    return render_template('result.html', product_name=product_name, questions_and_answers=questions_and_answers, personas=personas, survey_results=survey_results, role=role ,persona_type=persona_type)

if __name__ == '__main__':
    app.run(debug=True)
