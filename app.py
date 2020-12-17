from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextAreaField,SubmitField
from wtforms.validators import NumberRange

import numpy as np  
import spacy
import en_core_web_sm
from tensorflow.keras.models import load_model



def return_prediction(model,text):
    difficulty_map = {
        0: "Easy",
        1: "Intermediate",
        2: "Advanced"
    }

    nlp = en_core_web_sm.load()
    # For larger data features, you should probably write a for loop
    # That builds out this array for you

    vec = []
    for doc in nlp.pipe([text], batch_size=500):
        vec.append(doc.vector)

    X = np.expand_dims(np.array(vec), axis=2)
    
    class_ind = model.predict(X).argmax(axis=-1)
    
    return difficulty_map[class_ind[0]]



app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
text_model = load_model("model.h5")


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class TextForm(FlaskForm):
    input_text = TextAreaField('input_text')

    submit = SubmitField('Analyze')


@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = TextForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        input_text = form.input_text.data

        result = return_prediction(model=text_model,text=input_text)

        return render_template('home.html', form=form, result=result)

    return render_template('home.html', form=form)


if __name__ == '__main__':
    app.run()