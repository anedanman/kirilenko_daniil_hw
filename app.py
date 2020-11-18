from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import Required
from flask import request, redirect
import torch
import redis

from model import VQA_baseline

import os


app = Flask(__name__)

app.config['SECRET_KEY'] = 'some?bamboozle#string-foobar'
Bootstrap(app)
app.config['BOOTSTRAP_SERVE_LOCAL'] = True

path = os.path.dirname(os.path.abspath("app.py"))

model = VQA_baseline()

##model.load_state_dict(torch.load("model_dict.pth", map_location=torch.device('cpu')))

db = redis.StrictRedis(host="redis")

class QuestionForm(FlaskForm):
    usr_question = StringField('Ask a question to this image', validators=[Required()])
    usr_answer = StringField('Answer your question')
    submit = SubmitField('Submit')



@app.route("/", methods=["GET", "POST"])
def upload_image():
    filename = "images/pythia1.jpg"

    if request.method == "POST":
        if request.files:
            image = request.files["image"]

            image.save(os.path.join("./static/images", image.filename))
            filename = "images/" + image.filename

            print("Image saved")
            print(image.filename)

            #return redirect(request.url)

    form = QuestionForm()
    answer= ""

    if form.validate_on_submit():
        question = form.usr_question.data
        ground_truth_answer = form.usr_answer.data
        answer = model.forward(question, os.path.join("./static", filename))
     #   form.usr_question.data = ''
     #   form.usr_answer.data = ''
        db.hmset("image_name" + filename, dict(question=question, gt_answer=ground_truth_answer))

    return render_template("upload_image.html", form=form, message=answer, filename=filename, name='ddd')


if __name__ == '__main__':
    app.run(debug=True)
