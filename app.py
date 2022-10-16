import cohere
from cohere.classify import Example
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)


class Prediction(Resource):
    def post(self):
        co = cohere.Client("zICvk7J5i84T4u7DvBbjy9IUnAwXJWdGc4iDrIdh")

        examples = [Example("How many times should exercise in a day", "Fitness"),
                    Example(
                        "What kind of exercise can i do to lose weight", "Fitness"),
                    Example("What exercise rountine should i follow", "Fitness"),
                    Example("How do I gain muscle fast", "Fitness"),
                    Example("How do I remain healthy through exercise", "Fitness"),
                    Example("Is too much exercise bad", "Fitness"),
                    Example("How do I remain healthy through exercise", "Fitness"),
                    Example("How to decrease fat content", "Fitness"),
                    Example("How to change fat to muscle", "Fitness"),
                    Example("Is it possible to lose weight fast", "Fitness"),
                    Example("Why am I gaining weight so fast", "Fitness"),

                    Example("How many times should i eat in a day", "Diet"),
                    Example("Can I eat sugary foods", "Diet"),
                    Example("What can I eat to increase muscle", "Diet"),
                    Example("How can I lose fat through food", "Diet"),
                    Example("What kind of fruits should I eat", "Diet"),
                    Example("What carbohydrates are the best for energy", "Diet"),
                    Example("What time should I eat", "Diet"),
                    Example("What is Keto", "Diet"),
                    Example("Is being vegan good", "Diet"),
                    Example("Should I become vegetarian", "Diet"),
                    Example("How many calories should I eat a day", "Diet"),
                    Example("How long does alcohol stay in your system", "Diet"),

                    Example(
                        "What symptoms of COVID-19 are most evident", "COVID-19"),
                    Example("How to revcover from COVID-19", "COVID-19"),
                    Example(
                        "What medications to take to deal with the pandemic", "COVID-19"),
                    Example("When will the pandemic end", "COVID-19"),
                    Example("Percentage of survival from new viruse", "COVID-19"),
                    Example("Why did I lose my taste", "COVID-19"),
                    Example("Body ache", "COVID-19"),
                    Example("I have a cough. What disease is this", "COVID-19"),
                    Example("Fatigue", "COVID-19"),
                    Example("Chills", "COVID-19"),

                    Example(
                        "What medicines can I take to recover from fever", "Other"),
                    Example("What are the symptoms of ebola", "Other"),
                    Example("How can I deal with the flu", "Other"),
                    Example("How to lower blood pressure?", "Other"),
                    Example("How to get rid of hiccups?", "Other"),
                    Example("How long does the flu last?", "Other"),
                    Example("What are Low Vitamin D Symptoms", "Other")

                    ]
        
        inputs = [request.form.get("question")]
        
        response = co.classify(
            model='medium',
            inputs=inputs,
            examples=examples)

        PREDICTION = {
            'Prediction': response.classifications[0].prediction
        }

        return PREDICTION


api.add_resource(Prediction, '/prediction')


class Toxicity(Resource):
    def post(self):
        co = cohere.Client("zICvk7J5i84T4u7DvBbjy9IUnAwXJWdGc4iDrIdh")

        inputs = [request.form.get("comment")]

        response = co.classify(
            model='cohere-toxicity',
            inputs=inputs)

        PREDICTION = {
            'Prediction': response.classifications[0].prediction
        }

        return PREDICTION


api.add_resource(Toxicity, '/toxic')

if __name__ == "__main__":
    app.run()
