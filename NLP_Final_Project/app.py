from flask import Flask, render_template, request, redirect, url_for
from retrieve import load_dense_vectors, load_dataset, retrieve, generate_dense_vectors
from read import load_reader_model, answer_question
from transformers import pipeline
import os

app = Flask(__name__)

paragraphs, titles = load_dataset('data/wiki_movie_plots_deduped.csv')
transformer, tokenizer = load_reader_model('all-MiniLM-L6-v2')

dense_vectors_filename = "dense_vectors.pt"
dense_vectors = load_dense_vectors(dense_vectors_filename) if os.path.exists(dense_vectors_filename) else generate_dense_vectors(transformer, tokenizer, paragraphs)

model = pipeline('question-answering', model='deepset/roberta-base-squad2')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query")

        # Redirect to the results page with the query
        return redirect(url_for("results", query=query))

    return render_template("index.html")

@app.route("/results", methods=["GET"])
def results():
    query = request.args.get("query")
    top_k = 5 
    indices_scores = retrieve(transformer, tokenizer, query, dense_vectors, top_k)

    answers = []
    for index, score in indices_scores:
        paragraph = paragraphs[index]

        answer = answer_question(query, paragraph, model)
        # append only first 200 chars of para
        answers.append((paragraph[:200], answer, titles[index], score))

    return render_template("results.html", query=query, answers=answers)

if __name__ == "__main__":
    app.run(debug=True)
