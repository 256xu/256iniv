from flask import Flask
from flask import render_template, request, redirect, url_for, flash, make_response
import pandas as pd
import common
import kb
import cb
import hs
from flask import request, url_for, redirect, flash


app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template('homepage.html')

@app.route('/search', methods=['POST', 'GET'])
def search():
    if request.method == 'POST':
        inputs = _get_form_fields()
        print({"inputs": inputs})
        result = kb.inter_set(
            provider=inputs[common.PROVIDER],
            skills=inputs[common.SKILLS],
            subtitles=inputs[common.SUBTITLES],
            overall=inputs[common.OVERALL],
            enrolled=inputs[common.ENROLLED],
            topic=inputs[common.TOPIC],
            sentence=inputs[common.SENTENCE])
        result = result.values.tolist()
        print("\n hybrid result {0}\n\n".format(result))
        headline = "Recommended courses based on your needs by knowledge-based model"
        return render_template('search.html', items=result, page_headline=headline)
    else:
        return redirect(url_for('homepage'))

@app.route('/search2', methods=['POST', 'GET'])
def search2():
    if request.method == 'POST':
        inputs = _get_form_fields()
        print({"inputs": inputs})
        result = hs.kb_combined(
            provider=inputs[common.PROVIDER],
            skills=inputs[common.SKILLS],
            subtitles=inputs[common.SUBTITLES],
            overall=inputs[common.OVERALL],
            enrolled=inputs[common.ENROLLED],
            topic=inputs[common.TOPIC],
            sentence=inputs[common.SENTENCE])
        result = result.values.tolist()
        print("\n hybrid result {0}\n\n".format(result))
        headline = "Recommended courses based on your needs and contents similarity!"
        return render_template('search2.html', items=result, page_headline=headline)
    else:
        return redirect(url_for('homepage'))

@app.route('/similar', methods=['GET'])
def recommend_similar_patents():
    search_table = pd.read_csv('cleaned_data.csv')
    index = request.args.get('index')
    course_name = search_table['name'][int(index)]
    if index:
        result = cb.tfidf_sim(index)
        print("Content Based Recom: {0}".format(result))
        if len(result) > 0:
            result = result.values.tolist()
        page_headline = "Similar courses for  {0} based on Contents ".format(course_name)
        return render_template('search.html', items=result,
                               page_headline=page_headline)
    return redirect(url_for('homepage'))

def _get_form_fields():
    inputs = {}
    for key in common.SEARCH_FORM_FIELDS:
        if request.form.get(key):
            inputs[key] = request.form.get(key)
        else:
            inputs[key] = None
    return inputs

if __name__ == '__main__':
    app.run(port = 3334, debug=True)