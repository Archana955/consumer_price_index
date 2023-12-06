

from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd


app=Flask(__name__)

model = pickle.load(open("lasso_regression_model.pkl","rb"))

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/submit',methods=["POST","GET"])# route to show the predictions in a web UI 
def submit():
    #  reading the inputs given by the user
    input_feature=[float(x) for x in request.form.values()]  
    #input_feature = np.transpose(input_feature)
    x=[np.array(input_feature)]
    print(input_feature)
    names = ["Milk and products","Prepared meals, snacks, sweets etc.","Health","Personal care and effects","Miscellaneous","Meat_Egg","Food_Beverages"]
    data = pd.DataFrame(x, columns=names)
    print(data)
    pred = model.predict(data)  # Assuming model.predict returns a numerical prediction

    return render_template("inner-page.html", predict=pred)


if __name__ == "__main__":
    
    app.run(debug = True,port = 2222)
