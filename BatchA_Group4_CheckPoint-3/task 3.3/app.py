import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from model import *
import pickle


app = Flask(__name__)
model_knn = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/recommend',methods=['POST'])
def recommend():
	int_features = [float(x) for x in request.form.values()]
	final = [np.array(int_features)]
	user1 = final[0][0]
	if user1 in sample.USERID.values:
		rec= book(user1)
		index=1
		if type(rec)==list:
			print("Recomendations for the User ID ",user1,'\n')
			for i in rec:
				print(index,":",i)
				index+=1
		else:
			x=0
			for i in book_pivot.index:
				if i==rec:
					bookid_row=x
					break
				x+=1
			distances, indices = model_knn.kneighbors(book_pivot.iloc[bookid_row,:].values.reshape(1, -1),n_neighbors=11)
			k=0
			return render_template('index.html', prediction_text='Next Recommended Book: {}'.format(bookname(book_pivot.index[indices.flatten()[k]])))


if __name__ == "__main__":
    app.run(debug=True)