import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error 
from sklearn.linear_model import LinearRegression

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
modelTraining = st.beta_container()

st.markdown("""
	<style>
	.main{
	background-color: #f5f5f5;
	}
	</style>
	""",
	unsafe_allow_html=True
)

@st.cache
def get_data(filename):
	data = pd.read_csv(filename)
	return data

with header:
	st.title('Data science Project:')
	st.header('Predict House Price in La Jolla, San Diego, CA')
	st.text('ML multivariate linear regression  is used to predict La Jolla house price in Feb 2021')

with dataset:
	st.header('La Jolla house price dataset')
	st.text('I collect data for this project from online sites like Zillow, Redfin, Realtor, etc.')
	data = get_data('LaJolla-02-2021.csv')
	# get NaN value
	data = data.apply (pd.to_numeric, errors='coerce')
	# drop rows with NaN
	data = data.dropna()
	# reset index
	data = data.reset_index(drop=True)

	st.write(data.head())
	st.subheader('Number of bedrooms distribution:')
	price_list = pd.DataFrame(data['beds'].value_counts()).head(50)
	st.bar_chart(price_list)
	x = data['area']
	y = data['price']
	fig, ax = plt.subplots()
	ax.scatter(x,y)
	ax.set_xlabel('Area in sq ft')
	ax.set_ylabel('Price in 10 millions')
	st.subheader('House price vs. Area:')
	st.pyplot(fig)
	
with features:
	st.header('The ML model: Multivariate Linear Regression')
	st.markdown('* **Four features:** House area, number of bedrooms, number of full bathrooms, number of half bathrooms')

with modelTraining:
	st.header('Estimate the price for a house with the following selected features:')
	para_col,disp_col = st.beta_columns(2)
	n_area = para_col.slider('Area of the house (sq ft):',min_value=600,max_value=19000, value=2000)
	n_beds = para_col.selectbox('Number of bedrooms:',options=[1,2,3,4,5,6,7,8],index=2)
	n_baths = para_col.selectbox('Number of full bathrooms:',options=[1,2,3,4,5,6,7,8],index=2)
	n_half = para_col.selectbox('Number of 1.5-bathrooms:',options=[0,1,2,3,4],index=0)
	
	X=data[['area','beds','full_baths','1.5_baths']]
	y=data['price']
	# training set size = 80%  test size =20%
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

	lr = LinearRegression() 
	lr.fit(X_train,y_train)
	predictions = lr.predict(X_test)
	score = r2_score(y_test,predictions)

	disp_col.subheader('The price of the house:')
	disp_col.write(lr.predict([[n_area,n_beds,n_baths,n_half]]))
	disp_col.subheader('R squared score of the model:')
	disp_col.write(score)
	#disp_col.subheader('Mean squared error of the model:')
	#disp_col.write(mean_squared_error(y_test,predictions))
	#disp_col.subheader('Mean absolute error of the model:')
	#disp_col.write(mean_absolute_error(y_test,predictions))
