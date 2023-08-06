import pickle

import numpy as np
import streamlit as st

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('DF1.pkl','rb'))

st.title("Laptop Price Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size(in inches)')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu Brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))




from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Assuming you have your dataset loaded as 'df'


DF = pickle.load(open('DF (4).pkl','rb'))


# You might need to convert categorical variables to numerical values using LabelEncoder or OneHotEncoder

# Step 3: Split the dataset into features (X) and target variable (y)
X = DF[['Processor', 'Stars', 'Storage', 'RAM','CurrentPrice']]
y = DF['ProductName']

# Step 4: Scale the features (optional but recommended for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Create and fit the KNN model
k = 3  # You can adjust this value to set the number of neighbors to consider
knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
knn_model.fit(X_scaled)

# Step 6: Find the nearest close match to the input data

st.title('Laptop Recommender System')

st.write('Enter your preferences:')
processor = st.selectbox('Processor',['Intel Core i3', 'Intel Core i5', 'Intel Core i7','Intel Core i9','AMD Ryzen 3','AMD Ryzen 5','AMD Ryzen 7','AMD Ryzen 9'])
def fetch_processor2(text):
    if text =='AMD Ryzen 3':
        return 1
    elif text=='Intel Core i3':
        return 2
    elif text=='Intel Core i5' or text =='AMD Ryzen 5':
        return 3
    elif text=='Intel Core i7' or text =='AMD Ryzen 7':
        return 4
    elif text =='Intel Core i9':
        return 5
    elif text =='AMD Ryzen 9':
        return 6
PROCESSOR = fetch_processor2(processor)
stars = st.selectbox('Stars',range(1,6))
storage = st.selectbox('Storage (GB)', [256, 512, 128, 64])
ram = st.selectbox('RAM (GB)', [8, 32, 16, 4])
price = st.number_input('CurrentPrice')



input_data = [PROCESSOR,stars, storage, ram,price]

# Preprocess the input data (if needed)
input_data_scaled = scaler.transform([input_data])

# Find the nearest neighbor(s)
distances, indices = knn_model.kneighbors(input_data_scaled)

# Get the recommended row(s)
recommended_rows = DF.iloc[indices.flatten()]
print(recommended_rows)

if st.button('Recommend'):

    st.subheader('Top 3 Recommended Laptops:')
    st.write(recommended_rows)
