import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib
import plotly.graph_objects as go 
import zipfile
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="store Dashboard", page_icon=":bar_chart:",
                   layout="wide")
st.title(" :bar_chart: Sample store Dashboard")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

fl = st.file_uploader(":file_folder: Upload a file", type = (["csv", "txt", "xlsx", "xls"]))

if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename)
else:
    #os.chdir(r"D:\My_project\Freelencing\Sale_prediction\input")
    df = pd.read_csv(r"dataset.csv")
    
col1, col2 = st.columns((2))
df['date'] = pd.to_datetime(df['date'])

# Getting the min and max date from columns

startDate = pd.to_datetime(df["date"]).min()
endDate = pd.to_datetime(df["date"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))
    
with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))
    
df = df[(df['date']>= date1) & (df['date'] <= date2)].copy()

st.sidebar.header("Choose your filter: ")
# Create for Region
state = st.sidebar.multiselect("Pick your State", df["state"].unique())
if not state:
    df2 = df.copy()
else:
    df2 = df[df["state"].isin(state)]

# Create for State
outlet = st.sidebar.multiselect("Pick the State", df2["outlet"].unique())
if not outlet:
    df3 = df2.copy()
else:
    df3 = df2[df2["outlet"].isin(outlet)]

# Create for City
department_identifier = st.sidebar.multiselect("Pick the Department",df3["department_identifier"].unique())

# Filter the data based on state, outlet and department_identifier
# Filter the data based on Region, State and City

if not state and not outlet and not department_identifier:
    filtered_df = df
elif not outlet and not department_identifier:
    filtered_df = df[df["state"].isin(state)]
elif not state and not department_identifier:
    filtered_df = df[df["outlet"].isin(outlet)]
elif outlet and department_identifier:
    filtered_df = df3[df["outlet"].isin(outlet) & df3["department_identifier"].isin(department_identifier)]
elif state and department_identifier:
    filtered_df = df3[df["state"].isin(state) & df3["department_identifier"].isin(department_identifier)]
elif state and outlet:
    filtered_df = df3[df["state"].isin(state) & df3["outlet"].isin(outlet)]
elif department_identifier:
    filtered_df = df3[df3["department_identifier"].isin(department_identifier)]
else:
    filtered_df = df3[df3["state"].isin(state) & df3["outlet"].isin(outlet) & df3["department_identifier"].isin(department_identifier)]

category_df = filtered_df.groupby(by = ["category_of_product"], as_index = False)["sales"].sum()

with col1:
    st.subheader("Category wise Sales")
    fig = px.bar(category_df, x = "category_of_product", y = "sales", text = ['${:,.2f}'.format(x) for x in category_df["sales"]],
                 template = "seaborn")
    st.plotly_chart(fig,use_container_width=True, height = 200)

with col2:
    st.subheader("state wise Sales")
    fig = px.pie(filtered_df, values = "sales", names = "state", hole = 0.5)
    fig.update_traces(text = filtered_df["state"], textposition = "outside")
    st.plotly_chart(fig,use_container_width=True)
    
cl1, cl2 = st.columns((2))
with cl1:
    with st.expander("Category_ViewData"):
        st.write(category_df.style.background_gradient(cmap="Blues"))
        csv = category_df.to_csv(index = False).encode('utf-8')
        st.download_button("Download Data", data = csv, file_name = "Category.csv", mime = "text/csv",
                            help = 'Click here to download the data as a CSV file')

with cl2:
    with st.expander("State_ViewData"):
        region = filtered_df.groupby(by = "state", as_index = False)["sales"].sum()
        st.write(region.style.background_gradient(cmap="Oranges"))
        csv = region.to_csv(index = False).encode('utf-8')
        st.download_button("Download Data", data = csv, file_name = "Region.csv", mime = "text/csv",
                        help = 'Click here to download the data as a CSV file')
        

# cl1, cl2 = st.columns((2))
# with cl1:
#     year = st.selectbox("Select Year", options=filtered_df['date'].dt.year.unique())

# # Column 2: Month Selection
# with cl2:
#     month = st.selectbox("Select Month", options=filtered_df['date'].dt.strftime("%b").unique())  
    
# # Filter the DataFrame based on year and month selection
# filtered_month_year = filtered_df[
#     (filtered_df['date'].dt.year == year) &
#     (filtered_df['date'].dt.strftime("%b") == month)
# ]      
# # Plotting sales for the selected month and year
# if not filtered_month_year.empty:
#     linechart = pd.DataFrame(filtered_month_year.groupby(filtered_month_year["date"].dt.strftime("%Y : %b : %d"))["sales"].sum()).reset_index()
#     fig2 = px.line(linechart, x="date", y="sales", labels={"Sales": "Amount"}, height=500, width=1000, template="gridon")
#     st.plotly_chart(fig2, use_container_width=True)
# else:
#     st.write("No data available for the selected year and month.")
        
#filtered_df["month_year"] = filtered_df["date"].dt.to_period("M")
# Function to Perform ARIMA Forecast
def arima_forecast(data, periods=30):
    #ARIMA_model = joblib.load(r'D:\My_project\Freelencing\Sale_prediction\arima_model.pkl')
    ARIMA_model = joblib.load(r'arima_model.pkl')
    n_periods = periods
    fitted, confint = ARIMA_model.predict(n_periods=n_periods, return_conf_int=True)
    last_date = df.index[-1]
    index_of_fc = pd.date_range(last_date , periods=n_periods, freq='D')

    # index_of_fc = pd.date_range(data.index[-1] + pd.DateOffset(months=0), periods=n_periods, freq='D')
    fitted_series = pd.Series(fitted, index=index_of_fc)
    
    return fitted_series, confint

def load_sarima_model():
    # Specify the zip file and the pickled model file inside the zip
    zip_file_path = 'sarima_model.zip'
    model_file_inside_zip = 'sarima_model.pkl'

    # Open the zip file and extract the model file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extract(model_file_inside_zip, path='temp_folder')  # Extract to a temporary folder

    # Load the model using joblib
    ARIMA_model = joblib.load('temp_folder/' + model_file_inside_zip)
    os.remove('temp_folder/' + model_file_inside_zip)
    os.rmdir('temp_folder')
    return ARIMA_model
    
def sarima_forecast(data, periods=30):
    #ARIMA_model = joblib.load(r'D:\My_project\Freelencing\Sale_prediction\sarima_model.pkl')
   # ARIMA_model = joblib.load(r'sarima_model.pkl')
    ARIMA_model = load_sarima_model()
    
    n_periods = periods
    fitted, confint = ARIMA_model.predict(n_periods=n_periods, return_conf_int=True)
    last_date = df.index[-1]
    index_of_fc = pd.date_range(last_date , periods=n_periods, freq='D')

    #index_of_fc = pd.date_range(data.index[-1] + pd.DateOffset(months=0), periods=n_periods, freq='D')
    fitted_series = pd.Series(fitted, index=index_of_fc)
    
    return fitted_series, confint

st.subheader('Time Series Analysis')
#Sidebar - Model Selection
cl1, cl2 = st.columns((2))
with cl1:
    model_type = st.selectbox("Select Model", ("ARIMA", "SARIMA"))

# Column 2: Month Selection
with cl2:
    # Create a dropdown for selecting number of days
    days_options = [15, 30, 45, 60, 120, 'None']
    forecast_period = st.selectbox("Select Number of Days", days_options)

# Handling 'None' selection
if forecast_period == 'None':
    forecast_period = 30

df = filtered_df.groupby(['date'])['sales'].sum().reset_index()
df = df.set_index(['date'])

if model_type == "ARIMA":
    forecast, conf_interval = arima_forecast(df, forecast_period)
    # Convert Series to DataFrame
    forecast = forecast[1:].to_frame().reset_index()
    forecast.columns = ["date", "sales"]
    
    confidence_interval = pd.DataFrame({
    'date': forecast['date'], 
    'Predicted_sale': forecast['sales'], 
    'lower_bound': pd.DataFrame(conf_interval)[0],
    'upper_bound': pd.DataFrame(conf_interval)[1]})
    
    st.write(f"{model_type} Forecast for Next {forecast_period} Days:")
    fig2 = px.line(forecast, x = "date", y="sales", labels = {"sales": "Amount", 'date': 'Date'},height=500, width = 1000,template="gridon")
    # Adding shaded confidence interval area
    fig2.add_trace(go.Scatter(
        x=confidence_interval['date'],
        y=confidence_interval['upper_bound'],
        fill=None,
        mode='lines',
        line=dict(color='rgba(68, 68, 68, 0)'),
        showlegend=False
    ))
    fig2.add_trace(go.Scatter(
        x=confidence_interval['date'],
        y=confidence_interval['lower_bound'],
        fill='tonexty',  # Fill area between lower and upper bounds
        mode='lines',
        line=dict(color='rgba(68, 68, 68, 0.3)'),  # Adjust opacity as needed
        name='Confidence Interval'
    ))
    st.plotly_chart(fig2,use_container_width=True)
    with st.expander("View Data of TimeSeries:"):
        
        st.write(confidence_interval.T.style.background_gradient(cmap="Blues"))
        csv = confidence_interval.to_csv(index=False).encode("utf-8")
        st.download_button('Download Data', data = csv, file_name = "TimeSeries.csv", mime ='text/csv')

else:
    forecast, conf_interval = sarima_forecast(df, forecast_period)
    #Convert Series to DataFrame
    forecast = forecast[1:].to_frame().reset_index()
    forecast.columns = ["date", "sales"]
    confidence_interval = pd.DataFrame({
    'date': forecast['date'], 
    'Predicted_sale': forecast['sales'], 
    'lower_bound': pd.DataFrame(conf_interval)[0],
    'upper_bound': pd.DataFrame(conf_interval)[1]
})
    st.write(f"{model_type} Forecast for Next {forecast_period} Days:")
    fig2 = px.line(forecast, x = "date", y="sales", labels = {"sales": "Amount", 'date': 'Date'},height=500, width = 1000,template="gridon")
    fig2.add_trace(go.Scatter(
        x=confidence_interval['date'],
        y=confidence_interval['upper_bound'],
        fill=None,
        mode='lines',
        line=dict(color='rgba(68, 68, 68, 0)'),
        showlegend=False
    ))
    fig2.add_trace(go.Scatter(
        x=confidence_interval['date'],
        y=confidence_interval['lower_bound'],
        fill='tonexty',  # Fill area between lower and upper bounds
        mode='lines',
        line=dict(color='rgba(68, 68, 68, 0.3)'),  # Adjust opacity as needed
        name='Confidence Interval'
    ))
    st.plotly_chart(fig2,use_container_width=True)
    with st.expander("View Data of TimeSeries:"):
        
        st.write(confidence_interval.T.style.background_gradient(cmap="Blues"))
        csv = confidence_interval.to_csv(index=False).encode("utf-8")
        st.download_button('Download Data', data = csv, file_name = "TimeSeries.csv", mime ='text/csv')



###############################################################################3
# if model_type == "ARIMA":
#     forecast, conf_interval = arima_forecast(df, forecast_period)
#     # # Convert Series to DataFrame
#     # forecast = forecast[1:].to_frame().reset_index()
#     # forecast.columns = ["date", "sales"]
#     # st.write(f"{model_type} Forecast for Next {forecast_period} Days:")
#     # fig2 = px.line(forecast, x = "date", y="sales", labels = {"Sales": "Amount"},height=500, width = 1000,template="gridon")
#     # st.plotly_chart(fig2,use_container_width=True)
#     st.line_chart(forecast)
#     st.write("Confidence Interval:")
#     st.write(conf_interval)
# else:
#     forecast, conf_interval = sarima_forecast(df, forecast_period)
#     # Convert Series to DataFrame
#     # forecast = forecast[1:].to_frame().reset_index()
#     # forecast.columns = ["date", "sales"]
#     # st.write(f"{model_type} Forecast for Next {forecast_period} Days:")
#     # fig2 = px.line(forecast, x = "date", y="sales", labels = {"Sales": "Amount"},height=500, width = 1000,template="gridon")
#     # st.plotly_chart(fig2,use_container_width=True)
#     st.line_chart(forecast)
#     st.write("Confidence Interval:")
#     st.write(conf_interval)
#####################################################################################################3
# linechart = pd.DataFrame(filtered_df.groupby(filtered_df["date"].dt.strftime("%Y : %b : %d"))["sales"].sum()).reset_index()
# fig2 = px.line(linechart, x = "date", y="sales", labels = {"Sales": "Amount"},height=500, width = 1000,template="gridon")
# st.plotly_chart(fig2,use_container_width=True)

# with st.expander("View Data of TimeSeries:"):
#     st.write(linechart.T.style.background_gradient(cmap="Blues"))
#     csv = linechart.to_csv(index=False).encode("utf-8")
#     st.download_button('Download Data', data = csv, file_name = "TimeSeries.csv", mime ='text/csv')
    
# Create a treem based on Region, category, sub-Category
st.subheader("Hierarchical view of Sales using TreeMap")
fig3 = px.treemap(filtered_df, path = ["state","outlet","category_of_product"], values = "sales",hover_data = ["sales"],
                  color = "category_of_product")
fig3.update_layout(width = 800, height = 650)
st.plotly_chart(fig3, use_container_width=True)

