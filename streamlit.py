import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="VGA Report",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("VGA Report")


@st.cache_data
def load_data(path: str):
    data = pd.read_csv(path)
    return data

uploaded_file = st.sidebar.file_uploader("Choose a file")
st.sidebar.info("Upload a file through config", icon="📝")

if uploaded_file is None:
    uploaded_file = './cleaned_data.csv'

df = load_data(uploaded_file)

with st.expander("Data Preview"):
    st.dataframe(df)

col1, col2 = st.columns([2, 1])

def plot_categorical_feature(selected_feature):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    custom_palette = sns.color_palette("Set1")

    # Plot bar plot
    ax_barplot = axes[0]
    sns.countplot(x=selected_feature, data=df, ax=ax_barplot, palette=custom_palette)
    ax_barplot.set_title(f'Bar Plot of {selected_feature}')
    ax_barplot.set_xticklabels(ax_barplot.get_xticklabels(), rotation=45, ha='right')


    # Plot box plot
    ax_boxplot = axes[1]
    sns.boxplot(x=selected_feature, y='Price', data=df, ax=ax_boxplot, palette=custom_palette)
    ax_boxplot.set_title(f'Boxplot of {selected_feature} vs Price')
    ax_boxplot.set_xticklabels(ax_boxplot.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig


def plot_numerical_features(selected_feature):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.regplot(x=selected_feature, y='Price', data=df, ax=axes[0])
    axes[0].set_title(f'Regplot of {selected_feature} vs Price')

    sns.kdeplot(df[selected_feature]) 
    axes[1].set_title(f'Line Plot of {selected_feature} vs Price')
    plt.tight_layout()

    return fig

with col1:
    st.header("Plot numeric features")
    selected_feature = st.selectbox("Select a numeric feature:", [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != 'Price'])
    fig_numerical = plot_numerical_features(selected_feature)
    st.pyplot(fig_numerical)

    st.header("Plot categorical features:")
    selected_feature = st.selectbox("Select a categorical feature:", [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) and col != 'Name'])
    fig_boxplot = plot_categorical_feature(selected_feature)
    st.pyplot(fig_boxplot)


def plot_actual_vs_predicted():

    data = pd.read_csv('./predict.csv')

    fig = plt.figure(figsize=(8, 12))

    plt.subplot(2, 1, 1)
    sns.kdeplot(data['Price'], color='red', label='Actual Values')
    sns.kdeplot(data['LinearRegression'], color='blue', label='Predicted Values (LR)')
    plt.title('Distribution Plot - Linear Regression')
    plt.xlabel('Price')
    plt.ylabel('Density')
    plt.legend()

    # Plot actual vs predicted for Random Forest Regressor
    plt.subplot(2, 1, 2)
    sns.kdeplot(data['Price'], color='red', label='Actual Values')
    sns.kdeplot(data['RandomForestRegressor'], color='blue', label='Predicted Values (RFR)')
    plt.title('Distribution Plot - Random Forest Regressor')
    plt.xlabel('Price')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout()
    return fig

with col2:
    st.header("Regression plot of models")
    st.pyplot(plot_actual_vs_predicted())

    multi = '''Hiệu suất mô hình

    Liner Regression:
    - MSE: 6103953396994.168
    - R2: 0.788092672643712
    Random Forest Regressor:
    - MSE: 3772494540993.371 
    - R2: 0.8690325460148897 
    
    '''
    st.markdown(multi)

