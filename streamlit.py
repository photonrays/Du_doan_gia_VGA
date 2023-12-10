import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("VGA")

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

brands = ['NVIDIA', 'MSI', 'Intel']

def plot_left_most(feature):
    fig, ax = plt.subplots()
    sns.regplot(x=feature, y='Price', data=df, ax=ax)
    ax.set_title(f'Regplot of {feature} vs Price')
    st.pyplot(fig)

def plot_boxplot_and_pie_chart(selected_feature):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot pie chart
    ax_pie = axes[0]
    feature_counts = df[selected_feature].value_counts()
    filtered_counts = feature_counts[feature_counts / feature_counts.sum() >= 0.05]
    other_count = feature_counts[feature_counts / feature_counts.sum() < 0.05].sum()
    filtered_counts['Other'] = other_count
    ax_pie.pie(filtered_counts, labels=filtered_counts.index, autopct='%1.1f%%', startangle=90)
    ax_pie.axis('equal')
    ax_pie.set_title(f'Pie Chart of {selected_feature}')

    # Plot boxplot
    ax_boxplot = axes[1]
    sns.boxplot(x=selected_feature, y='Price', data=df, ax=ax_boxplot)
    ax_boxplot.set_title(f'Boxplot of {selected_feature} vs Price')
    ax_boxplot.set_xticklabels(ax_boxplot.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig

def plot_left_most(selected_feature):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.regplot(x=selected_feature, y='Price', data=df, ax=axes[0])
    axes[0].set_title(f'Regplot of {selected_feature} vs Price')

    sns.lineplot(x=selected_feature, y='Price', data=df, ax=axes[1]) 
    axes[1].set_title(f'Line Plot of AnotherNumericFeature vs Price')
    plt.tight_layout()

    return fig

st.header("Correlation of Numeral features")
selected_feature = st.selectbox("Select a numerical feature:", [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != 'Price'])
fig_numerical = plot_left_most(selected_feature)
st.pyplot(fig_numerical)

st.header("Correlation of Categorical features:")
selected_feature = st.selectbox("Select a categorical feature:", [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) and col != 'Name'])
fig_boxplot = plot_boxplot_and_pie_chart(selected_feature)
st.pyplot(fig_boxplot)
