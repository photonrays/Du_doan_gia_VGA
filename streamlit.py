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


def plot_categorical_feature(selected_feature):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot pie chart with all values
    ax_pie = axes[0]
    feature_counts = df[selected_feature].value_counts()
    wedges, _, _ = ax_pie.pie(feature_counts, labels=None, autopct='', startangle=90)

    # Add legend showing the pair of color, label, and percentage
    labels = [f'{label} ({percentage:.1f}%)' for label, percentage in zip(feature_counts.index, feature_counts / feature_counts.sum() * 100)]
    ax_pie.legend(wedges, labels, title="Legend", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    ax_pie.axis('equal')
    ax_pie.set_title(f'Pie Chart of {selected_feature}')

    # Plot boxplot
    ax_boxplot = axes[1]
    sns.boxplot(x=selected_feature, y='Price', data=df, ax=ax_boxplot)
    ax_boxplot.set_title(f'Boxplot of {selected_feature} vs Price')
    ax_boxplot.set_xticklabels(ax_boxplot.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig


def plot_numerical_features(selected_feature):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.regplot(x=selected_feature, y='Price', data=df, ax=axes[0])
    axes[0].set_title(f'Regplot of {selected_feature} vs Price')

    sns.lineplot(x=selected_feature, y='Price', data=df, ax=axes[1]) 
    axes[1].set_title(f'Line Plot of {selected_feature} vs Price')
    plt.tight_layout()

    return fig

st.header("Correlation of numeric features")
selected_feature = st.selectbox("Select a numeric feature:", [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != 'Price'])
fig_numerical = plot_numerical_features(selected_feature)
st.pyplot(fig_numerical)

st.header("Correlation of Categorical features:")
selected_feature = st.selectbox("Select a categorical feature:", [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) and col != 'Name'])
fig_boxplot = plot_categorical_feature(selected_feature)
st.pyplot(fig_boxplot)
