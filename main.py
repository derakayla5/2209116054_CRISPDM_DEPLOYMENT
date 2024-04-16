import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi untuk menampilkan visualisasi korelasi matriks
def show_correlation_heatmap(df, columns):
    if len(columns) < 2:
        st.warning("Pilih setidaknya dua kolom untuk menampilkan korelasi.")
        return
    
    selected_corr = df[columns].corr()
    fig = px.imshow(selected_corr)
    st.plotly_chart(fig)

# Fungsi untuk menampilkan distribusi data
def show_distribution(data, column):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data[column], kde=True, color='skyblue', bins=20, ax=ax)
    ax.set_title(f'Distribusi {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)
    st.write("Distribusi data di atas menggambarkan persebaran nilai pada kolom " + column + " dalam bentuk histogram. Histogram ini menunjukkan frekuensi kemunculan nilai-nilai tersebut.")

# Fungsi untuk menampilkan box plot
def show_boxplot(data, x_column, y_column):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=data[x_column], y=data[y_column], ax=ax)
    ax.set_title(f'Box Plot: {y_column} per {x_column}')
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    st.pyplot(fig)
    st.write("Box plot di atas menunjukkan distribusi nilai pada kolom " + y_column + " berdasarkan kategori pada kolom " + x_column + ". Garis di dalam box merupakan median, sedangkan box menunjukkan interquartile range (IQR) dari data.")

# Fungsi untuk menampilkan scatter plot
def show_scatterplot(data, x_column, y_column):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=data, x=x_column, y=y_column, ax=ax)
    ax.set_title(f'Scatter Plot: {x_column} vs {y_column}')
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    st.pyplot(fig)
    st.write("Scatter plot di atas menunjukkan hubungan antara nilai pada kolom " + x_column + " dengan nilai pada kolom " + y_column + ". Setiap titik merepresentasikan satu data, dan posisi titik menunjukkan nilai pada kedua kolom tersebut.")

# Fungsi untuk menampilkan hasil prediksi
def show_prediction_results(data, model_name):
    if 'Predicted Result' not in data.columns:
        st.error("Kolom 'Predicted Result' tidak ditemukan.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Predicted Result', data=data, palette='Set2', ax=ax)
    ax.set_title(f'Hasil Prediksi untuk Model {model_name}')
    ax.set_xlabel('Hasil Prediksi')
    ax.set_ylabel('Jumlah')
    st.pyplot(fig)
    st.write("Grafik di atas menunjukkan hasil prediksi dari model " + model_name + ". Hasil prediksi ini menunjukkan jumlah data yang diprediksi masuk ke dalam kelas tertentu.")

# Fungsi untuk menampilkan dataframe
def show_dataframe(data):
    st.write(data)

# Memuat DataFrame dari file CSV
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

# Memanggil fungsi untuk memuat data
df = load_data('Student_dataset.csv')

# Menampilkan judul Streamlit
st.title("Prediksi Nilai Berdasarkan Nilai Sebelumnya")

# Menampilkan visualisasi diagram batang untuk kolom G1, G2, dan G3
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Diagram batang untuk kolom G1
g1_counts = df['G1'].value_counts()
g1_counts.plot(kind='bar', ax=ax[0], color='skyblue')
ax[0].set_title('Diagram Batang G1')
ax[0].set_xlabel('G1')
ax[0].set_ylabel('Frekuensi')

# Diagram batang untuk kolom G2
g2_counts = df['G2'].value_counts()
g2_counts.plot(kind='bar', ax=ax[1], color='green')
ax[1].set_title('Diagram Batang G2')
ax[1].set_xlabel('G2')
ax[1].set_ylabel('Frekuensi')

# Diagram batang untuk kolom G3
g3_counts = df['G3'].value_counts()
g3_counts.plot(kind='bar', ax=ax[2], color='orange')
ax[2].set_title('Diagram Batang G3')
ax[2].set_xlabel('G3')
ax[2].set_ylabel('Frekuensi')

st.pyplot(fig)
st.write("Diagram batang di atas menunjukkan distribusi nilai pada kolom G1, G2, dan G3.")

# Menampilkan dataframe dengan seluruh kolom
st.title("DataFrame")
show_dataframe(df)

# Pilih kolom-kolom yang ingin ditampilkan di sidebar
selected_columns = st.sidebar.multiselect('Pilih Kolom untuk Korelasi', df.columns.tolist())

# Tampilkan visualisasi korelasi matriks di Streamlit (di dashboard utama)
st.title("Visualisasi Korelasi Antar Kolom")
st.markdown("Visualisasi di bawah ini menunjukkan korelasi antara kolom yang dipilih. Pada visualisasi ini menampilkan seberapa berpengaruh dan seberapa tinggi korelasi antar kolom yang ada")
show_correlation_heatmap(df, selected_columns)

# Menampilkan distribusi dan scatter plot untuk kolom yang dipilih
if len(selected_columns) >= 2:
    for column in selected_columns:
        st.title(f'Visualisasi untuk Kolom: {column}')
        st.markdown("Pilih tipe visualisasi dari dropdown di sidebar untuk menampilkan visualisasi yang diinginkan.")
        selected_chart = st.sidebar.selectbox(f'Pilih Tipe Visualisasi ({column})', ['Distribusi', 'Box Plot', 'Scatter Plot'], key=f'{column}_visualization')  # Tambahkan key yang unik
        
        if selected_chart == 'Distribusi':
            show_distribution(df, column)
        elif selected_chart == 'Box Plot':
            selected_x_column = st.sidebar.selectbox('Pilih Kolom untuk sumbu X', selected_columns, key=f'{column}_x')  # Tambahkan key yang unik
            show_boxplot(df, selected_x_column, column)
        elif selected_chart == 'Scatter Plot':
            selected_y_column = st.sidebar.selectbox('Pilih Kolom untuk sumbu Y', selected_columns, key=f'{column}_y')
            show_scatterplot(df, selected_y_column, column)
