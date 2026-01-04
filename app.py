# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Clustering Kemiskinan Provinsi",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Analisis Clustering Kemiskinan Provinsi 2020")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", 
                       ["📁 Data Overview", 
                        "🔧 Preprocessing", 
                        "📈 EDA & Outlier Detection",
                        "🤖 Clustering Analysis",
                        "📊 Visualization"])

# Load data
@st.cache_data
def load_data():
    # Create sample data based on the notebook structure
    data = {
        'Kota - Desa': ['Perkotaan', 'Pedesaan', 'Pedesaan + Perkotaan', 
                       'Perkotaan', 'Pedesaan', 'Pedesaan + Perkotaan'],
        'Persentase_Kemiskinan_Kota': [9.02, 13.83, 12.34, 9.59, 14.22, 12.76],
        'Persentase_Kemiskinan_Desa': [10.5, 15.3, 13.8, 10.8, 15.8, 14.1],
        'Garis_Kemiskinan_Kota': [500720, 433843, 453733, 504330, 437107, 457495],
        'Garis_Kemiskinan_Desa': [510000, 440000, 460000, 515000, 445000, 465000],
        'Indeks_Keparahan_Kota': [0.24, 0.49, 0.41, 0.29, 0.7, 0.57],
        'Indeks_Keparahan_Desa': [0.26, 0.52, 0.43, 0.31, 0.73, 0.6],
        'Indeks_Kedalaman_Kota': [1.23, 2.2, 1.9, 1.31, 2.48, 2.11],
        'Indeks_Kedalaman_Desa': [1.28, 2.25, 1.95, 1.35, 2.53, 2.16],
        'Jumlah_Penduduk_Miskin_Kota': [237.1, 812.22, 1049.32, 259.28, 831.86, 1091.14],
        'Jumlah_Penduduk_Miskin_Desa': [250, 850, 1100, 265, 870, 1150]
    }
    return pd.DataFrame(data)

df = load_data()

# Page 1: Data Overview
if page == "📁 Data Overview":
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Preview Data")
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        st.subheader("Informasi Dataset")
        st.write(f"**Jumlah baris:** {df.shape[0]}")
        st.write(f"**Jumlah kolom:** {df.shape[1]}")
        st.write(f"**Variabel numerik:** {len(df.select_dtypes(include=[np.number]).columns)}")
        st.write(f"**Variabel kategorikal:** {len(df.select_dtypes(include=['object']).columns)}")
    
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Missing Values")
    missing_df = pd.DataFrame({
        'Kolom': df.columns,
        'Missing Values': df.isnull().sum(),
        'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(missing_df, use_container_width=True)

# Page 2: Preprocessing
elif page == "🔧 Preprocessing":
    st.header("Data Preprocessing")
    
    # Create new feature
    st.subheader("1. Pembuatan Fitur Baru")
    st.markdown("""
    Fitur baru yang dibuat: **gap_kota_desa**
    - Menghitung selisih antara persentase kemiskinan kota dan desa
    - Indikator kesenjangan kemiskinan perkotaan-perdesaan
    """)
    
    # Calculate gap
    df['gap_kota_desa'] = df['Persentase_Kemiskinan_Kota'] - df['Persentase_Kemiskinan_Desa']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Statistik Gap Kota-Desa:**")
        st.dataframe(df['gap_kota_desa'].describe(), use_container_width=True)
    
    with col2:
        st.write("**10 Data Pertama Gap Kota-Desa:**")
        st.dataframe(df[['Kota - Desa', 'Persentase_Kemiskinan_Kota', 
                        'Persentase_Kemiskinan_Desa', 'gap_kota_desa']].head(10), 
                    use_container_width=True)
    
    # Normalization
    st.subheader("2. Normalisasi Data")
    st.markdown("""
    Menggunakan **StandardScaler** untuk normalisasi data:
    - Mengubah data ke skala yang sama (mean=0, std=1)
    - Penting untuk algoritma clustering yang berbasis jarak
    """)
    
    # Select numeric columns for normalization
    num_cols = st.multiselect(
        "Pilih kolom untuk dinormalisasi:",
        df.select_dtypes(include=[np.number]).columns.tolist(),
        default=['Persentase_Kemiskinan_Kota', 'Persentase_Kemiskinan_Desa', 'gap_kota_desa']
    )
    
    if num_cols:
        # Before normalization
        st.write("**Data Sebelum Normalisasi:**")
        st.dataframe(df[num_cols].head(), use_container_width=True)
        
        # Normalize
        scaler = StandardScaler()
        df_norm = df.copy()
        df_norm[num_cols] = scaler.fit_transform(df[num_cols])
        
        # After normalization
        st.write("**Data Setelah Normalisasi:**")
        st.dataframe(df_norm[num_cols].head(), use_container_width=True)
        
        # Show statistics
        st.write("**Statistik Data Normalisasi:**")
        st.dataframe(df_norm[num_cols].describe(), use_container_width=True)

# Page 3: EDA & Outlier Detection
elif page == "📈 EDA & Outlier Detection":
    st.header("Exploratory Data Analysis & Outlier Detection")
    
    # Select column for analysis
    selected_col = st.selectbox(
        "Pilih kolom untuk analisis:",
        df.select_dtypes(include=[np.number]).columns.tolist()
    )
    
    if selected_col:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Distribusi {selected_col}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[selected_col], kde=True, ax=ax)
            ax.set_title(f'Distribusi {selected_col}')
            st.pyplot(fig)
        
        with col2:
            st.subheader(f"Boxplot {selected_col}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(y=df[selected_col], ax=ax)
            ax.set_title(f'Boxplot {selected_col}')
            st.pyplot(fig)
        
        # Outlier detection
        st.subheader("Deteksi Outlier")
        
        # IQR Method
        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_iqr = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
        
        # Z-score Method
        z_scores = np.abs((df[selected_col] - df[selected_col].mean()) / df[selected_col].std())
        outliers_z = df[z_scores > 3]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Outlier (IQR Method)", len(outliers_iqr))
            if len(outliers_iqr) > 0:
                st.write("Contoh outlier:")
                st.dataframe(outliers_iqr[[selected_col]].head(), use_container_width=True)
        
        with col2:
            st.metric("Outlier (Z-score > 3)", len(outliers_z))
            if len(outliers_z) > 0:
                st.write("Contoh outlier:")
                st.dataframe(outliers_z[[selected_col]].head(), use_container_width=True)

# Page 4: Clustering Analysis
elif page == "🤖 Clustering Analysis":
    st.header("Clustering Analysis")
    
    # Select features for clustering
    st.subheader("1. Seleksi Fitur untuk Clustering")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    selected_features = st.multiselect(
    "Pilih fitur untuk clustering:",
    numeric_cols
)

    
    if len(selected_features) >= 2:
        X = df[selected_features].copy()
        
        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-Means Clustering
        st.subheader("2. K-Means Clustering")
        
        # Determine optimal k using elbow method
        st.write("**Menentukan jumlah cluster optimal (Elbow Method):**")
        
        inertia = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
        
        # Plot elbow curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_range, inertia, 'bo-')
        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method for Optimal k')
        ax.grid(True)
        st.pyplot(fig)
        
        # Find elbow point
        try:
            kn = KneeLocator(k_range, inertia, curve='convex', direction='decreasing')
            optimal_k = kn.knee
            st.success(f"**Jumlah cluster optimal:** {optimal_k}")
        except:
            optimal_k = st.slider("Pilih jumlah cluster:", 2, 10, 3)
        
        # Perform K-Means with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['Cluster_KMeans'] = clusters
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, clusters)
        db_index = davies_bouldin_score(X_scaled, clusters)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Silhouette Score", f"{silhouette:.3f}")
        with col2:
            st.metric("Davies-Bouldin Index", f"{db_index:.3f}")
        with col3:
            st.metric("Inertia", f"{kmeans.inertia_:.3f}")
        
        # Show clustering results
        st.subheader("3. Hasil Clustering")
        st.dataframe(df_clustered[['Kota - Desa'] + selected_features + ['Cluster_KMeans']], 
                    use_container_width=True)
        
        # Cluster distribution
        st.subheader("4. Distribusi Cluster")
        cluster_dist = df_clustered['Cluster_KMeans'].value_counts().sort_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        cluster_dist.plot(kind='bar', ax=ax1)
        ax1.set_title('Distribusi Jumlah Data per Cluster')
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Jumlah Data')
        
        # Pie chart
        ax2.pie(cluster_dist.values, labels=cluster_dist.index, autopct='%1.1f%%')
        ax2.set_title('Proporsi Cluster')
        
        st.pyplot(fig)
        
        # DBSCAN Clustering (optional)
        st.subheader("5. DBSCAN Clustering (Opsional)")
        
        if st.checkbox("Coba DBSCAN Clustering"):
            eps = st.slider("EPS (jarak maksimum antar titik):", 0.1, 5.0, 0.5, 0.1)
            min_samples = st.slider("Min Samples (titik minimum dalam cluster):", 1, 10, 3)
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan_clusters = dbscan.fit_predict(X_scaled)
            
            df_clustered['Cluster_DBSCAN'] = dbscan_clusters
            
            # Count clusters (excluding noise points labeled as -1)
            n_clusters = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
            n_noise = list(dbscan_clusters).count(-1)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Jumlah Cluster DBSCAN", n_clusters)
            with col2:
                st.metric("Noise Points", n_noise)
            
            st.dataframe(df_clustered[['Kota - Desa'] + selected_features + ['Cluster_DBSCAN']], 
                        use_container_width=True)

# Page 5: Visualization
elif page == "📊 Visualization":
    st.header("Visualization")
    
    # Select features for visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox("Pilih fitur untuk sumbu X:", numeric_cols)
    with col2:
        y_feature = st.selectbox("Pilih fitur untuk sumbu Y:", numeric_cols)
    
    # Scatter plot
    st.subheader(f"Scatter Plot: {x_feature} vs {y_feature}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df[x_feature], df[y_feature], alpha=0.6)
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(f'{x_feature} vs {y_feature}')
    ax.grid(True)
    st.pyplot(fig)
    
    # Correlation matrix
    st.subheader("Matriks Korelasi")
    
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, ax=ax)
    ax.set_title('Matriks Korelasi Antar Variabel')
    st.pyplot(fig)
    
    # Pairplot for selected features
    st.subheader("Pairplot (terbatas 5 fitur)")
    
    selected_for_pairplot = st.multiselect(
        "Pilih maksimal 5 fitur untuk pairplot:",
        numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
        max_selections=5
    )
    
    if len(selected_for_pairplot) >= 2:
        pairplot_fig = sns.pairplot(df[selected_for_pairplot], 
                                   diag_kind='kde',
                                   plot_kws={'alpha': 0.6})
        st.pyplot(pairplot_fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>© 2024 - Clustering Kemiskinan Provinsi 2020</p>
    <p><small>Dashboard dibuat dengan Streamlit • Data: Kemiskinan Provinsi 2020</small></p>
</div>
""", unsafe_allow_html=True)
