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
st.title("📊 Analisis Clustering & Perbandingan Kemiskinan Provinsi")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", 
                       ["📁 Data Overview", 
                        "🔧 Preprocessing", 
                        "📈 EDA & Outlier Detection",
                        "🤖 Clustering Analysis",
                        "📊 Visualization",
                        "🏆 Analisis Lampung vs Provinsi Lain"])

# Load data dengan data baru untuk Lampung dan provinsi lainnya
@st.cache_data
def load_data():
    # Data sampel yang lebih lengkap dengan berbagai provinsi
    data = {
        'Provinsi': ['Lampung', 'Lampung', 'Jawa Barat', 'Jawa Barat', 'Jawa Timur', 'Jawa Timur',
                    'Sumatera Utara', 'Sumatera Utara', 'Banten', 'Banten', 'DI Yogyakarta', 'DI Yogyakarta',
                    'Bali', 'Bali', 'Nusa Tenggara Timur', 'Nusa Tenggara Timur', 'Papua', 'Papua'],
        'Kota - Desa': ['Perkotaan', 'Pedesaan', 'Perkotaan', 'Pedesaan', 'Perkotaan', 'Pedesaan',
                       'Perkotaan', 'Pedesaan', 'Perkotaan', 'Pedesaan', 'Perkotaan', 'Pedesaan',
                       'Perkotaan', 'Pedesaan', 'Perkotaan', 'Pedesaan', 'Perkotaan', 'Pedesaan'],
        'Tahun': [2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020],
        'Semester': ['Semester 1', 'Semester 1', 'Semester 1', 'Semester 1', 'Semester 1', 'Semester 1',
                    'Semester 1', 'Semester 1', 'Semester 1', 'Semester 1', 'Semester 1', 'Semester 1',
                    'Semester 1', 'Semester 1', 'Semester 1', 'Semester 1', 'Semester 1', 'Semester 1'],
        'Persentase_Kemiskinan': [9.5, 15.2, 8.3, 12.1, 7.8, 11.5, 10.2, 14.8, 9.1, 13.5, 8.7, 14.2, 
                                  6.5, 9.8, 12.3, 18.7, 15.6, 23.4],
        'Garis_Kemiskinan': [485000, 420000, 520000, 445000, 510000, 435000, 475000, 410000, 530000, 
                            455000, 495000, 425000, 540000, 460000, 465000, 400000, 455000, 390000],
        'Indeks_Keparahan': [0.28, 0.52, 0.25, 0.45, 0.23, 0.42, 0.30, 0.55, 0.27, 0.49, 0.26, 0.51, 
                            0.20, 0.38, 0.35, 0.62, 0.40, 0.70],
        'Indeks_Kedalaman': [1.25, 2.15, 1.15, 1.95, 1.10, 1.85, 1.35, 2.25, 1.20, 2.05, 1.18, 2.10, 
                            0.95, 1.65, 1.50, 2.45, 1.75, 2.80],
        'Jumlah_Penduduk_Miskin': [280.5, 450.3, 1200.8, 850.2, 950.4, 680.7, 320.6, 520.4, 180.3, 
                                  290.5, 85.2, 150.8, 45.6, 75.3, 65.4, 210.7, 95.8, 350.2],
        'Populasi_Total': [8500000, 8500000, 49500000, 49500000, 40500000, 40500000, 14700000, 14700000,
                          12300000, 12300000, 3700000, 3700000, 4300000, 4300000, 5300000, 5300000,
                          4300000, 4300000]
    }
    df = pd.DataFrame(data)
    
    # Hitung tingkat kemiskinan relatif
    df['Tingkat_Kemiskinan_Relatif'] = df['Persentase_Kemiskinan'] / df['Persentase_Kemiskinan'].mean()
    df['Kesenjangan_Kota_Desa'] = df.groupby('Provinsi')['Persentase_Kemiskinan'].transform(lambda x: x.max() - x.min())
    
    return df

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
        st.write(f"**Jumlah Provinsi:** {df['Provinsi'].nunique()}")
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
    Fitur baru yang dibuat:
    1. **Tingkat_Kemiskinan_Relatif**: Perbandingan persentase kemiskinan dengan rata-rata nasional
    2. **Kesenjangan_Kota_Desa**: Selisih persentase kemiskinan kota dan desa per provinsi
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Statistik Fitur Baru:**")
        st.dataframe(df[['Tingkat_Kemiskinan_Relatif', 'Kesenjangan_Kota_Desa']].describe(), 
                    use_container_width=True)
    
    with col2:
        st.write("**Data dengan Fitur Baru:**")
        st.dataframe(df[['Provinsi', 'Kota - Desa', 'Persentase_Kemiskinan', 
                        'Tingkat_Kemiskinan_Relatif', 'Kesenjangan_Kota_Desa']].head(10), 
                    use_container_width=True)
    
    # Normalization
    st.subheader("2. Normalisasi Data")
    
    # Select numeric columns for normalization
    num_cols = st.multiselect(
        "Pilih kolom untuk dinormalisasi:",
        df.select_dtypes(include=[np.number]).columns.tolist(),
        default=['Persentase_Kemiskinan', 'Garis_Kemiskinan', 'Indeks_Keparahan']
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
        
        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_iqr = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Outlier (IQR Method)", len(outliers_iqr))
            if len(outliers_iqr) > 0:
                st.write("Outlier ditemukan:")
                st.dataframe(outliers_iqr[['Provinsi', 'Kota - Desa', selected_col]], 
                           use_container_width=True)

# Page 4: Clustering Analysis
elif page == "🤖 Clustering Analysis":
    st.header("Clustering Analysis")
    
    # Select features for clustering
    st.subheader("1. Seleksi Fitur untuk Clustering")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    selected_features = st.multiselect(
        "Pilih fitur untuk clustering:",
        numeric_cols,
        default=['Persentase_Kemiskinan', 'Indeks_Keparahan', 'Indeks_Kedalaman'] 
        if len(numeric_cols) >= 3 else numeric_cols
    )
    
    if selected_features and len(selected_features) >= 2:
        X = df[selected_features].copy()
        X = X.fillna(X.mean())
        
        if len(X) >= 2:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # K-Means Clustering
            st.subheader("2. K-Means Clustering")
            
            inertia = []
            k_range = range(2, min(11, len(X)))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)
            
            if len(inertia) >= 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(k_range, inertia, 'bo-')
                ax.set_xlabel('Number of clusters (k)')
                ax.set_ylabel('Inertia')
                ax.set_title('Elbow Method for Optimal k')
                ax.grid(True)
                st.pyplot(fig)
                
                try:
                    kn = KneeLocator(k_range, inertia, curve='convex', direction='decreasing')
                    optimal_k = kn.knee if kn.knee else 3
                    st.success(f"**Jumlah cluster optimal:** {optimal_k}")
                except:
                    optimal_k = st.slider("Pilih jumlah cluster:", 2, 10, 3)
                
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                
                df_clustered = df.copy()
                df_clustered['Cluster_KMeans'] = clusters
                
                if optimal_k > 1:
                    silhouette = silhouette_score(X_scaled, clusters)
                    db_index = davies_bouldin_score(X_scaled, clusters)
                else:
                    silhouette = 0
                    db_index = 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Silhouette Score", f"{silhouette:.3f}")
                with col2:
                    st.metric("Davies-Bouldin Index", f"{db_index:.3f}")
                with col3:
                    st.metric("Inertia", f"{kmeans.inertia_:.3f}")
                
                st.subheader("3. Hasil Clustering per Provinsi")
                cluster_summary = df_clustered.groupby(['Provinsi', 'Cluster_KMeans']).agg({
                    'Persentase_Kemiskinan': 'mean',
                    'Indeks_Keparahan': 'mean',
                    'Indeks_Kedalaman': 'mean',
                    'Kota - Desa': 'count'
                }).reset_index()
                
                st.dataframe(cluster_summary, use_container_width=True)
                
                # Highlight Lampung
                st.subheader("4. Posisi Lampung dalam Clustering")
                lampung_data = df_clustered[df_clustered['Provinsi'] == 'Lampung']
                st.dataframe(lampung_data, use_container_width=True)

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
    
    # Scatter plot with province differentiation
    st.subheader(f"Scatter Plot dengan Highlight Provinsi")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by province
    provinces = df['Provinsi'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(provinces)))
    
    for province, color in zip(provinces, colors):
        province_data = df[df['Provinsi'] == province]
        ax.scatter(province_data[x_feature], province_data[y_feature], 
                  alpha=0.7, label=province, color=color, s=100)
    
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(f'{x_feature} vs {y_feature} (dibedakan per Provinsi)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Correlation matrix
    st.subheader("Matriks Korelasi")
    
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, ax=ax)
    ax.set_title('Matriks Korelasi Antar Variabel')
    st.pyplot(fig)

# Page 6: Analisis Lampung vs Provinsi Lain
elif page == "🏆 Analisis Lampung vs Provinsi Lain":
    st.header("🏆 Analisis Komparatif: Lampung vs Provinsi Lain")
    
    # Filter data untuk Lampung
    lampung_data = df[df['Provinsi'] == 'Lampung']
    other_provinces_data = df[df['Provinsi'] != 'Lampung']
    
    st.markdown("### 1. Gambaran Umum Kemiskinan Lampung")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_poverty_lampung = lampung_data['Persentase_Kemiskinan'].mean()
        st.metric("Rata-rata % Kemiskinan Lampung", f"{avg_poverty_lampung:.1f}%")
    
    with col2:
        urban_poverty = lampung_data[lampung_data['Kota - Desa'] == 'Perkotaan']['Persentase_Kemiskinan'].values[0]
        st.metric("Kemiskinan Perkotaan", f"{urban_poverty:.1f}%")
    
    with col3:
        rural_poverty = lampung_data[lampung_data['Kota - Desa'] == 'Pedesaan']['Persentase_Kemiskinan'].values[0]
        st.metric("Kemiskinan Pedesaan", f"{rural_poverty:.1f}%")
    
    st.markdown("### 2. Peringkat Lampung diantara Provinsi Lain")
    
    # Calculate averages per province
    province_stats = df.groupby('Provinsi').agg({
        'Persentase_Kemiskinan': 'mean',
        'Garis_Kemiskinan': 'mean',
        'Indeks_Keparahan': 'mean',
        'Indeks_Kedalaman': 'mean'
    }).reset_index()
    
    # Rank provinces
    province_stats['Rank_Persentase'] = province_stats['Persentase_Kemiskinan'].rank(method='min', ascending=True)
    province_stats['Rank_Keparahan'] = province_stats['Indeks_Keparahan'].rank(method='min', ascending=True)
    province_stats['Rank_Kedalaman'] = province_stats['Indeks_Kedalaman'].rank(method='min', ascending=True)
    
    # Find Lampung's rank
    lampung_rank = province_stats[province_stats['Provinsi'] == 'Lampung']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rank_pct = int(lampung_rank['Rank_Persentase'].values[0])
        total_provinces = len(province_stats)
        st.metric("Peringkat % Kemiskinan", f"{rank_pct}/{total_provinces}")
    
    with col2:
        rank_sev = int(lampung_rank['Rank_Keparahan'].values[0])
        st.metric("Peringkat Keparahan", f"{rank_sev}/{total_provinces}")
    
    with col3:
        rank_depth = int(lampung_rank['Rank_Kedalaman'].values[0])
        st.metric("Peringkat Kedalaman", f"{rank_depth}/{total_provinces}")
    
    st.markdown("### 3. Perbandingan dengan Rata-rata Nasional")
    
    national_avg = df.groupby('Kota - Desa').agg({
        'Persentase_Kemiskinan': 'mean',
        'Garis_Kemiskinan': 'mean',
        'Indeks_Keparahan': 'mean',
        'Indeks_Kedalaman': 'mean'
    }).reset_index()
    
    comparison_data = pd.merge(lampung_data, national_avg, on='Kota - Desa', 
                              suffixes=('_Lampung', '_Nasional'))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['Persentase_Kemiskinan', 'Garis_Kemiskinan', 'Indeks_Keparahan', 'Indeks_Kedalaman']
    titles = ['Persentase Kemiskinan (%)', 'Garis Kemiskinan (Rp)', 
              'Indeks Keparahan', 'Indeks Kedalaman']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx//2, idx%2]
        
        x = np.arange(len(comparison_data))
        width = 0.35
        
        ax.bar(x - width/2, comparison_data[f'{metric}_Lampung'], width, label='Lampung', alpha=0.8)
        ax.bar(x + width/2, comparison_data[f'{metric}_Nasional'], width, label='Nasional', alpha=0.8)
        
        ax.set_xlabel('Wilayah')
        ax.set_ylabel(title)
        ax.set_title(f'Perbandingan {title}')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_data['Kota - Desa'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("### 4. Analisis Kesenjangan Kota-Desa")
    
    # Calculate urban-rural gap for each province
    gap_analysis = df.pivot_table(index='Provinsi', columns='Kota - Desa', 
                                 values='Persentase_Kemiskinan').reset_index()
    gap_analysis['Kesenjangan'] = gap_analysis['Pedesaan'] - gap_analysis['Perkotaan']
    
    # Sort by gap
    gap_analysis = gap_analysis.sort_values('Kesenjangan', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Highlight Lampung
    colors = ['red' if prov == 'Lampung' else 'skyblue' for prov in gap_analysis['Provinsi']]
    
    bars = ax.barh(gap_analysis['Provinsi'], gap_analysis['Kesenjangan'], color=colors)
    ax.set_xlabel('Kesenjangan Kemiskinan (Pedesaan - Perkotaan) dalam %')
    ax.set_title('Kesenjangan Kemiskinan Kota-Desa di Setiap Provinsi')
    ax.invert_yaxis()  # Highest at top
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center')
    
    st.pyplot(fig)
    
    st.markdown("### 5. Benchmarking dengan Provinsi Terbaik dan Terburuk")
    
    # Find best and worst provinces
    best_province = province_stats.loc[province_stats['Persentase_Kemiskinan'].idxmin()]
    worst_province = province_stats.loc[province_stats['Persentase_Kemiskinan'].idxmax()]
    
    benchmark_data = pd.DataFrame({
        'Provinsi': ['Lampung', best_province['Provinsi'], worst_province['Provinsi']],
        'Persentase_Kemiskinan': [
            lampung_rank['Persentase_Kemiskinan'].values[0],
            best_province['Persentase_Kemiskinan'],
            worst_province['Persentase_Kemiskinan']
        ],
        'Indeks_Keparahan': [
            lampung_rank['Indeks_Keparahan'].values[0],
            best_province['Indeks_Keparahan'],
            worst_province['Indeks_Keparahan']
        ],
        'Indeks_Kedalaman': [
            lampung_rank['Indeks_Kedalaman'].values[0],
            best_province['Indeks_Kedalaman'],
            worst_province['Indeks_Kedalaman']
        ]
    })
    
    st.dataframe(benchmark_data.style.highlight_min(subset=['Persentase_Kemiskinan', 
                                                           'Indeks_Keparahan', 
                                                           'Indeks_Kedalaman'], 
                                                   color='lightgreen')
                .highlight_max(subset=['Persentase_Kemiskinan', 
                                      'Indeks_Keparahan', 
                                      'Indeks_Kedalaman'], 
                              color='lightcoral'),
                use_container_width=True)
    
    st.markdown("### 6. Rekomendasi Berdasarkan Analisis")
    
    # Generate recommendations based on analysis
    recommendations = []
    
    if lampung_rank['Rank_Persentase'].values[0] > len(province_stats) / 2:
        recommendations.append("📉 **Prioritas Tinggi**: Persentase kemiskinan di atas rata-rata nasional")
    
    if gap_analysis[gap_analysis['Provinsi'] == 'Lampung']['Kesenjangan'].values[0] > gap_analysis['Kesenjangan'].median():
        recommendations.append("🏙️ **Fokus Perdesaan**: Kesenjangan kota-desa cukup tinggi, butuh intervensi khusus di pedesaan")
    
    if lampung_rank['Indeks_Keparahan'].values[0] > 0.3:
        recommendations.append("⚡ **Keparahan Tinggi**: Indeks keparahan menunjukkan kemiskinan yang mendalam")
    
    if recommendations:
        st.subheader("Rekomendasi untuk Lampung:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.success("✅ Kondisi kemiskinan di Lampung relatif baik dibanding provinsi lain")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>© 2024 - Analisis Kemiskinan Provinsi di Indonesia</p>
    <p><small>Dashboard dibuat dengan Streamlit • Data: Kemiskinan Provinsi 2020</small></p>
</div>
""", unsafe_allow_html=True)
