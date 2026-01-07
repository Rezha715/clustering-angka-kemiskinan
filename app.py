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
import os

# Set page config
st.set_page_config(
    page_title="Clustering Kemiskinan Provinsi",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Analisis Clustering Kemiskinan Provinsi 2020-2022")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", 
                       ["📁 Data Overview", 
                        "🔧 Preprocessing", 
                        "📈 EDA & Outlier Detection",
                        "🤖 Clustering Analysis",
                        "📊 Visualization"])

# Load data from CSV files
@st.cache_data
def load_data_from_csv():
    try:
        # Read CSV files
        data_2020 = pd.read_csv('data_Kemiskinan,_2020.csv')
        data_2021 = pd.read_csv('data_Kemiskinan_2021.csv')
        data_2022 = pd.read_csv('data_Kemiskinan_2022.csv')
        
        # Add year column
        data_2020['Tahun'] = 2020
        data_2021['Tahun'] = 2021
        data_2022['Tahun'] = 2022
        
        # Combine all data
        df_combined = pd.concat([data_2020, data_2021, data_2022], ignore_index=True)
        
        # Clean column names
        df_combined.columns = df_combined.columns.str.strip()
        
        # Remove empty rows if any
        df_combined = df_combined.dropna(how='all')
        
        return df_combined, data_2020, data_2021, data_2022
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        # Fallback to sample data if CSV files not found
        return load_sample_data(), None, None, None

@st.cache_data
def load_sample_data():
    # Fallback sample data (same as your original)
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
        'Jumlah_Penduduk_Miskin_Desa': [250, 850, 1100, 265, 870, 1150],
        'Tahun': [2020, 2020, 2020, 2020, 2020, 2020]
    }
    return pd.DataFrame(data)

# Load data
df_combined, df_2020, df_2021, df_2022 = load_data_from_csv()

# If CSV loading failed, use sample data
if df_2020 is None:
    df = load_sample_data()
    df_combined = df.copy()
else:
    # Process the combined data for clustering
    # We need to restructure the data for analysis
    df = df_combined.copy()
    
    # Extract relevant columns and restructure
    st.sidebar.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")

# Page 1: Data Overview
if page == "📁 Data Overview":
    st.header("Data Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Data Points", len(df_combined))
    
    with col2:
        st.metric("Years Available", len(df_combined['Tahun'].unique()))
    
    with col3:
        st.metric("Regions Types", len(df_combined['Kota - Desa'].unique()))
    
    st.subheader("Preview Data")
    
    # Show data preview with tabs for each year
    tab1, tab2, tab3, tab4 = st.tabs(["Combined Data", "2020 Data", "2021 Data", "2022 Data"])
    
    with tab1:
        st.write("**All Years Combined Data:**")
        st.dataframe(df_combined.head(), use_container_width=True)
        
    with tab2:
        if df_2020 is not None:
            st.write("**2020 Data:**")
            st.dataframe(df_2020.head(), use_container_width=True)
        else:
            st.write("2020 data not available")
    
    with tab3:
        if df_2021 is not None:
            st.write("**2021 Data:**")
            st.dataframe(df_2021.head(), use_container_width=True)
        else:
            st.write("2021 data not available")
    
    with tab4:
        if df_2022 is not None:
            st.write("**2022 Data:**")
            st.dataframe(df_2022.head(), use_container_width=True)
        else:
            st.write("2022 data not available")
    
    st.subheader("Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Structure:**")
        st.write(f"**Total rows:** {df_combined.shape[0]}")
        st.write(f"**Total columns:** {df_combined.shape[1]}")
        st.write(f"**Numeric columns:** {len(df_combined.select_dtypes(include=[np.number]).columns)}")
        st.write(f"**Categorical columns:** {len(df_combined.select_dtypes(include=['object']).columns)}")
    
    with col2:
        st.write("**Year Distribution:**")
        year_counts = df_combined['Tahun'].value_counts().sort_index()
        for year, count in year_counts.items():
            st.write(f"- **{year}:** {count} records ({count/len(df_combined)*100:.1f}%)")
        
        st.write("**Region Type Distribution:**")
        region_counts = df_combined['Kota - Desa'].value_counts()
        for region, count in region_counts.items():
            st.write(f"- **{region}:** {count} records")
    
    st.subheader("Descriptive Statistics")
    st.dataframe(df_combined.describe(), use_container_width=True)
    
    st.subheader("Missing Values")
    missing_df = pd.DataFrame({
        'Column': df_combined.columns,
        'Missing Values': df_combined.isnull().sum(),
        'Percentage': (df_combined.isnull().sum() / len(df_combined) * 100).round(2)
    })
    st.dataframe(missing_df, use_container_width=True)

# Page 2: Preprocessing
elif page == "🔧 Preprocessing":
    st.header("Data Preprocessing")
    
    # Data cleaning options
    st.subheader("1. Data Cleaning")
    
    if st.checkbox("Show raw data structure"):
        st.write("**Original Data Columns:**")
        st.write(list(df_combined.columns))
        st.write("**First few rows:**")
        st.dataframe(df_combined.head(), use_container_width=True)
    
    # Since the CSV structure is complex, let's create a simplified version for analysis
    st.subheader("2. Create Analysis Dataset")
    
    # Extract and transform data from the CSV structure
    if df_2020 is not None and df_2021 is not None and df_2022 is not None:
        # This is a simplified transformation - you may need to adjust based on your exact needs
        analysis_data = []
        
        for year, df_year in [(2020, df_2020), (2021, df_2021), (2022, df_2022)]:
            # Clean column names
            df_year.columns = [str(col).strip() for col in df_year.columns]
            
            # Extract relevant data
            for idx, row in df_year.iterrows():
                region_type = row['Kota - Desa']
                
                # Get values for each semester
                # Assuming the structure from your CSV
                if 'Garis Kemiskinan' in str(df_year.columns[1]):
                    # This is a simplified extraction - you'll need to adjust based on actual structure
                    analysis_data.append({
                        'Tahun': year,
                        'Region': region_type,
                        'Garis_Kemiskinan_S1': row.iloc[1] if len(row) > 1 else None,
                        'Garis_Kemiskinan_S2': row.iloc[2] if len(row) > 2 else None,
                        'Persentase_Kemiskinan_S1': row.iloc[6] if len(row) > 6 else None,
                        'Persentase_Kemiskinan_S2': row.iloc[7] if len(row) > 7 else None,
                        'Jumlah_Penduduk_Miskin_S1': row.iloc[9] if len(row) > 9 else None,
                        'Jumlah_Penduduk_Miskin_S2': row.iloc[10] if len(row) > 10 else None
                    })
        
        df_analysis = pd.DataFrame(analysis_data)
        
        # Clean numeric columns
        numeric_cols = ['Garis_Kemiskinan_S1', 'Garis_Kemiskinan_S2', 
                       'Persentase_Kemiskinan_S1', 'Persentase_Kemiskinan_S2',
                       'Jumlah_Penduduk_Miskin_S1', 'Jumlah_Penduduk_Miskin_S2']
        
        for col in numeric_cols:
            if col in df_analysis.columns:
                df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        
        st.write("**Analysis Dataset Created:**")
        st.dataframe(df_analysis.head(), use_container_width=True)
        st.write(f"Shape: {df_analysis.shape}")
        
        # Create new features
        st.subheader("3. Feature Engineering")
        
        if all(col in df_analysis.columns for col in ['Persentase_Kemiskinan_S1', 'Persentase_Kemiskinan_S2']):
            df_analysis['Persentase_Kemiskinan_Rata'] = (df_analysis['Persentase_Kemiskinan_S1'] + df_analysis['Persentase_Kemiskinan_S2']) / 2
            df_analysis['Perubahan_Persentase'] = df_analysis['Persentase_Kemiskinan_S2'] - df_analysis['Persentase_Kemiskinan_S1']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**New Features Statistics:**")
                st.dataframe(df_analysis[['Persentase_Kemiskinan_Rata', 'Perubahan_Persentase']].describe(), 
                           use_container_width=True)
            
            with col2:
                st.write("**Sample with New Features:**")
                st.dataframe(df_analysis[['Tahun', 'Region', 'Persentase_Kemiskinan_S1', 
                                        'Persentase_Kemiskinan_S2', 'Persentase_Kemiskinan_Rata',
                                        'Perubahan_Persentase']].head(), use_container_width=True)
        
        # Normalization
        st.subheader("4. Data Normalization")
        
        norm_cols = st.multiselect(
            "Select columns to normalize:",
            df_analysis.select_dtypes(include=[np.number]).columns.tolist(),
            default=['Persentase_Kemiskinan_Rata', 'Perubahan_Persentase'] 
            if 'Persentase_Kemiskinan_Rata' in df_analysis.columns 
            else df_analysis.select_dtypes(include=[np.number]).columns.tolist()[:2]
        )
        
        if norm_cols:
            scaler = StandardScaler()
            df_norm = df_analysis.copy()
            df_norm[norm_cols] = scaler.fit_transform(df_analysis[norm_cols])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Before Normalization:**")
                st.dataframe(df_analysis[norm_cols].head(), use_container_width=True)
            
            with col2:
                st.write("**After Normalization:**")
                st.dataframe(df_norm[norm_cols].head(), use_container_width=True)
            
            # Store normalized dataframe in session state
            st.session_state.df_norm = df_norm
            st.session_state.df_analysis = df_analysis

# Page 3: EDA & Outlier Detection
elif page == "📈 EDA & Outlier Detection":
    st.header("Exploratory Data Analysis & Outlier Detection")
    
    # Check if we have analysis data
    if 'df_analysis' in st.session_state:
        df_analysis = st.session_state.df_analysis
        
        # Select column for analysis
        selected_col = st.selectbox(
            "Pilih kolom untuk analisis:",
            df_analysis.select_dtypes(include=[np.number]).columns.tolist()
        )
        
        if selected_col:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Distribusi {selected_col}")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df_analysis[selected_col], kde=True, ax=ax)
                ax.set_title(f'Distribusi {selected_col}')
                st.pyplot(fig)
            
            with col2:
                st.subheader(f"Boxplot {selected_col}")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(y=df_analysis[selected_col], ax=ax)
                ax.set_title(f'Boxplot {selected_col}')
                st.pyplot(fig)
            
            # Time series analysis by year
            st.subheader(f"Trend {selected_col} per Tahun")
            
            if 'Tahun' in df_analysis.columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Group by year and region
                if 'Region' in df_analysis.columns:
                    pivot_data = df_analysis.pivot_table(
                        values=selected_col,
                        index='Tahun',
                        columns='Region',
                        aggfunc='mean'
                    )
                    pivot_data.plot(marker='o', ax=ax)
                else:
                    yearly_avg = df_analysis.groupby('Tahun')[selected_col].mean()
                    yearly_avg.plot(marker='o', ax=ax)
                
                ax.set_title(f'Trend {selected_col} per Tahun')
                ax.set_xlabel('Tahun')
                ax.set_ylabel(selected_col)
                ax.grid(True)
                ax.legend(title='Region')
                st.pyplot(fig)
            
            # Outlier detection
            st.subheader("Deteksi Outlier")
            
            # IQR Method
            Q1 = df_analysis[selected_col].quantile(0.25)
            Q3 = df_analysis[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_iqr = df_analysis[(df_analysis[selected_col] < lower_bound) | (df_analysis[selected_col] > upper_bound)]
            
            # Z-score Method
            z_scores = np.abs((df_analysis[selected_col] - df_analysis[selected_col].mean()) / df_analysis[selected_col].std())
            outliers_z = df_analysis[z_scores > 3]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Outlier (IQR Method)", len(outliers_iqr))
                if len(outliers_iqr) > 0:
                    st.write("Contoh outlier:")
                    st.dataframe(outliers_iqr[['Tahun', 'Region', selected_col]].head(), 
                               use_container_width=True)
            
            with col2:
                st.metric("Outlier (Z-score > 3)", len(outliers_z))
                if len(outliers_z) > 0:
                    st.write("Contoh outlier:")
                    st.dataframe(outliers_z[['Tahun', 'Region', selected_col]].head(), 
                               use_container_width=True)
    else:
        st.warning("Silakan buat dataset analisis terlebih dahulu di halaman Preprocessing.")

# Page 4: Clustering Analysis
elif page == "🤖 Clustering Analysis":
    st.header("Clustering Analysis")
    
    # Check if we have normalized data
    if 'df_norm' in st.session_state and 'df_analysis' in st.session_state:
        df_norm = st.session_state.df_norm
        df_analysis = st.session_state.df_analysis
        
        # Select features for clustering
        st.subheader("1. Seleksi Fitur untuk Clustering")
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        
        selected_features = st.multiselect(
            "Pilih fitur untuk clustering:",
            numeric_cols,
            default=['Persentase_Kemiskinan_Rata', 'Perubahan_Persentase'] 
            if 'Persentase_Kemiskinan_Rata' in numeric_cols and 'Perubahan_Persentase' in numeric_cols
            else numeric_cols[:min(2, len(numeric_cols))]
        )
        
        if selected_features and len(selected_features) >= 2:
            X = df_norm[selected_features].copy()
            
            # Check for missing values
            if X.isnull().any().any():
                st.warning("Terdapat missing values. Mengisi dengan mean...")
                X = X.fillna(X.mean())
            
            # Check if data is sufficient for clustering
            if len(X) < 2:
                st.error("Data terlalu sedikit untuk clustering. Minimal diperlukan 2 sampel.")
            else:
                # K-Means Clustering
                st.subheader("2. K-Means Clustering")
                
                # Determine optimal k using elbow method
                st.write("**Menentukan jumlah cluster optimal (Elbow Method):**")
                
                inertia = []
                k_range = range(2, min(11, len(X) + 1))
                
                for k in k_range:
                    try:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(X)
                        inertia.append(kmeans.inertia_)
                    except Exception as e:
                        st.warning(f"Error pada k={k}: {str(e)}")
                        break
                
                if len(inertia) >= 2:
                    # Plot elbow curve
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(k_range[:len(inertia)], inertia, 'bo-')
                    ax.set_xlabel('Number of clusters (k)')
                    ax.set_ylabel('Inertia')
                    ax.set_title('Elbow Method for Optimal k')
                    ax.grid(True)
                    st.pyplot(fig)
                    
                    # Find elbow point
                    try:
                        if len(inertia) >= 3:
                            kn = KneeLocator(k_range[:len(inertia)], inertia, curve='convex', direction='decreasing')
                            optimal_k = kn.knee
                            if optimal_k is not None:
                                st.success(f"**Jumlah cluster optimal:** {optimal_k}")
                                k_slider_default = optimal_k
                            else:
                                k_slider_default = min(3, len(inertia)+1)
                                optimal_k = st.slider("Pilih jumlah cluster:", 2, len(inertia)+1, k_slider_default)
                        else:
                            k_slider_default = min(3, len(inertia)+2)
                            optimal_k = st.slider("Pilih jumlah cluster:", 2, len(inertia)+2, k_slider_default)
                    except Exception as e:
                        st.warning(f"Tidak dapat menentukan elbow point: {e}")
                        optimal_k = st.slider("Pilih jumlah cluster:", 2, min(10, len(X)), 3)
                    
                    # Perform K-Means with optimal k
                    try:
                        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(X)
                        
                        # Add cluster labels to dataframe
                        df_clustered = df_analysis.copy()
                        df_clustered['Cluster_KMeans'] = clusters
                        
                        # Calculate metrics
                        if optimal_k > 1 and len(set(clusters)) > 1:
                            silhouette = silhouette_score(X, clusters)
                            db_index = davies_bouldin_score(X, clusters)
                        else:
                            silhouette = 0
                            db_index = 0
                            st.warning("Tidak dapat menghitung metrik untuk 1 cluster")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Silhouette Score", f"{silhouette:.3f}")
                        with col2:
                            st.metric("Davies-Bouldin Index", f"{db_index:.3f}")
                        with col3:
                            st.metric("Inertia", f"{kmeans.inertia_:.3f}")
                        
                        # Show clustering results
                        st.subheader("3. Hasil Clustering")
                        
                        display_cols = ['Tahun', 'Region']
                        if 'Tahun' in df_clustered.columns and 'Region' in df_clustered.columns:
                            display_cols.extend(selected_features)
                            display_cols.append('Cluster_KMeans')
                        
                        st.dataframe(df_clustered[display_cols], 
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
                        
                        # Cluster characteristics
                        st.subheader("5. Karakteristik Cluster")
                        
                        cluster_stats = df_clustered.groupby('Cluster_KMeans')[selected_features].mean()
                        st.dataframe(cluster_stats, use_container_width=True)
                        
                        # Visualization of clusters
                        if len(selected_features) >= 2:
                            st.subheader("6. Visualisasi Cluster (2D)")
                            
                            # Use first two features for 2D visualization
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], 
                                                c=clusters, cmap='viridis', 
                                                alpha=0.6, s=50)
                            
                            # Plot centroids
                            centroids = kmeans.cluster_centers_
                            ax.scatter(centroids[:, 0], centroids[:, 1],
                                      c='red', marker='X', s=200, 
                                      label='Centroids', alpha=0.8)
                            
                            ax.set_xlabel(selected_features[0])
                            ax.set_ylabel(selected_features[1])
                            ax.set_title('Cluster Visualization')
                            ax.legend()
                            ax.grid(True)
                            
                            # Add colorbar
                            plt.colorbar(scatter, ax=ax, label='Cluster')
                            
                            st.pyplot(fig)
                    
                    except Exception as e:
                        st.error(f"Error dalam K-Means clustering: {str(e)}")
                else:
                    st.error("Tidak dapat membuat elbow plot. Data mungkin terlalu sedikit.")
        else:
            st.warning("Pilih minimal 2 fitur untuk melakukan clustering.")
    else:
        st.warning("Silakan buat dataset analisis terlebih dahulu di halaman Preprocessing.")

# Page 5: Visualization
elif page == "📊 Visualization":
    st.header("Visualization")
    
    # Check if we have analysis data
    if 'df_analysis' in st.session_state:
        df_analysis = st.session_state.df_analysis
        
        st.subheader("Time Series Analysis")
        
        # Select metric for time series
        time_series_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
        
        selected_metric = st.selectbox(
            "Pilih metrik untuk analisis time series:",
            time_series_cols,
            index=0 if 'Persentase_Kemiskinan_Rata' in time_series_cols else 0
        )
        
        if selected_metric and 'Tahun' in df_analysis.columns and 'Region' in df_analysis.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Group by year and region
            pivot_data = df_analysis.pivot_table(
                values=selected_metric,
                index='Tahun',
                columns='Region',
                aggfunc='mean'
            )
            
            pivot_data.plot(marker='o', linewidth=2, ax=ax)
            
            ax.set_title(f'Trend {selected_metric} per Tahun dan Region')
            ax.set_xlabel('Tahun')
            ax.set_ylabel(selected_metric)
            ax.grid(True)
            ax.legend(title='Region Type')
            
            st.pyplot(fig)
        
        # Select features for scatter plot
        numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_feature = st.selectbox("Pilih fitur untuk sumbu X:", numeric_cols)
            with col2:
                y_feature = st.selectbox("Pilih fitur untuk sumbu Y:", numeric_cols)
            
            # Scatter plot
            st.subheader(f"Scatter Plot: {x_feature} vs {y_feature}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Color by region if available
            if 'Region' in df_analysis.columns:
                regions = df_analysis['Region'].unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(regions)))
                
                for region, color in zip(regions, colors):
                    region_data = df_analysis[df_analysis['Region'] == region]
                    ax.scatter(region_data[x_feature], region_data[y_feature], 
                              alpha=0.6, label=region, color=color, s=50)
                
                ax.legend(title='Region')
            else:
                scatter = ax.scatter(df_analysis[x_feature], df_analysis[y_feature], 
                                    alpha=0.6, s=50)
            
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title(f'{x_feature} vs {y_feature}')
            ax.grid(True)
            st.pyplot(fig)
            
            # Correlation matrix
            st.subheader("Matriks Korelasi")
            
            # Select columns for correlation matrix
            corr_cols = st.multiselect(
                "Pilih kolom untuk matriks korelasi:",
                numeric_cols,
                default=numeric_cols[:min(8, len(numeric_cols))]
            )
            
            if len(corr_cols) >= 2:
                corr_matrix = df_analysis[corr_cols].corr()
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                           square=True, ax=ax, cbar_kws={"shrink": 0.8})
                ax.set_title('Matriks Korelasi Antar Variabel')
                st.pyplot(fig)
            
            # Pairplot for selected features
            st.subheader("Pairplot (terbatas 4 fitur)")
            
            selected_for_pairplot = st.multiselect(
                "Pilih maksimal 4 fitur untuk pairplot:",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))],
                max_selections=4
            )
            
            if len(selected_for_pairplot) >= 2:
                pairplot_data = df_analysis[selected_for_pairplot]
                
                # Add region for hue if available
                if 'Region' in df_analysis.columns and len(df_analysis['Region'].unique()) <= 8:
                    pairplot_data['Region'] = df_analysis['Region']
                    hue_col = 'Region'
                else:
                    hue_col = None
                
                pairplot_fig = sns.pairplot(pairplot_data, 
                                           hue=hue_col,
                                           diag_kind='kde',
                                           plot_kws={'alpha': 0.6},
                                           palette='Set2' if hue_col else None)
                
                if hue_col:
                    pairplot_fig._legend.set_title('Region')
                
                st.pyplot(pairplot_fig)
    else:
        st.warning("Silakan buat dataset analisis terlebih dahulu di halaman Preprocessing.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>© 2024 - Clustering Kemiskinan Provinsi 2020-2022</p>
    <p><small>Dashboard dibuat dengan Streamlit • Data: Kemiskinan Provinsi 2020-2022</small></p>
</div>
""", unsafe_allow_html=True)
