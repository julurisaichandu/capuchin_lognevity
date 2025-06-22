import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Capuchin Health Dashboard",
    page_icon="🐒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Reference Ranges and Health Logic
# ==============================================================================
# Based on literature values for adult capuchin monkeys
REFERENCE_RANGES = {
    'RBC': {'min': 4.5, 'max': 7.0, 'units': '10^6/uL'},
    'Hematocrit': {'min': 35.0, 'max': 50.0, 'units': '%'},
    'Hemoglobin': {'min': 12.0, 'max': 17.0, 'units': 'g/dL'},
    'MCV': {'min': 65.0, 'max': 85.0, 'units': 'fL'},
    'MCH': {'min': 20.0, 'max': 28.0, 'units': 'pg'},
    'MCHC': {'min': 30.0, 'max': 36.0, 'units': 'g/dL'},
    'WBC': {'min': 3.5, 'max': 15.0, 'units': '10^3/uL'},
    'Neutrophils (Absolute)': {'min': 1.5, 'max': 8.0, 'units': '10^3/uL'},
    'Lymphocytes (Absolute)': {'min': 0.8, 'max': 4.0, 'units': '10^3/uL'},
    'Monocytes (Absolute)': {'min': 0.1, 'max': 0.8, 'units': '10^3/uL'},
    'Eosinophils (Absolute)': {'min': 0.0, 'max': 0.6, 'units': '10^3/uL'},
    'Basophils (Absolute)': {'min': 0.0, 'max': 0.2, 'units': '10^3/uL'},
    'Platelets': {'min': 200.0, 'max': 600.0, 'units': '10^3/uL'},
    'Glucose': {'min': 60.0, 'max': 140.0, 'units': 'mg/dL'},
    'BUN': {'min': 8.0, 'max': 30.0, 'units': 'mg/dL'},
    'Creatinine': {'min': 0.4, 'max': 1.0, 'units': 'mg/dL'},
    'Phosphorus': {'min': 2.5, 'max': 6.0, 'units': 'mg/dL'},
    'Calcium': {'min': 8.0, 'max': 11.0, 'units': 'mg/dL'},
    'Sodium': {'min': 140.0, 'max': 155.0, 'units': 'mmol/L'},
    'Potassium': {'min': 3.0, 'max': 5.0, 'units': 'mmol/L'},
    'Chloride': {'min': 100.0, 'max': 120.0, 'units': 'mmol/L'},
    'Total Protein': {'min': 5.5, 'max': 8.0, 'units': 'g/dL'},
    'Albumin': {'min': 3.0, 'max': 5.0, 'units': 'g/dL'},
    'Globulin': {'min': 1.5, 'max': 3.5, 'units': 'g/dL'},
    'ALT': {'min': 10.0, 'max': 50.0, 'units': 'U/L'},
    'AST': {'min': 10.0, 'max': 50.0, 'units': 'U/L'},
    'ALP': {'min': 30.0, 'max': 120.0, 'units': 'U/L'},
    'GGT': {'min': 10.0, 'max': 70.0, 'units': 'U/L'},
    'Total Bilirubin': {'min': 0.0, 'max': 0.3, 'units': 'mg/dL'},
    'Cholesterol': {'min': 100.0, 'max': 250.0, 'units': 'mg/dL'}
}

def get_reference_range(parameter):
    """Gets the reference range for a parameter."""
    return REFERENCE_RANGES.get(parameter)

def get_deviation(value, parameter):
    """Gets the deviation status (High, Low, Normal) for a value."""
    range_info = get_reference_range(parameter)
    if range_info is None or pd.isna(value):
        return ''
    
    if value < range_info['min']:
        return 'Low'
    elif value > range_info['max']:
        return 'High'
    else:
        return 'Normal'

def calculate_anomaly_severity(value, parameter):
    """Calculates a severity score (0-4) for an anomalous value."""
    range_info = get_reference_range(parameter)
    if range_info is None or pd.isna(value):
        return 0
    
    range_size = range_info['max'] - range_info['min']
    if range_size <= 0: return 0

    if value < range_info['min']:
        deviation = (range_info['min'] - value) / range_size
    elif value > range_info['max']:
        deviation = (value - range_info['max']) / range_size
    else:
        return 0

    if deviation <= 0.1: return 1
    if deviation <= 0.25: return 2
    if deviation <= 0.5: return 3
    return 4

def calculate_health_score(df, animal_id):
    """Calculates a health score from 0-100 based on the most recent lab results."""
    animal_data = df[df['id'] == animal_id]
    if animal_data.empty:
        return "N/A"
        
    most_recent_date = animal_data['date'].max()
    recent_data = animal_data[animal_data['date'] == most_recent_date]
    
    total_tests = 0
    total_severity = 0
    
    for _, row in recent_data.iterrows():
        if row['test'] in REFERENCE_RANGES:
            total_tests += 1
            severity = calculate_anomaly_severity(row['result'], row['test'])
            total_severity += severity
    
    if total_tests == 0:
        return "N/A"
        
    max_possible_severity = 4 * total_tests
    if max_possible_severity == 0: return "100.0 / 100"
    
    health_score = 100 * (1 - (total_severity / max_possible_severity))
    
    return f"{health_score:.1f} / 100"

# ==============================================================================
# Data Loading and Preprocessing
# ==============================================================================

@st.cache_data
def load_and_process_data(uploaded_files):
    """Loads multiple CSVs, assigns IDs, preprocesses, and merges them."""
    if not uploaded_files:
        return None
        
    all_dfs = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            try:
                # Extract animal ID from filename
                animal_id = os.path.basename(uploaded_file.name).split(' ')[0].capitalize()
                
                df = pd.read_csv(uploaded_file)
                df.columns = df.columns.str.strip()
                df['id'] = animal_id
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                # Convert result to numeric, coercing errors
                df['result'] = pd.to_numeric(df['result'], errors='coerce')
                
                # Drop rows where essential data is missing
                df.dropna(subset=['date', 'test', 'result'], inplace=True)
                
                all_dfs.append(df)
            except Exception as e:
                st.sidebar.error(f"Failed to process {uploaded_file.name}: {e}")
                continue

    if not all_dfs:
        return None
        
    # Merge all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.sort_values(by=['date', 'test'], inplace=True)
    
    return combined_df

# ==============================================================================
# Plotting and Analysis Functions
# ==============================================================================

def plot_time_series(df, animal_id, parameter):
    animal_data = df[(df['id'] == animal_id) & (df['test'] == parameter)].sort_values('date')
    if animal_data.empty:
        return None
        
    ref_range = get_reference_range(parameter)
    fig = go.Figure()

    # Add reference range as a shaded area
    if ref_range:
        fig.add_shape(
            type="rect",
            x0=animal_data['date'].min(), x1=animal_data['date'].max(),
            y0=ref_range['min'], y1=ref_range['max'],
            fillcolor="rgba(0,255,0,0.15)", line_width=0, layer="below"
        )
    
    # Add the time series line
    fig.add_trace(go.Scatter(
        x=animal_data['date'], y=animal_data['result'],
        mode='lines+markers', name=parameter, line=dict(color='royalblue', width=2)
    ))
    
    fig.update_layout(
        title=f"{parameter} Trend for {animal_id}",
        xaxis_title="Date",
        yaxis_title=f"{parameter} ({ref_range['units']})" if ref_range else parameter,
        template="plotly_white", showlegend=False
    )
    return fig

def plot_health_radar(df, animal_id):
    animal_data = df[df['id'] == animal_id]
    if animal_data.empty: return None

    most_recent_date = animal_data['date'].max()
    recent_data = animal_data[animal_data['date'] == most_recent_date]

    radar_values, labels = [], []
    for _, row in recent_data.iterrows():
        param = row['test']
        ref_range = get_reference_range(param)
        if ref_range:
            labels.append(param)
            # Normalize the value
            value = row['result']
            min_val, max_val = ref_range['min'], ref_range['max']
            range_size = max_val - min_val if max_val > min_val else 1
            normalized = (value - min_val) / range_size if pd.notna(value) else 0.5
            radar_values.append(normalized)

    fig = go.Figure()
    # Reference range (0 to 1 represents the normal range)
    fig.add_trace(go.Scatterpolar(
        r=[1] * len(labels), theta=labels, fill='toself',
        fillcolor='rgba(0,255,0,0.1)', line=dict(color='rgba(0,255,0,0.5)'), name='Normal Range High'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[0] * len(labels), theta=labels, fill='toself',
        fillcolor='white', line=dict(color='rgba(0,255,0,0.5)'), name='Normal Range Low'
    ))
    # Animal data
    fig.add_trace(go.Scatterpolar(
        r=radar_values, theta=labels, fill='toself',
        line=dict(color='royalblue'), name=animal_id
    ))
    fig.update_layout(
        title=f"Health Snapshot for {animal_id} ({most_recent_date.date()})",
        polar=dict(radialaxis=dict(visible=True, range=[-0.2, 1.2])), showlegend=False
    )
    return fig

def get_most_recent_data(df):
    """Get the most recent test results for each animal."""
    return df.sort_values('date').groupby(['id', 'test']).last().reset_index()

def perform_pca(df):
    """Pivot data, handle missing values, scale, and perform PCA."""
    # This pivots the data: animals as rows, tests as columns, results as values
    pivot_df = df.pivot_table(index='id', columns='test', values='result').reset_index()
    
    # Get only the numeric columns for analysis
    numeric_cols = pivot_df.select_dtypes(include=np.number).columns.tolist()
    
    # Check if we have enough data to perform PCA
    # We need at least 2 animals (samples) and at least 2 tests (features)
    if pivot_df.shape[0] < 2 or len(numeric_cols) < 2:
        st.warning("PCA requires at least 2 animals and 2 numeric test parameters with recent data. Not enough data to proceed.")
        return None, None, None, None

    # Handle missing values by filling with the mean of the column
    analysis_df = pivot_df[numeric_cols].fillna(pivot_df[numeric_cols].mean())
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(analysis_df)
    
    # Apply PCA
    # if n_components > min(n_samples, n_features)
    n_components = 2
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['id'] = pivot_df['id'].values
    
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=numeric_cols)
    
    # Return the analysis_df along with other results
    return pca_df, loadings, pca.explained_variance_ratio_, analysis_df

# ==============================================================================
# Main Dashboard UI
# ==============================================================================
def create_dashboard():
    st.title("Capuchin Monkey Health Dashboard")
    
    # --- Sidebar ---
    st.sidebar.header("Dashboard Controls")
    st.sidebar.info("Upload one or more cleaned CSV files. Animal names will be extracted from filenames.")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload cleaned capuchin data files", 
        type="csv", 
        accept_multiple_files=True,
        key="file_uploader"
    )

    if 'data' not in st.session_state or uploaded_files:
        st.session_state.data = load_and_process_data(uploaded_files)

    if st.session_state.data is None:
        st.warning("Please upload at least one cleaned CSV file to begin analysis.")
        st.stop()

    data = st.session_state.data
    animal_options = sorted(data['id'].unique().tolist())
    all_tests = sorted(data['test'].unique().tolist())
    
    # --- Main Content ---
    tab1, tab2, tab3 = st.tabs([
        "🐒 Individual Analysis", 
        "📊 Population Overview", 
        "🔍 Pattern Detection"
    ])

    # =================== Individual Analysis Tab ===================
    with tab1:
        st.header("Individual Animal Analysis")
        selected_animal = st.selectbox("Select Animal", animal_options, key="individual_select")
        
        if selected_animal:
            animal_data = data[data['id'] == selected_animal]
            most_recent_date = animal_data['date'].max()
            
            # Display basic info and health score
            col1, col2, col3 = st.columns(3)
            col1.metric("Animal ID", selected_animal)
            col2.metric("Most Recent Data", f"{most_recent_date.date()}")
            col3.metric("Overall Health Score", calculate_health_score(data, selected_animal), 
                         help="A score from 0-100 indicating overall health based on the number and severity of anomalies in the latest lab results. Higher is better.")
            
            # Display Most Recent Test Results
            st.subheader(f"Most Recent Test Results ({most_recent_date.date()})")
            recent_data = animal_data[animal_data['date'] == most_recent_date].copy()
            recent_data['Status'] = recent_data.apply(lambda row: get_deviation(row['result'], row['test']), axis=1)
            recent_data['Ref. Range'] = recent_data['test'].apply(lambda x: f"{get_reference_range(x)['min']:.1f} - {get_reference_range(x)['max']:.1f}" if get_reference_range(x) else 'N/A')
            
            def highlight_status(val):
                color = {'High': 'rgba(255, 75, 75, 0.3)', 'Low': 'rgba(75, 75, 255, 0.3)'}.get(val, '')
                return f'background-color: {color}'
            
            st.dataframe(
                recent_data[['test', 'result', 'units', 'Ref. Range', 'Status']].style.map(highlight_status, subset=['Status']),
                use_container_width=True
            )   
            
            # --- Time Series Plots ---
            st.subheader("Parameter Trends Over Time")
            selected_params = st.multiselect("Select Parameters to Plot", all_tests, default=all_tests[:min(3, len(all_tests))], key="param_select")
            
            with st.expander("What do these charts show?"):
                st.info("""
                    These charts track selected health parameters over time for an individual capuchin.
                    - **Blue Line:** The animal's test results.
                    - **Green Shaded Area:** The standard reference range for a healthy adult capuchin.
                    
                    **How to use it:** Look for trends where the blue line consistently stays above or below the green area. This can indicate chronic health issues. Sudden spikes or drops can point to acute events.
                """)
            
            if selected_params:
                for param in selected_params:
                    fig = plot_time_series(data, selected_animal, param)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select one or more parameters to display trends.")

            # --- Health Radar Chart ---
            st.subheader("Health Snapshot Radar")
            with st.expander("Understanding the Health Radar"):
                st.info("""
                    This chart provides a holistic snapshot of the animal's most recent health status across key parameters.
                    - **Values are normalized:** A result in the middle of the healthy reference range is at the center of the green band. The inner and outer edges of the green shaded area represent the minimum and maximum of the healthy range.
                    - **Blue Shape:** The animal's current health profile. A balanced profile will be a relatively smooth shape centered within the green area.
                    - **Spikes:** Points extending far outside the green area highlight significant abnormalities that may require attention.
                """)
            radar_fig = plot_health_radar(data, selected_animal)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)

    # =================== Population Overview Tab ===================
    with tab2:
        st.header("Population Health Overview")
        
        # --- Anomaly Hotspot ---
        st.subheader("Anomaly Hotspot")
        with st.expander("About this chart"):
            st.info("This heatmap highlights which animals have the most severe or frequent anomalies for each health parameter, based on their most recent test results. Darker cells indicate a more significant issue.")

        recent_data = get_most_recent_data(data)
        recent_data['severity'] = recent_data.apply(lambda row: calculate_anomaly_severity(row['result'], row['test']), axis=1)
        
        anomaly_pivot = recent_data.pivot_table(index='id', columns='test', values='severity', fill_value=0)
        
        # Keep only columns with at least one anomaly
        anomaly_pivot = anomaly_pivot.loc[:, (anomaly_pivot != 0).any(axis=0)]

        if not anomaly_pivot.empty:
            fig = px.imshow(anomaly_pivot,
                            labels=dict(x="Health Parameter", y="Animal ID", color="Severity Score"),
                            x=anomaly_pivot.columns, y=anomaly_pivot.index,
                            color_continuous_scale='Reds',
                            title="Anomaly Severity Heatmap (Most Recent Data)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No anomalies detected in the most recent data for the population.")
        
        # --- Parameter Distribution ---
        st.subheader("Parameter Distribution Across Population")
        param_to_dist = st.selectbox("Select Parameter to View Distribution", all_tests, key="dist_select")
        
        if param_to_dist:
            param_data = data[data['test'] == param_to_dist]
            fig = px.box(param_data, x='id', y='result', title=f"Distribution of {param_to_dist} Across All Animals")
            
            ref_range = get_reference_range(param_to_dist)
            if ref_range:
                fig.add_hline(y=ref_range['min'], line_dash="dash", line_color="green", annotation_text="Normal Min")
                fig.add_hline(y=ref_range['max'], line_dash="dash", line_color="green", annotation_text="Normal Max")
                
            st.plotly_chart(fig, use_container_width=True)
    
    # =================== Pattern Detection Tab ===================
    with tab3:
        st.header("Pattern Detection & Clustering")
        
        if data['id'].nunique() < 2:
            st.info("Pattern detection requires data from at least 3 different animals.")
        else:
            recent_data = get_most_recent_data(data)
            pca_df, loadings, explained_variance, analysis_df_numeric = perform_pca(recent_data)
            
            if pca_df is not None:
                # --- PCA ---
                st.subheader("Principal Component Analysis (PCA)")
                with st.expander("How to Interpret the PCA Plot"):
                    st.info("""
                        This plot simplifies the complex health data, showing each animal's overall profile in two dimensions.
                        - **Animals that are close together** have similar overall health profiles.
                        - **Animals that are far apart** have different health profiles.
                        - The chart helps visualize natural groupings without pre-defining them.
                    """)
                fig_pca = px.scatter(pca_df, x='PC1', y='PC2', text='id', title="PCA of Capuchin Health Profiles",
                                 labels={'PC1': f'Principal Component 1 ({explained_variance[0]:.1%})',
                                         'PC2': f'Principal Component 2 ({explained_variance[1]:.1%})'})
                fig_pca.update_traces(textposition='top center')
                st.plotly_chart(fig_pca, use_container_width=True)
                
                # --- Clustering ---
                st.subheader("Hierarchical Clustering")
                with st.expander("Understanding the Dendrogram"):
                    st.info("""
                        This dendrogram visually represents the health similarities between animals.
                        - Animals joined by a 'U' shape lower down on the chart are more similar.
                        - You can visually identify clusters by looking for groups that are joined at a higher level.
                        - The number of clusters slider below allows you to formally group them based on this structure.
                    """)
                
                # Perform clustering on the PCA results for stability
                Z = linkage(pca_df[['PC1', 'PC2']], method='ward')
                plt.figure(figsize=(12, 6))
                dendrogram(Z, labels=pca_df['id'].values, leaf_rotation=90)
                plt.title('Health Profile Dendrogram')
                plt.ylabel('Distance (Dissimilarity)')
                st.pyplot(plt)
                
                # --- Cluster Analysis ---
                st.subheader("Cluster Analysis")
                n_clusters = st.slider("Select Number of Clusters to Analyze", 2, min(10, data['id'].nunique()-1), 3)
                
                clusters = fcluster(Z, n_clusters, criterion='maxclust')
                
                # Create the final analysis_df here
                analysis_df = analysis_df_numeric.copy()
                analysis_df['id'] = pca_df['id'] 
                analysis_df['cluster'] = clusters
                
                # Display cluster contents
                st.write("Cluster Membership:")
                st.dataframe(analysis_df[['id', 'cluster']].sort_values('cluster'))

                # Display cluster profiles
                st.write("What defines each cluster?")
                cluster_profiles = analysis_df.groupby('cluster').mean(numeric_only=True)
                
                # Normalize profiles for easier comparison (z-score)
                zscore_profiles = (cluster_profiles - analysis_df.drop(columns=['id', 'cluster']).mean()) / analysis_df.drop(columns=['id', 'cluster']).std()
                
                fig_heatmap = px.imshow(zscore_profiles.T,
                                labels=dict(x="Cluster", y="Health Parameter", color="Z-Score"),
                                y=zscore_profiles.columns,
                                x=zscore_profiles.index,
                                color_continuous_scale='RdBu_r', 
                                color_continuous_midpoint=0,  # FIX: Replaced zmid with this
                                title="Cluster Profile Heatmap (Z-Scores vs. Population Mean)")
                st.plotly_chart(fig_heatmap, use_container_width=True)
                with st.expander("How to Interpret the Heatmap"):
                    st.info("""
                        This heatmap shows the defining characteristics of each cluster.
                        - **Red cells** indicate that a health parameter is significantly HIGHER than the population average for that cluster.
                        - **Blue cells** indicate that a parameter is significantly LOWER than the population average.
                        - **White cells** indicate the parameter is close to the average for that cluster.
                        This helps you understand what makes each group of animals distinct (e.g., "Cluster 2 is characterized by high glucose and low RBC").
                    """)
            else:
                st.warning("Not enough numeric data to perform PCA and clustering.")

if __name__ == "__main__":
    create_dashboard()