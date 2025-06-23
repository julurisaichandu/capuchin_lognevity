import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Capuchin Health Monitor",
    page_icon="🐒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Simplified Reference Ranges and Health Categories
# ==============================================================================
REFERENCE_RANGES = {
    'RBC': {'min': 4.5, 'max': 7.0, 'units': '10^6/uL', 'category': 'Blood Health'},
    'Hematocrit': {'min': 35.0, 'max': 50.0, 'units': '%', 'category': 'Blood Health'},
    'Hemoglobin': {'min': 12.0, 'max': 17.0, 'units': 'g/dL', 'category': 'Blood Health'},
    'WBC': {'min': 3.5, 'max': 15.0, 'units': '10^3/uL', 'category': 'Immune System'},
    'Neutrophils (Absolute)': {'min': 1.5, 'max': 8.0, 'units': '10^3/uL', 'category': 'Immune System'},
    'Lymphocytes (Absolute)': {'min': 0.8, 'max': 4.0, 'units': '10^3/uL', 'category': 'Immune System'},
    'Platelets': {'min': 200.0, 'max': 600.0, 'units': '10^3/uL', 'category': 'Blood Health'},
    'Glucose': {'min': 60.0, 'max': 140.0, 'units': 'mg/dL', 'category': 'Metabolism'},
    'BUN': {'min': 8.0, 'max': 30.0, 'units': 'mg/dL', 'category': 'Kidney Function'},
    'Creatinine': {'min': 0.4, 'max': 1.0, 'units': 'mg/dL', 'category': 'Kidney Function'},
    'Calcium': {'min': 8.0, 'max': 11.0, 'units': 'mg/dL', 'category': 'Bone & Minerals'},
    'Phosphorus': {'min': 2.5, 'max': 6.0, 'units': 'mg/dL', 'category': 'Bone & Minerals'},
    'Total Protein': {'min': 5.5, 'max': 8.0, 'units': 'g/dL', 'category': 'Nutrition'},
    'Albumin': {'min': 3.0, 'max': 5.0, 'units': 'g/dL', 'category': 'Nutrition'},
    'ALT': {'min': 10.0, 'max': 50.0, 'units': 'U/L', 'category': 'Liver Health'},
    'AST': {'min': 10.0, 'max': 50.0, 'units': 'U/L', 'category': 'Liver Health'},
    'Cholesterol': {'min': 100.0, 'max': 250.0, 'units': 'mg/dL', 'category': 'Heart Health'},
    'MCV': {'min': 65.0, 'max': 85.0, 'units': 'fL', 'category': 'Blood Health'},
    'MCH': {'min': 20.0, 'max': 28.0, 'units': 'pg', 'category': 'Blood Health'},
    'MCHC': {'min': 30.0, 'max': 36.0, 'units': 'g/dL', 'category': 'Blood Health'},
    'Monocytes (Absolute)': {'min': 0.1, 'max': 0.8, 'units': '10^3/uL', 'category': 'Immune System'},
    'Eosinophils (Absolute)': {'min': 0.0, 'max': 0.6, 'units': '10^3/uL', 'category': 'Immune System'},
    'Basophils (Absolute)': {'min': 0.0, 'max': 0.2, 'units': '10^3/uL', 'category': 'Immune System'},
    'Sodium': {'min': 140.0, 'max': 155.0, 'units': 'mmol/L', 'category': 'Bone & Minerals'},
    'Potassium': {'min': 3.0, 'max': 5.0, 'units': 'mmol/L', 'category': 'Bone & Minerals'},
    'Chloride': {'min': 100.0, 'max': 120.0, 'units': 'mmol/L', 'category': 'Bone & Minerals'},
    'Globulin': {'min': 1.5, 'max': 3.5, 'units': 'g/dL', 'category': 'Nutrition'},
    'ALP': {'min': 30.0, 'max': 120.0, 'units': 'U/L', 'category': 'Liver Health'},
    'GGT': {'min': 10.0, 'max': 70.0, 'units': 'U/L', 'category': 'Liver Health'},
    'Total Bilirubin': {'min': 0.0, 'max': 0.3, 'units': 'mg/dL', 'category': 'Liver Health'}
}

# Simplified category names for display
CATEGORY_DISPLAY = {
    'Blood Health': '🩸 Blood Health',
    'Immune System': '🛡️ Immune System',
    'Metabolism': '⚡ Energy & Sugar',
    'Kidney Function': '🫘 Kidney Health',
    'Bone & Minerals': '🦴 Bones & Minerals',
    'Nutrition': '🥗 Nutrition',
    'Liver Health': '🫀 Liver Health',
    'Heart Health': '❤️ Heart Health'
}

# ==============================================================================
# Helper Functions for Simplified Analysis
# ==============================================================================

def get_health_status(value, parameter):
    """Returns a simple status with emoji"""
    ref = REFERENCE_RANGES.get(parameter)
    if not ref or pd.isna(value):
        return "❓ Unknown"
    
    if ref['min'] <= value <= ref['max']:
        return "🟢 Good"
    elif value < ref['min'] * 0.8 or value > ref['max'] * 1.2:
        return "🔴 Needs Attention"
    else:
        return "🟡 Watch Closely"

def get_letter_grade(health_score):
    """Converts health score to letter grade"""
    if health_score >= 90: 
        return "A", "Excellent"
    elif health_score >= 80: 
        return "B", "Good"
    elif health_score >= 70: 
        return "C", "Fair"
    elif health_score >= 60: 
        return "D", "Poor"
    else: 
        return "F", "Needs Immediate Attention"

def get_health_weather(health_score):
    """Returns weather emoji based on health"""
    if health_score >= 85: 
        return "☀️", "Sunny - Great Health!"
    elif health_score >= 70: 
        return "⛅", "Partly Cloudy - Some Concerns"
    else: 
        return "🌧️", "Stormy - Needs Care"

def calculate_simple_health_score(df, animal_id):
    """Calculates a simplified health score"""
    animal_data = df[df['id'] == animal_id]
    if animal_data.empty:
        return 0
        
    most_recent_date = animal_data['date'].max()
    recent_data = animal_data[animal_data['date'] == most_recent_date]
    
    good_count = 0
    total_count = 0
    
    for _, row in recent_data.iterrows():
        if row['test'] in REFERENCE_RANGES:
            total_count += 1
            status = get_health_status(row['result'], row['test'])
            if "🟢" in status:
                good_count += 1
            elif "🟡" in status:
                good_count += 0.5
    
    if total_count == 0:
        return 0
        
    return (good_count / total_count) * 100

def get_trend(df, animal_id, parameter, days=30):
    """Gets simple trend direction"""
    animal_data = df[(df['id'] == animal_id) & (df['test'] == parameter)]
    if len(animal_data) < 2:
        return "→", "Stable"
    
    recent_data = animal_data.sort_values('date').tail(5)
    if len(recent_data) < 2:
        return "→", "Stable"
    
    first_val = recent_data.iloc[0]['result']
    last_val = recent_data.iloc[-1]['result']
    
    change_pct = ((last_val - first_val) / first_val) * 100
    
    if change_pct > 10:
        return "↑", "Increasing"
    elif change_pct < -10:
        return "↓", "Decreasing"
    else:
        return "→", "Stable"

def get_category_health(df, animal_id, category):
    """Gets health status for a category"""
    animal_data = df[df['id'] == animal_id]
    if animal_data.empty:
        return 0
        
    most_recent_date = animal_data['date'].max()
    recent_data = animal_data[animal_data['date'] == most_recent_date]
    
    # Filter by category
    category_tests = [test for test, info in REFERENCE_RANGES.items() 
                     if info.get('category') == category]
    category_data = recent_data[recent_data['test'].isin(category_tests)]
    
    if category_data.empty:
        return 0
    
    good_count = 0
    total_count = 0
    
    for _, row in category_data.iterrows():
        total_count += 1
        status = get_health_status(row['result'], row['test'])
        if "🟢" in status:
            good_count += 1
        elif "🟡" in status:
            good_count += 0.5
    
    return (good_count / total_count) * 100 if total_count > 0 else 0

# ==============================================================================
# Data Loading
# ==============================================================================
@st.cache_data
def load_and_process_data(uploaded_files):
    """Loads and processes CSV files"""
    if not uploaded_files:
        return None
        
    all_dfs = []
    for uploaded_file in uploaded_files:
        try:
            if isinstance(uploaded_file, str):
                # It's a file path
                animal_id = os.path.basename(uploaded_file).split(' ')[0].capitalize()
                df = pd.read_csv(uploaded_file)
            else:
                # It's an uploaded file object
                animal_id = os.path.basename(uploaded_file.name).split(' ')[0].capitalize()
                df = pd.read_csv(uploaded_file)
            # animal_id = os.path.basename(uploaded_file.name).split(' ')[0].capitalize()
            # df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            df['id'] = animal_id
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['result'] = pd.to_numeric(df['result'], errors='coerce')
            df.dropna(subset=['date', 'test', 'result'], inplace=True)
            
            # Handle unit conversions for tests that come in different scales
            # If units contain "10^3/uL" and value is < 1, multiply by 1000
            mask = (df['units'] == '10^3/uL') & (df['result'] < 10)
            df.loc[mask, 'result'] = df.loc[mask, 'result'] * 1000
            
            # Handle RBC if needed (10^6/uL)
            mask = (df['units'] == '10^6/uL') & (df['test'] == 'RBC') & (df['result'] < 1)
            df.loc[mask, 'result'] = df.loc[mask, 'result'] * 1000000
            
            all_dfs.append(df)
        except Exception as e:
            st.sidebar.error(f"Error with {uploaded_file.name}: {str(e)}")
            
    if not all_dfs:
        return None
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df.sort_values(['date', 'test'])

# ==============================================================================
# Visualization Functions
# ==============================================================================

def create_health_report_card(df, animal_id):
    """Creates a visual health report card"""
    health_score = calculate_simple_health_score(df, animal_id)
    grade, grade_desc = get_letter_grade(health_score)
    weather, weather_desc = get_health_weather(health_score)
    
    # Create the report card
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Overall Health Grade", 
            f"{grade}",
            f"{grade_desc}",
            help="A simple grade based on how many tests are in the healthy range"
        )
    
    with col2:
        st.metric(
            "Health Weather",
            weather,
            weather_desc,
            help="A fun way to visualize overall health status"
        )
    
    with col3:
        st.metric(
            "Health Score",
            f"{health_score:.0f}/100",
            help="Percentage of tests in the healthy range"
        )
    
    # Category breakdown
    st.subheader("Health by Body System")
    
    categories = []
    scores = []
    colors = []
    
    for category, display_name in CATEGORY_DISPLAY.items():
        score = get_category_health(df, animal_id, category)
        categories.append(display_name)
        scores.append(score)
        
        if score >= 80:
            colors.append('#4CAF50')  # Green
        elif score >= 60:
            colors.append('#FFC107')  # Yellow
        else:
            colors.append('#F44336')  # Red
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=scores,
            marker_color=colors,
            text=[f"{s:.0f}%" for s in scores],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="How healthy is each body system?",
        yaxis_title="Health Score (%)",
        yaxis_range=[0, 110],
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_simple_timeline(df, animal_id, parameter):
    """Creates a simple timeline with health zones"""
    animal_data = df[(df['id'] == animal_id) & (df['test'] == parameter)].sort_values('date')
    if animal_data.empty:
        return None
    
    ref = REFERENCE_RANGES.get(parameter)
    if not ref:
        return None
    
    fig = go.Figure()
    
    # Add healthy zone
    fig.add_shape(
        type="rect",
        x0=animal_data['date'].min(),
        x1=animal_data['date'].max(),
        y0=ref['min'],
        y1=ref['max'],
        fillcolor="lightgreen",
        opacity=0.3,
        line_width=0,
        layer="below"
    )
    
    # Add the data line
    fig.add_trace(go.Scatter(
        x=animal_data['date'],
        y=animal_data['result'],
        mode='lines+markers',
        name=parameter,
        line=dict(color='darkblue', width=3),
        marker=dict(size=8)
    ))
    
    # Add annotations for healthy zone
    fig.add_annotation(
        x=animal_data['date'].mean(),
        y=ref['max'],
        text="Healthy Zone",
        showarrow=False,
        yshift=10,
        font=dict(color="green", size=12)
    )
    
    fig.update_layout(
        title=f"{parameter} Over Time",
        xaxis_title="Date",
        yaxis_title=f"{parameter} ({ref['units']})",
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_comparison_chart(df):
    """Creates a simple comparison of all monkeys"""
    monkeys = df['id'].unique()
    scores = []
    colors = []
    
    for monkey in monkeys:
        score = calculate_simple_health_score(df, monkey)
        scores.append(score)
        
        if score >= 80:
            colors.append('#4CAF50')
        elif score >= 60:
            colors.append('#FFC107')
        else:
            colors.append('#F44336')
    
    # Sort by score
    sorted_data = sorted(zip(monkeys, scores, colors), key=lambda x: x[1], reverse=True)
    monkeys, scores, colors = zip(*sorted_data)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(monkeys),
            y=list(scores),
            marker_color=list(colors),
            text=[f"{s:.0f}" for s in scores],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="🏆 Health Leaderboard",
        yaxis_title="Health Score",
        yaxis_range=[0, 110],
        showlegend=False,
        height=400
    )
    
    return fig

def create_alert_dashboard(df):
    """Creates an alert dashboard for monkeys needing attention"""
    alerts = []
    
    for animal_id in df['id'].unique():
        animal_data = df[df['id'] == animal_id]
        most_recent_date = animal_data['date'].max()
        recent_data = animal_data[animal_data['date'] == most_recent_date]
        
        red_alerts = []
        yellow_alerts = []
        
        for _, row in recent_data.iterrows():
            status = get_health_status(row['result'], row['test'])
            if "🔴" in status:
                red_alerts.append(row['test'])
            elif "🟡" in status:
                yellow_alerts.append(row['test'])
        
        if red_alerts or yellow_alerts:
            alerts.append({
                'Monkey': animal_id,
                'Urgent Issues': len(red_alerts),
                'Watch Items': len(yellow_alerts),
                'Details': ', '.join(red_alerts[:3]) + ('...' if len(red_alerts) > 3 else '')
            })
    
    if alerts:
        alert_df = pd.DataFrame(alerts)
        alert_df = alert_df.sort_values('Urgent Issues', ascending=False)
        
        # Style the dataframe
        def highlight_urgent(val):
            if isinstance(val, (int, float)) and val > 0:
                return 'background-color: #ffcccc'
            return ''
        
        styled_df = alert_df.style.map(highlight_urgent, subset=['Urgent Issues'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.success("🎉 Great news! All monkeys are healthy!")

# ==============================================================================
# Main Dashboard
# ==============================================================================
def main():
    st.title("🐒 Capuchin Monkey Health Monitor")
    st.markdown("*Simple, visual health tracking for your capuchin monkeys*")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Health Data")
        st.info("Upload CSV files for each monkey. The monkey's name will be taken from the filename.")
        
        # uploaded_files = st.file_uploader(
        #     "Choose CSV files",
        #     type="csv",
        #     accept_multiple_files=True
        # )
        uploaded_files = ['Allie one deceased_final.csv','Annie one deceased_final.csv','Bambi one living_final.csv','daisy one_final.csv','davey one_final.csv']

        
        if uploaded_files:
            st.success(f"✅ Loaded {len(uploaded_files)} monkey files")
    
    # Load data
    if uploaded_files:
        data = load_and_process_data(uploaded_files)
        
        if data is not None:
            # Create tabs
            tab1, tab2, tab3 = st.tabs([
                "🐒 Individual Health Reports",
                "📊 Compare All Monkeys",
                "🚨 Health Alerts & Insights"
            ])
            
            # Tab 1: Individual Reports
            with tab1:
                st.header("Individual Monkey Health Report")
                
                monkey_list = sorted(data['id'].unique())
                selected_monkey = st.selectbox(
                    "Choose a monkey to view:",
                    monkey_list,
                    format_func=lambda x: f"🐒 {x}"
                )
                
                if selected_monkey:
                    # Health Report Card
                    create_health_report_card(data, selected_monkey)
                    
                    # Recent Test Results
                    st.subheader("📋 Recent Test Results")
                    
                    animal_data = data[data['id'] == selected_monkey]
                    most_recent_date = animal_data['date'].max()
                    recent_data = animal_data[animal_data['date'] == most_recent_date]
                    
                    # Create a simple display
                    display_data = []
                    for _, row in recent_data.iterrows():
                        status = get_health_status(row['result'], row['test'])
                        trend, trend_desc = get_trend(data, selected_monkey, row['test'])
                        
                        display_data.append({
                            'Test': row['test'],
                            'Result': f"{row['result']:.1f} {row['units']}",
                            'Status': status,
                            'Trend': f"{trend} {trend_desc}"
                        })
                    
                    display_df = pd.DataFrame(display_data)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Simple Timeline
                    st.subheader("📈 Health Trends")
                    
                    # Group tests by category
                    categories = {}
                    for test, info in REFERENCE_RANGES.items():
                        cat = info.get('category', 'Other')
                        if cat not in categories:
                            categories[cat] = []
                        categories[cat].append(test)
                    
                    selected_category = st.selectbox(
                        "Choose a health category:",
                        list(CATEGORY_DISPLAY.keys()),
                        format_func=lambda x: CATEGORY_DISPLAY[x]
                    )
                    
                    if selected_category:
                        tests_in_category = categories[selected_category]
                        selected_test = st.selectbox(
                            "Choose a specific test:",
                            tests_in_category
                        )
                        
                        if selected_test:
                            fig = create_simple_timeline(data, selected_monkey, selected_test)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 2: Comparison
            with tab2:
                st.header("Compare All Monkeys")
                
                # Health Leaderboard
                fig = create_comparison_chart(data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Side by side comparison
                st.subheader("🔍 Compare Two Monkeys")
                
                col1, col2 = st.columns(2)
                with col1:
                    monkey1 = st.selectbox("First monkey:", monkey_list, key="m1")
                with col2:
                    monkey2 = st.selectbox("Second monkey:", 
                                         [m for m in monkey_list if m != monkey1], key="m2")
                
                if monkey1 and monkey2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"### 🐒 {monkey1}")
                        score1 = calculate_simple_health_score(data, monkey1)
                        grade1, _ = get_letter_grade(score1)
                        weather1, _ = get_health_weather(score1)
                        st.metric("Health Grade", f"{grade1} {weather1}")
                        
                        # Show category scores
                        for category, display_name in CATEGORY_DISPLAY.items():
                            score = get_category_health(data, monkey1, category)
                            if score >= 80:
                                st.markdown(f"{display_name}: 🟢 **Good**")
                            elif score >= 60:
                                st.markdown(f"{display_name}: 🟡 **Fair**")
                            else:
                                st.markdown(f"{display_name}: 🔴 **Needs Care**")
                    
                    with col2:
                        st.markdown(f"### 🐒 {monkey2}")
                        score2 = calculate_simple_health_score(data, monkey2)
                        grade2, _ = get_letter_grade(score2)
                        weather2, _ = get_health_weather(score2)
                        st.metric("Health Grade", f"{grade2} {weather2}")
                        
                        # Show category scores
                        for category, display_name in CATEGORY_DISPLAY.items():
                            score = get_category_health(data, monkey2, category)
                            if score >= 80:
                                st.markdown(f"{display_name}: 🟢 **Good**")
                            elif score >= 60:
                                st.markdown(f"{display_name}: 🟡 **Fair**")
                            else:
                                st.markdown(f"{display_name}: 🔴 **Needs Care**")
            
            # Tab 3: Alerts
            with tab3:
                st.header("🚨 Health Alerts & Insights")
                
                st.subheader("Monkeys Needing Attention")
                create_alert_dashboard(data)
                
                # Success Stories
                st.subheader("🌟 Success Stories")
                
                improvements = []
                for animal_id in data['id'].unique():
                    animal_data = data[data['id'] == animal_id]
                    
                    for test in REFERENCE_RANGES.keys():
                        test_data = animal_data[animal_data['test'] == test].sort_values('date')
                        if len(test_data) >= 3:
                            early_avg = test_data.head(3)['result'].mean()
                            recent_avg = test_data.tail(3)['result'].mean()
                            
                            # Check if moved into healthy range
                            ref = REFERENCE_RANGES[test]
                            was_unhealthy = early_avg < ref['min'] or early_avg > ref['max']
                            now_healthy = ref['min'] <= recent_avg <= ref['max']
                            
                            if was_unhealthy and now_healthy:
                                improvements.append(f"🎉 {animal_id}'s {test} is now in the healthy range!")
                
                if improvements:
                    for improvement in improvements[:5]:  # Show top 5
                        st.success(improvement)
                else:
                    st.info("No major improvements to report yet, but keep monitoring!")
                
                # Simple Insights
                st.subheader("💡 Quick Insights")
                
                # Find monkeys with similar issues
                st.markdown("**Monkeys with Similar Health Patterns:**")
                
                # Collect health issues for each monkey
                monkey_issues = {}
                for animal_id in data['id'].unique():
                    animal_data = data[data['id'] == animal_id]
                    most_recent_date = animal_data['date'].max()
                    recent_data = animal_data[animal_data['date'] == most_recent_date]
                    
                    issues = set()
                    for _, row in recent_data.iterrows():
                        status = get_health_status(row['result'], row['test'])
                        if "🔴" in status or "🟡" in status:
                            if row['test'] in REFERENCE_RANGES:
                                issues.add(REFERENCE_RANGES[row['test']]['category'])
                    
                    monkey_issues[animal_id] = issues
                
                # Find monkeys with overlapping issues (at least 2 categories in common)
                shown_pairs = set()
                found_similar = False
                
                for monkey1, issues1 in monkey_issues.items():
                    for monkey2, issues2 in monkey_issues.items():
                        if monkey1 != monkey2 and (monkey1, monkey2) not in shown_pairs:
                            common_issues = issues1.intersection(issues2)
                            if len(common_issues) >= 2:  # At least 2 categories in common
                                found_similar = True
                                shown_pairs.add((monkey1, monkey2))
                                shown_pairs.add((monkey2, monkey1))
                                
                                categories = [CATEGORY_DISPLAY.get(cat, cat) for cat in common_issues]
                                st.info(f"🔗 {monkey1} and {monkey2} both need attention for: {', '.join(categories)}")
                
                # Original exact pattern matching (kept as fallback)
                if not found_similar:
                    groupings = {}
                    for animal_id in data['id'].unique():
                        animal_data = data[data['id'] == animal_id]
                        most_recent_date = animal_data['date'].max()
                        recent_data = animal_data[animal_data['date'] == most_recent_date]
                        
                        issues = []
                        for _, row in recent_data.iterrows():
                            status = get_health_status(row['result'], row['test'])
                            if "🔴" in status or "🟡" in status:
                                if row['test'] in REFERENCE_RANGES:
                                    issues.append(REFERENCE_RANGES[row['test']]['category'])
                        
                        issue_pattern = tuple(sorted(set(issues)))
                        if issue_pattern not in groupings:
                            groupings[issue_pattern] = []
                        groupings[issue_pattern].append(animal_id)
                    
                    for pattern, monkeys in groupings.items():
                        if len(monkeys) > 1 and pattern:
                            categories = [CATEGORY_DISPLAY.get(cat, cat) for cat in pattern]
                            st.info(f"🔗 {', '.join(monkeys)} all need attention for: {', '.join(categories)}")
                            found_similar = True
                
                if not found_similar:
                    st.info("Each monkey has unique health patterns. No two monkeys share similar health concerns at this time.")
        
        else:
            st.error("Could not load the data. Please check your CSV files.")
    else:
        # Welcome screen
        st.info("""
        👋 Welcome to the Capuchin Health Monitor!
        
        This tool helps you track the health of your capuchin monkeys in a simple, visual way.
        
        **To get started:**
        1. Upload CSV files for each monkey using the sidebar
        2. Each file should be named with the monkey's name (e.g., "Allie.csv")
        3. The dashboard will automatically create easy-to-understand health reports
        
        **No complex charts or technical terms** - just simple grades, colors, and clear recommendations!
        """)

if __name__ == "__main__":
    main()