import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json

# Set page config
st.set_page_config(
    page_title="Multi-File OCR Confidence Analyzer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert > div {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Generate sample OCR data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate mixed confidence scores (some high confidence, some low)
    high_conf = np.random.normal(85, 8, int(n_samples * 0.7))
    low_conf = np.random.normal(45, 15, int(n_samples * 0.3))
    
    confidence_scores = np.concatenate([high_conf, low_conf])
    confidence_scores = np.clip(confidence_scores, 0, 100)
    
    # Generate corresponding text data
    texts = [f"Sample_OCR_Text_{i}" for i in range(len(confidence_scores))]
    
    data = pd.DataFrame({
        'text': texts,
        'confidence_score': confidence_scores,
        'word_id': range(len(confidence_scores))
    })
    
    return data

def load_csv_file(file, filename):
    """Load and validate CSV file"""
    try:
        data = pd.read_csv(file)
        
        if 'confidence_score' not in data.columns:
            return None, f"CSV must contain 'confidence_score' column"
        
        if 'text' not in data.columns:
            data['text'] = [f"Text_{i}" for i in range(len(data))]
        
        return data, None
    except Exception as e:
        return None, f"Error reading CSV: {str(e)}"

def load_json_file(file, filename):
    """Load and parse JSON file"""
    try:
        json_data = json.load(file)
        
        confidence_scores = []
        texts = []
        
        # Handle different JSON structures
        if isinstance(json_data, list):
            for item in json_data:
                if isinstance(item, dict):
                    conf_score = item.get('confidence', item.get('confidence_score', item.get('score')))
                    text = item.get('text', item.get('word', item.get('content', f"Text_{len(confidence_scores)}")))
                    
                    if conf_score is not None:
                        confidence_scores.append(float(conf_score))
                        texts.append(str(text))
        
        elif isinstance(json_data, dict):
            # Handle Azure Form Recognizer OCR structure
            if 'analyzeResult' in json_data:
                analyze_result = json_data['analyzeResult']
                if 'pages' in analyze_result:
                    for page in analyze_result['pages']:
                        if 'words' in page:
                            for word in page['words']:
                                if 'confidence' in word and 'content' in word:
                                    confidence_scores.append(float(word['confidence']))
                                    texts.append(str(word['content']))
            else:
                def extract_from_nested(obj, path=""):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if key == 'confidence' and isinstance(value, (int, float)):
                                parent = path.split('.')[-2] if '.' in path else 'root'
                                confidence_scores.append(float(value))
                                text_content = obj.get('content', obj.get('text', obj.get('word', f"{parent}_{len(confidence_scores)}")))
                                texts.append(str(text_content))
                            elif isinstance(value, (dict, list)):
                                extract_from_nested(value, f"{path}.{key}" if path else key)
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            extract_from_nested(item, f"{path}[{i}]" if path else f"[{i}]")
                
                extract_from_nested(json_data)
        
        if confidence_scores:
            data = pd.DataFrame({
                'text': texts,
                'confidence_score': confidence_scores
            })
            return data, None
        else:
            return None, "Could not extract confidence scores from JSON"
            
    except Exception as e:
        return None, f"Error reading JSON: {str(e)}"

def fit_normal_distribution(data):
    """Fit normal distribution to confidence scores"""
    mu, sigma = norm.fit(data)
    return mu, sigma

def calculate_percentiles(scores, mu, sigma):
    """Calculate percentiles based on fitted normal distribution"""
    percentiles = norm.cdf(scores, mu, sigma) * 100
    return percentiles

def analyze_file(data, filename, threshold, review_percentage):
    """Perform complete analysis on a single file"""
    results = {}
    
    # Basic statistics
    results['filename'] = filename
    results['total_records'] = len(data)
    results['mean_confidence'] = data['confidence_score'].mean()
    results['std_confidence'] = data['confidence_score'].std()
    results['min_confidence'] = data['confidence_score'].min()
    results['max_confidence'] = data['confidence_score'].max()
    
    # Fit normal distribution
    mu, sigma = fit_normal_distribution(data['confidence_score'])
    results['mu'] = mu
    results['sigma'] = sigma
    
    # Threshold analysis
    below_threshold = data[data['confidence_score'] < threshold]
    results['below_threshold_count'] = len(below_threshold)
    results['below_threshold_pct'] = len(below_threshold) / len(data) * 100
    results['threshold_percentile'] = norm.cdf(threshold, mu, sigma) * 100
    
    # Add percentiles to data
    data_with_percentiles = data.copy()
    data_with_percentiles['percentile'] = calculate_percentiles(data['confidence_score'], mu, sigma)
    data_with_percentiles['below_threshold'] = data_with_percentiles['confidence_score'] < threshold
    
    results['data'] = data
    results['data_with_percentiles'] = data_with_percentiles
    results['below_threshold_data'] = below_threshold
    
    # Determine if review is needed based on user-defined percentage
    results['review_needed'] = results['below_threshold_pct'] >= review_percentage
    
    return results

def plot_distribution_analysis(data, threshold, mu, sigma, filename):
    """Create comprehensive distribution analysis plot"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Confidence Score Distribution with Normal Fit',
            'Cumulative Distribution Function (CDF)',
            'Scores Below Threshold',
            'Percentile Distribution'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Histogram with normal distribution overlay
    fig.add_trace(
        go.Histogram(
            x=data['confidence_score'],
            nbinsx=50,
            name='Actual Data',
            opacity=0.7,
            histnorm='probability density'
        ),
        row=1, col=1
    )
    
    # Normal distribution curve
    x_range = np.linspace(data['confidence_score'].min(), data['confidence_score'].max(), 100)
    normal_curve = norm.pdf(x_range, mu, sigma)
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=normal_curve,
            mode='lines',
            name='Normal Fit',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Add threshold line
    fig.add_vline(
        x=threshold,
        line=dict(color='orange', width=2, dash='dash'),
        annotation_text=f'Threshold: {threshold}',
        row=1, col=1
    )
    
    # Plot 2: CDF
    sorted_scores = np.sort(data['confidence_score'])
    cdf_empirical = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    cdf_theoretical = norm.cdf(sorted_scores, mu, sigma)
    
    fig.add_trace(
        go.Scatter(
            x=sorted_scores,
            y=cdf_empirical,
            mode='lines',
            name='Empirical CDF',
            line=dict(color='blue')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=sorted_scores,
            y=cdf_theoretical,
            mode='lines',
            name='Theoretical CDF',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=2
    )
    
    # Add threshold line to CDF
    fig.add_vline(
        x=threshold,
        line=dict(color='orange', width=2, dash='dash'),
        row=1, col=2
    )
    
    # Plot 3: Below threshold analysis
    below_threshold = data[data['confidence_score'] < threshold]['confidence_score']
    above_threshold = data[data['confidence_score'] >= threshold]['confidence_score']
    
    fig.add_trace(
        go.Histogram(
            x=below_threshold,
            name='Below Threshold',
            opacity=0.7,
            nbinsx=30,
            marker_color='red'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=above_threshold,
            name='Above Threshold',
            opacity=0.7,
            nbinsx=30,
            marker_color='green'
        ),
        row=2, col=1
    )
    
    # Plot 4: Percentile distribution
    percentiles = calculate_percentiles(data['confidence_score'], mu, sigma)
    
    fig.add_trace(
        go.Histogram(
            x=percentiles,
            name='Percentile Distribution',
            opacity=0.7,
            nbinsx=20,
            marker_color='purple'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"OCR Confidence Score Analysis - {filename}"
    )
    
    fig.update_xaxes(title_text="Confidence Score", row=1, col=1)
    fig.update_xaxes(title_text="Confidence Score", row=1, col=2)
    fig.update_xaxes(title_text="Confidence Score", row=2, col=1)
    fig.update_xaxes(title_text="Percentile", row=2, col=2)
    
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Probability", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    return fig

def main():
    st.title("📊 Multi-File OCR Confidence Score Analyzer")
    st.markdown("Analyze multiple OCR files with a common threshold and get a comprehensive summary")
    
    # Sidebar for controls
    st.sidebar.header("Configuration")
    
    # Data input method
    data_source = st.sidebar.selectbox(
        "Choose data source:",
        ["Upload Multiple Files", "Use Sample Data"]
    )
    
    all_file_results = []
    files_data = {}
    
    if data_source == "Upload Multiple Files":
        uploaded_files = st.sidebar.file_uploader(
            "Choose CSV or JSON files",
            type=["csv", "json"],
            accept_multiple_files=True,
            help="Upload multiple CSV or JSON files containing OCR confidence scores"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) uploaded")
            
            # Load all files
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                file_ext = filename.split('.')[-1].lower()
                
                if file_ext == 'csv':
                    data, error = load_csv_file(uploaded_file, filename)
                elif file_ext == 'json':
                    data, error = load_json_file(uploaded_file, filename)
                else:
                    error = "Unsupported file type"
                    data = None
                
                if error:
                    st.error(f"❌ {filename}: {error}")
                else:
                    files_data[filename] = data
                    st.success(f"✅ {filename}: Loaded {len(data)} records")
    
    else:  # Use sample data
        st.info("📝 Using sample OCR data for demonstration")
        # Create multiple sample files
        for i in range(3):
            filename = f"sample_file_{i+1}.csv"
            data = load_sample_data()
            # Vary the data slightly for each sample
            data['confidence_score'] = data['confidence_score'] + np.random.normal(0, 5, len(data))
            data['confidence_score'] = np.clip(data['confidence_score'], 0, 100)
            files_data[filename] = data
    
    if files_data:
        st.success(f"✅ Total files loaded: {len(files_data)}")
        
        # Calculate global min/max for threshold slider
        all_scores = []
        for data in files_data.values():
            all_scores.extend(data['confidence_score'].tolist())
        
        global_min = min(all_scores)
        global_max = max(all_scores)
        global_mean = np.mean(all_scores)
        global_std = np.std(all_scores)
        
        # Common threshold configuration
        st.sidebar.subheader("🎯 Common Threshold Configuration")
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            threshold_slider = st.slider(
                "Threshold slider:",
                min_value=float(global_min),
                max_value=float(global_max),
                value=float(global_mean - global_std),
                step=0.01,
                help="This threshold will be applied to all files",
                key="threshold_slider"
            )
        with col2:
            threshold_input = st.number_input(
                "Or type value:",
                min_value=float(global_min),
                max_value=float(global_max),
                value=float(global_mean - global_std),
                step=0.01,
                format="%.2f",
                help="Type exact threshold value",
                key="threshold_input"
            )
        
        # Use the most recently changed value
        if 'last_threshold' not in st.session_state:
            st.session_state.last_threshold = threshold_slider
        
        # Check which input was changed
        if threshold_input != st.session_state.last_threshold:
            threshold = threshold_input
            st.session_state.last_threshold = threshold_input
        else:
            threshold = threshold_slider
            st.session_state.last_threshold = threshold_slider
        
        st.sidebar.info(f"Current Threshold: {threshold:.2f}")
        
        # Review percentage configuration
        st.sidebar.subheader("📝 Review Criteria Configuration")
        
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            review_pct_slider = st.slider(
                "Review threshold %:",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=0.5,
                help="Files with this % or more records below threshold need review",
                key="review_pct_slider"
            )
        with col2:
            review_pct_input = st.number_input(
                "Or type %:",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=0.5,
                format="%.1f",
                help="Type exact review percentage",
                key="review_pct_input"
            )
        
        # Use the most recently changed value
        if 'last_review_pct' not in st.session_state:
            st.session_state.last_review_pct = review_pct_slider
        
        # Check which input was changed
        if review_pct_input != st.session_state.last_review_pct:
            review_percentage = review_pct_input
            st.session_state.last_review_pct = review_pct_input
        else:
            review_percentage = review_pct_slider
            st.session_state.last_review_pct = review_pct_slider
        
        st.sidebar.info(f"Review if ≥ {review_percentage:.1f}% below threshold")
        
        # Analyze all files
        for filename, data in files_data.items():
            results = analyze_file(data, filename, threshold, review_percentage)
            all_file_results.append(results)
        
        # Create summary table
        st.header("📋 Summary Table - All Files")
        
        summary_data = []
        for result in all_file_results:
            summary_data.append({
                'File Name': result['filename'],
                'Total Records': result['total_records'],
                'Mean Confidence': f"{result['mean_confidence']:.2f}",
                'Std Dev': f"{result['std_confidence']:.2f}",
                'Below Threshold': result['below_threshold_count'],
                'Below Threshold %': f"{result['below_threshold_pct']:.2f}%",
                'Review Needed': '✅ Yes' if result['review_needed'] else '❌ No'
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Color code the summary table
        def highlight_review_needed(row):
            if '✅ Yes' in row['Review Needed']:
                return ['background-color: #ffcccc'] * len(row)
            else:
                return ['background-color: #ccffcc'] * len(row)
        
        st.dataframe(
            summary_df.style.apply(highlight_review_needed, axis=1),
            use_container_width=True,
            height=min(400, (len(summary_df) + 1) * 35)
        )
        
        # Download summary
        csv_buffer = io.StringIO()
        summary_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Summary Table",
            data=csv_buffer.getvalue(),
            file_name=f"ocr_analysis_summary_threshold_{threshold:.1f}.csv",
            mime="text/csv"
        )
        
        # Show statistics
        st.header("📊 Overall Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        total_files = len(all_file_results)
        files_need_review = sum(1 for r in all_file_results if r['review_needed'])
        total_records = sum(r['total_records'] for r in all_file_results)
        total_below_threshold = sum(r['below_threshold_count'] for r in all_file_results)
        
        with col1:
            st.metric("Total Files", total_files)
        with col2:
            st.metric("Files Needing Review", files_need_review, 
                     delta=f"{files_need_review/total_files*100:.1f}%")
        with col3:
            st.metric("Total Records", total_records)
        with col4:
            st.metric("Total Below Threshold", total_below_threshold,
                     delta=f"{total_below_threshold/total_records*100:.1f}%")
        
        # Detailed analysis for each file
        st.header("🔍 Detailed Analysis by File")
        
        # File selector
        selected_file = st.selectbox(
            "Select a file to view detailed analysis:",
            options=[r['filename'] for r in all_file_results]
        )
        
        # Get selected file results
        selected_result = next(r for r in all_file_results if r['filename'] == selected_file)
        
        # Display detailed metrics
        st.subheader(f"📄 Analysis: {selected_file}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", selected_result['total_records'])
        with col2:
            st.metric("Mean Confidence", f"{selected_result['mean_confidence']:.2f}")
        with col3:
            st.metric("Std Deviation", f"{selected_result['std_confidence']:.2f}")
        with col4:
            st.metric("Min/Max", f"{selected_result['min_confidence']:.1f} / {selected_result['max_confidence']:.1f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean (μ)", f"{selected_result['mu']:.2f}")
        with col2:
            st.metric("Standard Deviation (σ)", f"{selected_result['sigma']:.2f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Records Below Threshold", 
                f"{selected_result['below_threshold_count']} ({selected_result['below_threshold_pct']:.1f}%)"
            )
        with col2:
            st.metric(
                "Threshold Percentile", 
                f"{selected_result['threshold_percentile']:.1f}%"
            )
        with col3:
            review_status = "⚠️ YES" if selected_result['review_needed'] else "✅ NO"
            st.metric("Review Needed", review_status)
        
        # Plot distribution analysis
        fig = plot_distribution_analysis(
            selected_result['data'], 
            threshold, 
            selected_result['mu'], 
            selected_result['sigma'],
            selected_file
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show records below threshold
        if selected_result['below_threshold_count'] > 0:
            st.subheader("⚠️ Records Below Threshold")
            below_threshold_display = selected_result['data_with_percentiles'][
                selected_result['data_with_percentiles']['below_threshold']
            ].sort_values('confidence_score')[['text', 'confidence_score', 'percentile']].head(100)
            
            st.dataframe(below_threshold_display, use_container_width=True)
            
            # Download button
            csv_buffer = io.StringIO()
            selected_result['data_with_percentiles'][
                selected_result['data_with_percentiles']['below_threshold']
            ].to_csv(csv_buffer, index=False)
            
            st.download_button(
                label=f"📥 Download Low Confidence Records - {selected_file}",
                data=csv_buffer.getvalue(),
                file_name=f"low_confidence_{selected_file}_threshold_{threshold:.1f}.csv",
                mime="text/csv"
            )
        
        # Key insights
        st.subheader("💡 Key Insights")
        insights = [
            f"File '{selected_file}' has a mean confidence of {selected_result['mean_confidence']:.2f} with standard deviation {selected_result['std_confidence']:.2f}",
            f"With threshold set at {threshold:.2f}, {selected_result['below_threshold_pct']:.1f}% of records require review",
            f"The threshold corresponds to the {selected_result['threshold_percentile']:.1f}th percentile of the fitted normal distribution",
            f"Review is {'recommended' if selected_result['review_needed'] else 'not necessary'} for this file (≥{review_percentage:.1f}% threshold)"
        ]
        
        for insight in insights:
            st.info(f"• {insight}")

if __name__ == "__main__":
    main()