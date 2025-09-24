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
    page_title="OCR Confidence Analyzer",
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

def fit_normal_distribution(data):
    """Fit normal distribution to confidence scores"""
    mu, sigma = norm.fit(data)
    return mu, sigma

def calculate_percentiles(scores, mu, sigma):
    """Calculate percentiles based on fitted normal distribution"""
    percentiles = norm.cdf(scores, mu, sigma) * 100
    return percentiles

def plot_distribution_analysis(data, threshold, mu, sigma):
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
        title_text="OCR Confidence Score Analysis Dashboard"
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
    st.title("📊 OCR Confidence Score Analyzer")
    st.markdown("Analyze OCR confidence scores using normal distribution fitting and percentile conversion")
    
    # Sidebar for controls
    st.sidebar.header("Configuration")
    
    # Data input method
    data_source = st.sidebar.selectbox(
        "Choose data source:",
        ["Upload CSV File", "Upload JSON File", "Use Sample Data"]
    )
    
    data = None
    
    if data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="CSV should contain 'confidence_score' column and optionally 'text' column"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                
                # Validate required columns
                if 'confidence_score' not in data.columns:
                    st.error("CSV must contain 'confidence_score' column")
                    return
                
                # Add text column if not present
                if 'text' not in data.columns:
                    data['text'] = [f"Text_{i}" for i in range(len(data))]
                    
                st.success(f"✅ Loaded {len(data)} records from CSV")
                
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                return
    
    elif data_source == "Upload JSON File":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a JSON file",
            type="json",
            help="JSON should contain OCR results with confidence scores"
        )
        
        if uploaded_file is not None:
            try:
                json_data = json.load(uploaded_file)
                
                # Try to extract confidence scores from various JSON structures
                confidence_scores = []
                texts = []
                
                st.info("🔍 Detected JSON structure. Analyzing for confidence scores...")
                
                # Handle different JSON structures
                if isinstance(json_data, list):
                    for item in json_data:
                        if isinstance(item, dict):
                            # Look for confidence score in various fields
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
                    
                    # Handle generic nested structure
                    else:
                        def extract_from_nested(obj, path=""):
                            if isinstance(obj, dict):
                                for key, value in obj.items():
                                    if key == 'confidence' and isinstance(value, (int, float)):
                                        parent = path.split('.')[-2] if '.' in path else 'root'
                                        confidence_scores.append(float(value))
                                        # Try to find associated text
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
                    
                    # Display JSON structure info
                    st.success(f"✅ Loaded {len(data)} records from JSON")
                    
                    # Show some details about the extracted data
                    if 'analyzeResult' in json_data:
                        st.info("📄 Detected Azure Form Recognizer OCR format")
                        if 'pages' in json_data['analyzeResult']:
                            st.info(f"📖 Found {len(json_data['analyzeResult']['pages'])} page(s)")
                    
                    # Show confidence score range
                    min_conf = min(confidence_scores)
                    max_conf = max(confidence_scores)
                    st.info(f"📊 Confidence scores range: {min_conf:.3f} to {max_conf:.3f}")
                    
                else:
                    st.error("Could not extract confidence scores from JSON. Please check the file format.")
                    st.info("💡 Expected JSON formats:")
                    st.code("""
1. Azure Form Recognizer format:
{
  "analyzeResult": {
    "pages": [
      {
        "words": [
          {
            "content": "word_text",
            "confidence": 0.995
          }
        ]
      }
    ]
  }
}

2. Simple list format:
[
  {
    "text": "word",
    "confidence": 0.95
  }
]

3. Generic nested format with confidence fields
                    """, language="json")
                    return
                    
            except Exception as e:
                st.error(f"Error reading JSON file: {str(e)}")
                return
    
    else:  # Use sample data
        data = load_sample_data()
        st.info("📝 Using sample OCR data for demonstration")
    
    if data is not None:
        # Show preview of extracted data
        st.subheader("📋 Data Preview")
        with st.expander("Click to view sample of extracted data"):
            st.dataframe(data.head(10), use_container_width=True)
        
        # Display basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Mean Confidence", f"{data['confidence_score'].mean():.2f}")
        with col3:
            st.metric("Std Deviation", f"{data['confidence_score'].std():.2f}")
        with col4:
            st.metric("Min/Max", f"{data['confidence_score'].min():.1f} / {data['confidence_score'].max():.1f}")
        
        # Fit normal distribution
        mu, sigma = fit_normal_distribution(data['confidence_score'])
        
        st.subheader("📈 Normal Distribution Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean (μ)", f"{mu:.2f}")
        with col2:
            st.metric("Standard Deviation (σ)", f"{sigma:.2f}")
        
        # Threshold slider
        st.subheader("🎯 Threshold Configuration")
        threshold = st.slider(
            "Set confidence threshold:",
            min_value=float(data['confidence_score'].min()),
            max_value=float(data['confidence_score'].max()),
            value=float(mu - sigma),
            step=0.1,
            help="Scores below this threshold will be flagged for review"
        )
        
        # Calculate statistics based on threshold
        below_threshold = data[data['confidence_score'] < threshold]
        below_threshold_pct = len(below_threshold) / len(data) * 100
        
        # Calculate percentile of threshold
        threshold_percentile = norm.cdf(threshold, mu, sigma) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Records Below Threshold", 
                f"{len(below_threshold)} ({below_threshold_pct:.1f}%)",
                delta=f"-{100-below_threshold_pct:.1f}% above"
            )
        with col2:
            st.metric(
                "Threshold Percentile", 
                f"{threshold_percentile:.1f}%",
                help="Percentage of normal distribution below threshold"
            )
        with col3:
            expected_below = norm.cdf(threshold, mu, sigma) * len(data)
            st.metric(
                "Expected Below (Normal)", 
                f"{int(expected_below)} ({expected_below/len(data)*100:.1f}%)"
            )
        
        # Plot comprehensive analysis
        st.subheader("📊 Distribution Analysis")
        fig = plot_distribution_analysis(data, threshold, mu, sigma)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.subheader("📋 Detailed Analysis")
        
        # Add percentiles to data
        data_with_percentiles = data.copy()
        data_with_percentiles['percentile'] = calculate_percentiles(data['confidence_score'], mu, sigma)
        data_with_percentiles['below_threshold'] = data_with_percentiles['confidence_score'] < threshold
        
        # Show records below threshold
        if len(below_threshold) > 0:
            st.subheader("⚠️ Records Below Threshold (Requires Review)")
            below_threshold_with_percentiles = data_with_percentiles[
                data_with_percentiles['below_threshold']
            ].sort_values('confidence_score')
            
            st.dataframe(
                below_threshold_with_percentiles[['text', 'confidence_score', 'percentile']].head(100),
                use_container_width=True
            )
            
            # Download button for filtered results
            csv_buffer = io.StringIO()
            below_threshold_with_percentiles.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download Records Below Threshold",
                data=csv_buffer.getvalue(),
                file_name=f"low_confidence_ocr_results_threshold_{threshold:.1f}.csv",
                mime="text/csv"
            )
        
        # Statistical tests
        st.subheader("🔍 Statistical Analysis")
        
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(data['confidence_score'].sample(min(5000, len(data))))
        ks_stat, ks_p = stats.kstest(data['confidence_score'], lambda x: norm.cdf(x, mu, sigma))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Shapiro-Wilk Test p-value", 
                f"{shapiro_p:.4f}",
                help="Tests if data follows normal distribution (p > 0.05 suggests normal)"
            )
        with col2:
            st.metric(
                "Kolmogorov-Smirnov Test p-value", 
                f"{ks_p:.4f}",
                help="Tests goodness of fit to normal distribution"
            )
        
        # Interpretation
        if shapiro_p > 0.05:
            st.success("✅ Data appears to follow a normal distribution (Shapiro-Wilk p > 0.05)")
        else:
            st.warning("⚠️ Data may not perfectly follow normal distribution, but analysis is still useful")
        
        # Show sample of all data with percentiles
        st.subheader("📊 Sample Data with Percentiles")
        st.dataframe(
            data_with_percentiles.head(100),
            use_container_width=True
        )
        
        # Summary insights
        st.subheader("💡 Key Insights")
        insights = [
            f"Your OCR data has a mean confidence of {mu:.2f} with standard deviation {sigma:.2f}",
            f"With threshold set at {threshold:.2f}, {below_threshold_pct:.1f}% of records require review",
            f"The threshold corresponds to the {threshold_percentile:.1f}th percentile of the fitted normal distribution",
            f"Records with percentiles below {threshold_percentile:.1f}% should be flagged for quality review"
        ]
        
        for insight in insights:
            st.info(f"• {insight}")

if __name__ == "__main__":
    main()