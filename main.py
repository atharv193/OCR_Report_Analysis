# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# from scipy.stats import norm
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import io
# import json

# # Set page config
# st.set_page_config(
#     page_title="OCR Confidence Analyzer",
#     page_icon="📊",
#     layout="wide"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .metric-container {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 0.5rem 0;
#     }
#     .stAlert > div {
#         padding: 1rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# def load_sample_data():
#     """Generate sample OCR data for demonstration"""
#     np.random.seed(42)
#     n_samples = 1000
    
#     # Generate mixed confidence scores (some high confidence, some low)
#     high_conf = np.random.normal(85, 8, int(n_samples * 0.7))
#     low_conf = np.random.normal(45, 15, int(n_samples * 0.3))
    
#     confidence_scores = np.concatenate([high_conf, low_conf])
#     confidence_scores = np.clip(confidence_scores, 0, 100)
    
#     # Generate corresponding text data
#     texts = [f"Sample_OCR_Text_{i}" for i in range(len(confidence_scores))]
    
#     data = pd.DataFrame({
#         'text': texts,
#         'confidence_score': confidence_scores,
#         'word_id': range(len(confidence_scores))
#     })
    
#     return data

# def fit_normal_distribution(data):
#     """Fit normal distribution to confidence scores"""
#     mu, sigma = norm.fit(data)
#     return mu, sigma

# def calculate_percentiles(scores, mu, sigma):
#     """Calculate percentiles based on fitted normal distribution"""
#     percentiles = norm.cdf(scores, mu, sigma) * 100
#     return percentiles

# def plot_distribution_analysis(data, threshold, mu, sigma):
#     """Create comprehensive distribution analysis plot"""
#     fig = make_subplots(
#         rows=2, cols=2,
#         subplot_titles=(
#             'Confidence Score Distribution with Normal Fit',
#             'Cumulative Distribution Function (CDF)',
#             'Scores Below Threshold',
#             'Percentile Distribution'
#         ),
#         specs=[[{"secondary_y": False}, {"secondary_y": False}],
#                [{"secondary_y": False}, {"secondary_y": False}]]
#     )
    
#     # Plot 1: Histogram with normal distribution overlay
#     fig.add_trace(
#         go.Histogram(
#             x=data['confidence_score'],
#             nbinsx=50,
#             name='Actual Data',
#             opacity=0.7,
#             histnorm='probability density'
#         ),
#         row=1, col=1
#     )
    
#     # Normal distribution curve
#     x_range = np.linspace(data['confidence_score'].min(), data['confidence_score'].max(), 100)
#     normal_curve = norm.pdf(x_range, mu, sigma)
    
#     fig.add_trace(
#         go.Scatter(
#             x=x_range,
#             y=normal_curve,
#             mode='lines',
#             name='Normal Fit',
#             line=dict(color='red', width=2)
#         ),
#         row=1, col=1
#     )
    
#     # Add threshold line
#     fig.add_vline(
#         x=threshold,
#         line=dict(color='orange', width=2, dash='dash'),
#         annotation_text=f'Threshold: {threshold}',
#         row=1, col=1
#     )
    
#     # Plot 2: CDF
#     sorted_scores = np.sort(data['confidence_score'])
#     cdf_empirical = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
#     cdf_theoretical = norm.cdf(sorted_scores, mu, sigma)
    
#     fig.add_trace(
#         go.Scatter(
#             x=sorted_scores,
#             y=cdf_empirical,
#             mode='lines',
#             name='Empirical CDF',
#             line=dict(color='blue')
#         ),
#         row=1, col=2
#     )
    
#     fig.add_trace(
#         go.Scatter(
#             x=sorted_scores,
#             y=cdf_theoretical,
#             mode='lines',
#             name='Theoretical CDF',
#             line=dict(color='red', dash='dash')
#         ),
#         row=1, col=2
#     )
    
#     # Add threshold line to CDF
#     fig.add_vline(
#         x=threshold,
#         line=dict(color='orange', width=2, dash='dash'),
#         row=1, col=2
#     )
    
#     # Plot 3: Below threshold analysis
#     below_threshold = data[data['confidence_score'] < threshold]['confidence_score']
#     above_threshold = data[data['confidence_score'] >= threshold]['confidence_score']
    
#     fig.add_trace(
#         go.Histogram(
#             x=below_threshold,
#             name='Below Threshold',
#             opacity=0.7,
#             nbinsx=30,
#             marker_color='red'
#         ),
#         row=2, col=1
#     )
    
#     fig.add_trace(
#         go.Histogram(
#             x=above_threshold,
#             name='Above Threshold',
#             opacity=0.7,
#             nbinsx=30,
#             marker_color='green'
#         ),
#         row=2, col=1
#     )
    
#     # Plot 4: Percentile distribution
#     percentiles = calculate_percentiles(data['confidence_score'], mu, sigma)
    
#     fig.add_trace(
#         go.Histogram(
#             x=percentiles,
#             name='Percentile Distribution',
#             opacity=0.7,
#             nbinsx=20,
#             marker_color='purple'
#         ),
#         row=2, col=2
#     )
    
#     # Update layout
#     fig.update_layout(
#         height=800,
#         showlegend=True,
#         title_text="OCR Confidence Score Analysis Dashboard"
#     )
    
#     fig.update_xaxes(title_text="Confidence Score", row=1, col=1)
#     fig.update_xaxes(title_text="Confidence Score", row=1, col=2)
#     fig.update_xaxes(title_text="Confidence Score", row=2, col=1)
#     fig.update_xaxes(title_text="Percentile", row=2, col=2)
    
#     fig.update_yaxes(title_text="Density", row=1, col=1)
#     fig.update_yaxes(title_text="Cumulative Probability", row=1, col=2)
#     fig.update_yaxes(title_text="Frequency", row=2, col=1)
#     fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
#     return fig

# def main():
#     st.title("📊 OCR Confidence Score Analyzer")
#     st.markdown("Analyze OCR confidence scores using normal distribution fitting and percentile conversion")
    
#     # Sidebar for controls
#     st.sidebar.header("Configuration")
    
#     # Data input method
#     data_source = st.sidebar.selectbox(
#         "Choose data source:",
#         ["Upload CSV File", "Upload JSON File", "Use Sample Data"]
#     )
    
#     data = None
    
#     if data_source == "Upload CSV File":
#         uploaded_file = st.sidebar.file_uploader(
#             "Choose a CSV file",
#             type="csv",
#             help="CSV should contain 'confidence_score' column and optionally 'text' column"
#         )
        
#         if uploaded_file is not None:
#             try:
#                 data = pd.read_csv(uploaded_file)
                
#                 # Validate required columns
#                 if 'confidence_score' not in data.columns:
#                     st.error("CSV must contain 'confidence_score' column")
#                     return
                
#                 # Add text column if not present
#                 if 'text' not in data.columns:
#                     data['text'] = [f"Text_{i}" for i in range(len(data))]
                    
#                 st.success(f"✅ Loaded {len(data)} records from CSV")
                
#             except Exception as e:
#                 st.error(f"Error reading CSV file: {str(e)}")
#                 return
    
#     elif data_source == "Upload JSON File":
#         uploaded_file = st.sidebar.file_uploader(
#             "Choose a JSON file",
#             type="json",
#             help="JSON should contain OCR results with confidence scores"
#         )
        
#         if uploaded_file is not None:
#             try:
#                 json_data = json.load(uploaded_file)
                
#                 # Try to extract confidence scores from various JSON structures
#                 confidence_scores = []
#                 texts = []
                
#                 st.info("🔍 Detected JSON structure. Analyzing for confidence scores...")
                
#                 # Handle different JSON structures
#                 if isinstance(json_data, list):
#                     for item in json_data:
#                         if isinstance(item, dict):
#                             # Look for confidence score in various fields
#                             conf_score = item.get('confidence', item.get('confidence_score', item.get('score')))
#                             text = item.get('text', item.get('word', item.get('content', f"Text_{len(confidence_scores)}")))
                            
#                             if conf_score is not None:
#                                 confidence_scores.append(float(conf_score))
#                                 texts.append(str(text))
                
#                 elif isinstance(json_data, dict):
#                     # Handle Azure Form Recognizer OCR structure
#                     if 'analyzeResult' in json_data:
#                         analyze_result = json_data['analyzeResult']
#                         if 'pages' in analyze_result:
#                             for page in analyze_result['pages']:
#                                 if 'words' in page:
#                                     for word in page['words']:
#                                         if 'confidence' in word and 'content' in word:
#                                             confidence_scores.append(float(word['confidence']))
#                                             texts.append(str(word['content']))
                    
#                     # Handle generic nested structure
#                     else:
#                         def extract_from_nested(obj, path=""):
#                             if isinstance(obj, dict):
#                                 for key, value in obj.items():
#                                     if key == 'confidence' and isinstance(value, (int, float)):
#                                         parent = path.split('.')[-2] if '.' in path else 'root'
#                                         confidence_scores.append(float(value))
#                                         # Try to find associated text
#                                         text_content = obj.get('content', obj.get('text', obj.get('word', f"{parent}_{len(confidence_scores)}")))
#                                         texts.append(str(text_content))
#                                     elif isinstance(value, (dict, list)):
#                                         extract_from_nested(value, f"{path}.{key}" if path else key)
#                             elif isinstance(obj, list):
#                                 for i, item in enumerate(obj):
#                                     extract_from_nested(item, f"{path}[{i}]" if path else f"[{i}]")
                        
#                         extract_from_nested(json_data)
                
#                 if confidence_scores:
#                     data = pd.DataFrame({
#                         'text': texts,
#                         'confidence_score': confidence_scores
#                     })
                    
#                     # Display JSON structure info
#                     st.success(f"✅ Loaded {len(data)} records from JSON")
                    
#                     # Show some details about the extracted data
#                     if 'analyzeResult' in json_data:
#                         st.info("📄 Detected Azure Form Recognizer OCR format")
#                         if 'pages' in json_data['analyzeResult']:
#                             st.info(f"📖 Found {len(json_data['analyzeResult']['pages'])} page(s)")
                    
#                     # Show confidence score range
#                     min_conf = min(confidence_scores)
#                     max_conf = max(confidence_scores)
#                     st.info(f"📊 Confidence scores range: {min_conf:.3f} to {max_conf:.3f}")
                    
#                 else:
#                     st.error("Could not extract confidence scores from JSON. Please check the file format.")
#                     st.info("💡 Expected JSON formats:")
#                     st.code("""
# 1. Azure Form Recognizer format:
# {
#   "analyzeResult": {
#     "pages": [
#       {
#         "words": [
#           {
#             "content": "word_text",
#             "confidence": 0.995
#           }
#         ]
#       }
#     ]
#   }
# }

# 2. Simple list format:
# [
#   {
#     "text": "word",
#     "confidence": 0.95
#   }
# ]

# 3. Generic nested format with confidence fields
#                     """, language="json")
#                     return
                    
#             except Exception as e:
#                 st.error(f"Error reading JSON file: {str(e)}")
#                 return
    
#     else:  # Use sample data
#         data = load_sample_data()
#         st.info("📝 Using sample OCR data for demonstration")
    
#     if data is not None:
#         # Show preview of extracted data
#         st.subheader("📋 Data Preview")
#         with st.expander("Click to view sample of extracted data"):
#             st.dataframe(data.head(10), use_container_width=True)
        
#         # Display basic statistics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("Total Records", len(data))
#         with col2:
#             st.metric("Mean Confidence", f"{data['confidence_score'].mean():.2f}")
#         with col3:
#             st.metric("Std Deviation", f"{data['confidence_score'].std():.2f}")
#         with col4:
#             st.metric("Min/Max", f"{data['confidence_score'].min():.1f} / {data['confidence_score'].max():.1f}")
        
#         # Fit normal distribution
#         mu, sigma = fit_normal_distribution(data['confidence_score'])
        
#         st.subheader("📈 Normal Distribution Parameters")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("Mean (μ)", f"{mu:.2f}")
#         with col2:
#             st.metric("Standard Deviation (σ)", f"{sigma:.2f}")
        
#         # Threshold slider
#         st.subheader("🎯 Threshold Configuration")
#         threshold = st.slider(
#             "Set confidence threshold:",
#             min_value=float(data['confidence_score'].min()),
#             max_value=float(data['confidence_score'].max()),
#             value=float(mu - sigma),
#             step=0.1,
#             help="Scores below this threshold will be flagged for review"
#         )
        
#         # Calculate statistics based on threshold
#         below_threshold = data[data['confidence_score'] < threshold]
#         below_threshold_pct = len(below_threshold) / len(data) * 100
        
#         # Calculate percentile of threshold
#         threshold_percentile = norm.cdf(threshold, mu, sigma) * 100
        
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric(
#                 "Records Below Threshold", 
#                 f"{len(below_threshold)} ({below_threshold_pct:.1f}%)",
#                 delta=f"-{100-below_threshold_pct:.1f}% above"
#             )
#         with col2:
#             st.metric(
#                 "Threshold Percentile", 
#                 f"{threshold_percentile:.1f}%",
#                 help="Percentage of normal distribution below threshold"
#             )
#         with col3:
#             expected_below = norm.cdf(threshold, mu, sigma) * len(data)
#             st.metric(
#                 "Expected Below (Normal)", 
#                 f"{int(expected_below)} ({expected_below/len(data)*100:.1f}%)"
#             )
        
#         # Plot comprehensive analysis
#         st.subheader("📊 Distribution Analysis")
#         fig = plot_distribution_analysis(data, threshold, mu, sigma)
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Detailed results table
#         st.subheader("📋 Detailed Analysis")
        
#         # Add percentiles to data
#         data_with_percentiles = data.copy()
#         data_with_percentiles['percentile'] = calculate_percentiles(data['confidence_score'], mu, sigma)
#         data_with_percentiles['below_threshold'] = data_with_percentiles['confidence_score'] < threshold
        
#         # Show records below threshold
#         if len(below_threshold) > 0:
#             st.subheader("⚠️ Records Below Threshold (Requires Review)")
#             below_threshold_with_percentiles = data_with_percentiles[
#                 data_with_percentiles['below_threshold']
#             ].sort_values('confidence_score')
            
#             st.dataframe(
#                 below_threshold_with_percentiles[['text', 'confidence_score', 'percentile']].head(100),
#                 use_container_width=True
#             )
            
#             # Download button for filtered results
#             csv_buffer = io.StringIO()
#             below_threshold_with_percentiles.to_csv(csv_buffer, index=False)
            
#             st.download_button(
#                 label="📥 Download Records Below Threshold",
#                 data=csv_buffer.getvalue(),
#                 file_name=f"low_confidence_ocr_results_threshold_{threshold:.1f}.csv",
#                 mime="text/csv"
#             )
        
#         # Statistical tests
#         # st.subheader("🔍 Statistical Analysis")
        
#         # # Normality test
#         # shapiro_stat, shapiro_p = stats.shapiro(data['confidence_score'].sample(min(5000, len(data))))
#         # ks_stat, ks_p = stats.kstest(data['confidence_score'], lambda x: norm.cdf(x, mu, sigma))
        
#         # col1, col2 = st.columns(2)
#         # with col1:
#         #     st.metric(
#         #         "Shapiro-Wilk Test p-value", 
#         #         f"{shapiro_p:.4f}",
#         #         help="Tests if data follows normal distribution (p > 0.05 suggests normal)"
#         #     )
#         # with col2:
#         #     st.metric(
#         #         "Kolmogorov-Smirnov Test p-value", 
#         #         f"{ks_p:.4f}",
#         #         help="Tests goodness of fit to normal distribution"
#         #     )
        
#         # # Interpretation
#         # if shapiro_p > 0.05:
#         #     st.success("✅ Data appears to follow a normal distribution (Shapiro-Wilk p > 0.05)")
#         # else:
#         #     st.warning("⚠️ Data may not perfectly follow normal distribution, but analysis is still useful")
        
#         # Show sample of all data with percentiles
#         st.subheader("📊 Sample Data with Percentiles")
#         st.dataframe(
#             data_with_percentiles.head(100),
#             use_container_width=True
#         )
        
#         # Summary insights
#         st.subheader("💡 Key Insights")
#         insights = [
#             f"Your OCR data has a mean confidence of {mu:.2f} with standard deviation {sigma:.2f}",
#             f"With threshold set at {threshold:.2f}, {below_threshold_pct:.1f}% of records require review",
#             f"The threshold corresponds to the {threshold_percentile:.1f}th percentile of the fitted normal distribution",
#             f"Records with percentiles below {threshold_percentile:.1f}% should be flagged for quality review"
#         ]
        
#         for insight in insights:
#             st.info(f"• {insight}")

# if __name__ == "__main__":
#     main()


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# from scipy.stats import norm
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import io
# import json
# from typing import List, Dict, Tuple

# # Set page config
# st.set_page_config(
#     page_title="Multi-File OCR Confidence Analyzer",
#     page_icon="📊",
#     layout="wide"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .metric-container {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 0.5rem 0;
#     }
#     .stAlert > div {
#         padding: 1rem;
#     }
#     .review-needed {
#         background-color: #ffebee;
#         padding: 0.5rem;
#         border-radius: 0.25rem;
#         border-left: 4px solid #ef5350;
#     }
#     .review-ok {
#         background-color: #e8f5e9;
#         padding: 0.5rem;
#         border-radius: 0.25rem;
#         border-left: 4px solid #66bb6a;
#     }
#     .file-section {
#         border: 2px solid #e0e0e0;
#         border-radius: 0.5rem;
#         padding: 1rem;
#         margin: 1rem 0;
#         background-color: #fafafa;
#     }
# </style>
# """, unsafe_allow_html=True)

# def load_sample_data():
#     """Generate sample OCR data for demonstration"""
#     np.random.seed(42)
#     n_samples = 1000
    
#     # Generate mixed confidence scores (some high confidence, some low)
#     high_conf = np.random.normal(85, 8, int(n_samples * 0.7))
#     low_conf = np.random.normal(45, 15, int(n_samples * 0.3))
    
#     confidence_scores = np.concatenate([high_conf, low_conf])
#     confidence_scores = np.clip(confidence_scores, 0, 100)
    
#     # Generate corresponding text data
#     texts = [f"Sample_OCR_Text_{i}" for i in range(len(confidence_scores))]
    
#     data = pd.DataFrame({
#         'text': texts,
#         'confidence_score': confidence_scores,
#         'word_id': range(len(confidence_scores))
#     })
    
#     return data

# def process_csv_file(file, filename: str) -> pd.DataFrame:
#     """Process a CSV file and extract confidence scores"""
#     try:
#         data = pd.read_csv(file)
        
#         if 'confidence_score' not in data.columns:
#             st.warning(f"⚠️ {filename}: No 'confidence_score' column found")
#             return None
        
#         if 'text' not in data.columns:
#             data['text'] = [f"Text_{i}" for i in range(len(data))]
        
#         data['filename'] = filename
#         return data
        
#     except Exception as e:
#         st.error(f"❌ Error reading {filename}: {str(e)}")
#         return None

# def process_json_file(file, filename: str) -> pd.DataFrame:
#     """Process a JSON file and extract confidence scores"""
#     try:
#         json_data = json.load(file)
#         confidence_scores = []
#         texts = []
        
#         # Handle different JSON structures
#         if isinstance(json_data, list):
#             for item in json_data:
#                 if isinstance(item, dict):
#                     conf_score = item.get('confidence', item.get('confidence_score', item.get('score')))
#                     text = item.get('text', item.get('word', item.get('content', f"Text_{len(confidence_scores)}")))
                    
#                     if conf_score is not None:
#                         confidence_scores.append(float(conf_score))
#                         texts.append(str(text))
        
#         elif isinstance(json_data, dict):
#             # Handle Azure Form Recognizer OCR structure
#             if 'analyzeResult' in json_data:
#                 analyze_result = json_data['analyzeResult']
#                 if 'pages' in analyze_result:
#                     for page in analyze_result['pages']:
#                         if 'words' in page:
#                             for word in page['words']:
#                                 if 'confidence' in word and 'content' in word:
#                                     confidence_scores.append(float(word['confidence']))
#                                     texts.append(str(word['content']))
#             else:
#                 # Handle generic nested structure
#                 def extract_from_nested(obj, path=""):
#                     if isinstance(obj, dict):
#                         for key, value in obj.items():
#                             if key == 'confidence' and isinstance(value, (int, float)):
#                                 parent = path.split('.')[-2] if '.' in path else 'root'
#                                 confidence_scores.append(float(value))
#                                 text_content = obj.get('content', obj.get('text', obj.get('word', f"{parent}_{len(confidence_scores)}")))
#                                 texts.append(str(text_content))
#                             elif isinstance(value, (dict, list)):
#                                 extract_from_nested(value, f"{path}.{key}" if path else key)
#                     elif isinstance(obj, list):
#                         for i, item in enumerate(obj):
#                             extract_from_nested(item, f"{path}[{i}]" if path else f"[{i}]")
                
#                 extract_from_nested(json_data)
        
#         if confidence_scores:
#             data = pd.DataFrame({
#                 'text': texts,
#                 'confidence_score': confidence_scores,
#                 'filename': filename
#             })
#             return data
#         else:
#             st.warning(f"⚠️ {filename}: Could not extract confidence scores")
#             return None
            
#     except Exception as e:
#         st.error(f"❌ Error reading {filename}: {str(e)}")
#         return None

# def fit_normal_distribution(data):
#     """Fit normal distribution to confidence scores"""
#     mu, sigma = norm.fit(data)
#     return mu, sigma

# def calculate_percentiles(scores, mu, sigma):
#     """Calculate percentiles based on fitted normal distribution"""
#     percentiles = norm.cdf(scores, mu, sigma) * 100
#     return percentiles

# def analyze_single_file(file_data: pd.DataFrame, filename: str, threshold_method: str, custom_threshold: float = None) -> Dict:
#     """Analyze a single file with its own distribution"""
    
#     # Fit normal distribution for this file
#     mu, sigma = fit_normal_distribution(file_data['confidence_score'])
    
#     # Determine threshold based on method
#     if threshold_method == "Custom":
#         threshold = custom_threshold
#     elif threshold_method == "Mean - 1 SD":
#         threshold = mu - sigma
#     elif threshold_method == "Mean - 2 SD":
#         threshold = mu - 2 * sigma
#     elif threshold_method == "25th Percentile":
#         threshold = np.percentile(file_data['confidence_score'], 25)
#     elif threshold_method == "Median":
#         threshold = np.median(file_data['confidence_score'])
#     else:
#         threshold = mu - sigma
    
#     # Calculate percentiles for this file
#     percentiles = calculate_percentiles(file_data['confidence_score'], mu, sigma)
#     file_data_with_percentiles = file_data.copy()
#     file_data_with_percentiles['percentile'] = percentiles
    
#     # Calculate statistics
#     total_records = len(file_data)
#     below_threshold = file_data[file_data['confidence_score'] < threshold]
#     below_count = len(below_threshold)
#     below_pct = (below_count / total_records * 100) if total_records > 0 else 0
    
#     mean_confidence = file_data['confidence_score'].mean()
#     median_confidence = file_data['confidence_score'].median()
#     min_confidence = file_data['confidence_score'].min()
#     max_confidence = file_data['confidence_score'].max()
#     std_confidence = file_data['confidence_score'].std()
    
#     # Calculate average percentile
#     avg_percentile = percentiles.mean()
    
#     # Determine if review is needed (more than 10% below threshold OR mean below threshold)
#     review_needed = below_pct > 10 or mean_confidence < threshold
    
#     # Calculate threshold percentile
#     threshold_percentile = norm.cdf(threshold, mu, sigma) * 100
    
#     return {
#         'filename': filename,
#         'mu': mu,
#         'sigma': sigma,
#         'threshold': threshold,
#         'threshold_percentile': threshold_percentile,
#         'total_records': total_records,
#         'mean_confidence': mean_confidence,
#         'median_confidence': median_confidence,
#         'std_confidence': std_confidence,
#         'min_confidence': min_confidence,
#         'max_confidence': max_confidence,
#         'below_threshold_count': below_count,
#         'below_threshold_pct': below_pct,
#         'avg_percentile': avg_percentile,
#         'review_needed': review_needed,
#         'data_with_percentiles': file_data_with_percentiles
#     }

# def plot_single_file_analysis(file_analysis: Dict):
#     """Create analysis plot for a single file"""
#     data = file_analysis['data_with_percentiles']
#     threshold = file_analysis['threshold']
#     mu = file_analysis['mu']
#     sigma = file_analysis['sigma']
#     filename = file_analysis['filename']
    
#     fig = make_subplots(
#         rows=2, cols=2,
#         subplot_titles=(
#             f'Distribution with Normal Fit',
#             f'Cumulative Distribution Function',
#             f'Below vs Above Threshold',
#             f'Percentile Distribution'
#         )
#     )
    
#     # Plot 1: Histogram with normal distribution overlay
#     fig.add_trace(
#         go.Histogram(
#             x=data['confidence_score'],
#             nbinsx=50,
#             name='Actual Data',
#             opacity=0.7,
#             histnorm='probability density',
#             marker_color='lightblue'
#         ),
#         row=1, col=1
#     )
    
#     x_range = np.linspace(data['confidence_score'].min(), data['confidence_score'].max(), 100)
#     normal_curve = norm.pdf(x_range, mu, sigma)
    
#     fig.add_trace(
#         go.Scatter(
#             x=x_range,
#             y=normal_curve,
#             mode='lines',
#             name='Normal Fit',
#             line=dict(color='red', width=2)
#         ),
#         row=1, col=1
#     )
    
#     fig.add_vline(
#         x=threshold,
#         line=dict(color='orange', width=2, dash='dash'),
#         annotation_text=f'Threshold: {threshold:.2f}',
#         row=1, col=1
#     )
    
#     # Plot 2: CDF
#     sorted_scores = np.sort(data['confidence_score'])
#     cdf_empirical = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
#     cdf_theoretical = norm.cdf(sorted_scores, mu, sigma)
    
#     fig.add_trace(
#         go.Scatter(
#             x=sorted_scores,
#             y=cdf_empirical,
#             mode='lines',
#             name='Empirical CDF',
#             line=dict(color='blue')
#         ),
#         row=1, col=2
#     )
    
#     fig.add_trace(
#         go.Scatter(
#             x=sorted_scores,
#             y=cdf_theoretical,
#             mode='lines',
#             name='Theoretical CDF',
#             line=dict(color='red', dash='dash')
#         ),
#         row=1, col=2
#     )
    
#     fig.add_vline(x=threshold, line=dict(color='orange', width=2, dash='dash'), row=1, col=2)
    
#     # Plot 3: Below vs Above threshold
#     below_threshold = data[data['confidence_score'] < threshold]['confidence_score']
#     above_threshold = data[data['confidence_score'] >= threshold]['confidence_score']
    
#     fig.add_trace(
#         go.Histogram(
#             x=below_threshold,
#             name='Below Threshold',
#             opacity=0.7,
#             nbinsx=30,
#             marker_color='red'
#         ),
#         row=2, col=1
#     )
    
#     fig.add_trace(
#         go.Histogram(
#             x=above_threshold,
#             name='Above Threshold',
#             opacity=0.7,
#             nbinsx=30,
#             marker_color='green'
#         ),
#         row=2, col=1
#     )
    
#     # Plot 4: Percentile distribution
#     fig.add_trace(
#         go.Histogram(
#             x=data['percentile'],
#             name='Percentile Distribution',
#             opacity=0.7,
#             nbinsx=20,
#             marker_color='purple'
#         ),
#         row=2, col=2
#     )
    
#     fig.update_layout(
#         height=700,
#         showlegend=True,
#         title_text=f"Analysis for: {filename}"
#     )
    
#     fig.update_xaxes(title_text="Confidence Score", row=1, col=1)
#     fig.update_xaxes(title_text="Confidence Score", row=1, col=2)
#     fig.update_xaxes(title_text="Confidence Score", row=2, col=1)
#     fig.update_xaxes(title_text="Percentile", row=2, col=2)
    
#     fig.update_yaxes(title_text="Density", row=1, col=1)
#     fig.update_yaxes(title_text="Cumulative Probability", row=1, col=2)
#     fig.update_yaxes(title_text="Frequency", row=2, col=1)
#     fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
#     return fig

# def main():
#     st.title("📊 Multi-File OCR Confidence Score Analyzer")
#     st.markdown("Analyze multiple OCR files **independently** with separate distributions, then get a summary of which files need review")
    
#     # Sidebar for controls
#     st.sidebar.header("Configuration")
    
#     # Threshold method selection
#     threshold_method = st.sidebar.selectbox(
#         "Threshold Calculation Method:",
#         ["Mean - 1 SD", "Mean - 2 SD", "25th Percentile", "Median", "Custom"],
#         help="How to calculate the threshold for each file independently"
#     )
    
#     custom_threshold = None
#     if threshold_method == "Custom":
#         custom_threshold = st.sidebar.number_input(
#             "Custom Threshold Value:",
#             min_value=0.0,
#             max_value=100.0,
#             value=70.0,
#             step=0.1
#         )
    
#     # Data input method
#     data_source = st.sidebar.selectbox(
#         "Choose data source:",
#         ["Upload Multiple Files", "Use Sample Data"]
#     )
    
#     all_file_data = []
#     file_names = []
    
#     if data_source == "Upload Multiple Files":
#         uploaded_files = st.sidebar.file_uploader(
#             "Choose CSV or JSON files",
#             type=["csv", "json"],
#             accept_multiple_files=True,
#             help="Upload multiple OCR result files for batch analysis"
#         )
        
#         if uploaded_files:
#             st.sidebar.info(f"📁 {len(uploaded_files)} file(s) uploaded")
            
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             for idx, uploaded_file in enumerate(uploaded_files):
#                 status_text.text(f"Processing {uploaded_file.name}...")
                
#                 if uploaded_file.name.endswith('.csv'):
#                     file_data = process_csv_file(uploaded_file, uploaded_file.name)
#                 elif uploaded_file.name.endswith('.json'):
#                     file_data = process_json_file(uploaded_file, uploaded_file.name)
#                 else:
#                     continue
                
#                 if file_data is not None:
#                     all_file_data.append(file_data)
#                     file_names.append(uploaded_file.name)
                
#                 progress_bar.progress((idx + 1) / len(uploaded_files))
            
#             status_text.empty()
#             progress_bar.empty()
            
#             if all_file_data:
#                 st.success(f"✅ Successfully processed {len(all_file_data)} file(s)")
#             else:
#                 st.error("❌ No valid data could be extracted from uploaded files")
#                 return
#         else:
#             st.info("👈 Please upload files to begin analysis")
#             return
    
#     else:  # Use sample data
#         st.info("📝 Using sample OCR data for demonstration (3 sample files)")
        
#         # Generate 3 sample files with different quality levels
#         np.random.seed(42)
        
#         # High quality file
#         high_quality = load_sample_data()
#         high_quality['confidence_score'] = np.clip(np.random.normal(90, 5, len(high_quality)), 0, 100)
#         high_quality['filename'] = 'report_high_quality.json'
        
#         # Medium quality file
#         medium_quality = load_sample_data()
#         medium_quality['confidence_score'] = np.clip(np.random.normal(75, 12, len(medium_quality)), 0, 100)
#         medium_quality['filename'] = 'report_medium_quality.json'
        
#         # Low quality file
#         low_quality = load_sample_data()
#         low_quality['confidence_score'] = np.clip(np.random.normal(60, 18, len(low_quality)), 0, 100)
#         low_quality['filename'] = 'report_low_quality.json'
        
#         all_file_data = [high_quality, medium_quality, low_quality]
#         file_names = ['report_high_quality.json', 'report_medium_quality.json', 'report_low_quality.json']
    
#     # Analyze each file independently
#     st.header("📈 Individual File Analysis")
#     st.markdown(f"**Threshold Method**: {threshold_method}")
#     if custom_threshold:
#         st.markdown(f"**Custom Threshold**: {custom_threshold}")
    
#     st.markdown("---")
    
#     file_analyses = []
    
#     for idx, file_data in enumerate(all_file_data):
#         filename = file_data['filename'].iloc[0]
        
#         st.markdown(f'<div class="file-section">', unsafe_allow_html=True)
#         st.subheader(f"📄 File {idx + 1}: {filename}")
        
#         # Analyze this file independently
#         file_analysis = analyze_single_file(file_data, filename, threshold_method, custom_threshold)
#         file_analyses.append(file_analysis)
        
#         # Display key metrics
#         col1, col2, col3, col4, col5 = st.columns(5)
        
#         with col1:
#             st.metric("Total Records", file_analysis['total_records'])
#         with col2:
#             st.metric("Mean (μ)", f"{file_analysis['mu']:.2f}")
#         with col3:
#             st.metric("Std Dev (σ)", f"{file_analysis['sigma']:.2f}")
#         with col4:
#             st.metric("Threshold", f"{file_analysis['threshold']:.2f}")
#         with col5:
#             review_status = "⚠️ YES" if file_analysis['review_needed'] else "✅ NO"
#             st.metric("Review Needed", review_status)
        
#         # More detailed metrics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("Median", f"{file_analysis['median_confidence']:.2f}")
#         with col2:
#             st.metric("Min / Max", f"{file_analysis['min_confidence']:.1f} / {file_analysis['max_confidence']:.1f}")
#         with col3:
#             st.metric(
#                 "Below Threshold",
#                 f"{file_analysis['below_threshold_count']} ({file_analysis['below_threshold_pct']:.1f}%)"
#             )
#         with col4:
#             st.metric(
#                 "Avg Percentile",
#                 f"{file_analysis['avg_percentile']:.1f}%",
#                 help="Average percentile based on this file's distribution"
#             )
        
#         # Show distribution plot
#         with st.expander(f"📊 View Detailed Analysis for {filename}", expanded=False):
#             fig = plot_single_file_analysis(file_analysis)
#             st.plotly_chart(fig, use_container_width=True)
            
#             # Show sample of records below threshold
#             data_with_percentiles = file_analysis['data_with_percentiles']
#             below_threshold_data = data_with_percentiles[
#                 data_with_percentiles['confidence_score'] < file_analysis['threshold']
#             ].sort_values('confidence_score')
            
#             if len(below_threshold_data) > 0:
#                 st.markdown("**🔍 Sample of Records Below Threshold:**")
#                 st.dataframe(
#                     below_threshold_data[['text', 'confidence_score', 'percentile']].head(20),
#                     use_container_width=True
#                 )
                
#                 # Download button for this file's low confidence records
#                 csv_buffer = io.StringIO()
#                 below_threshold_data.to_csv(csv_buffer, index=False)
                
#                 st.download_button(
#                     label=f"📥 Download Low Confidence Records",
#                     data=csv_buffer.getvalue(),
#                     file_name=f"low_confidence_{filename.replace('.', '_')}.csv",
#                     mime="text/csv",
#                     key=f"download_{idx}"
#                 )
#             else:
#                 st.success("✅ No records below threshold in this file")
        
#         st.markdown('</div>', unsafe_allow_html=True)
#         st.markdown("---")
    
#     # Summary Section
#     st.header("📋 Summary: Files Requiring Review")
    
#     # Create summary dataframe
#     summary_data = []
#     for analysis in file_analyses:
#         summary_data.append({
#             'Filename': analysis['filename'],
#             'Total Records': analysis['total_records'],
#             'Mean (μ)': analysis['mu'],
#             'Std Dev (σ)': analysis['sigma'],
#             'Threshold': analysis['threshold'],
#             'Threshold Percentile': analysis['threshold_percentile'],
#             'Mean Confidence': analysis['mean_confidence'],
#             'Below Threshold Count': analysis['below_threshold_count'],
#             'Below Threshold %': analysis['below_threshold_pct'],
#             'Avg Percentile': analysis['avg_percentile'],
#             'Review Required': '⚠️ YES' if analysis['review_needed'] else '✅ NO'
#         })
    
#     summary_df = pd.DataFrame(summary_data)
    
#     # Sort by review needed first, then by below threshold percentage
#     summary_df['review_sort'] = summary_df['Review Required'].map({'⚠️ YES': 0, '✅ NO': 1})
#     summary_df = summary_df.sort_values(['review_sort', 'Below Threshold %'], ascending=[True, False])
#     summary_df = summary_df.drop('review_sort', axis=1)
    
#     # Count files needing review
#     files_need_review = summary_df[summary_df['Review Required'] == '⚠️ YES']
#     files_ok = summary_df[summary_df['Review Required'] == '✅ NO']
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.metric(
#             "Total Files Analyzed",
#             len(all_file_data)
#         )
#     with col2:
#         st.metric(
#             "Files Requiring Review",
#             len(files_need_review),
#             delta=f"{len(files_need_review)/len(all_file_data)*100:.1f}% of total",
#             delta_color="inverse"
#         )
#     with col3:
#         st.metric(
#             "Files Passing Review",
#             len(files_ok),
#             delta=f"{len(files_ok)/len(all_file_data)*100:.1f}% of total"
#         )
    
#     # Display files requiring review
#     if len(files_need_review) > 0:
#         st.subheader("⚠️ Files That Require Review")
#         st.markdown('<div class="review-needed">', unsafe_allow_html=True)
        
#         review_display = files_need_review[[
#             'Filename', 'Total Records', 'Mean (μ)', 'Std Dev (σ)', 
#             'Threshold', 'Mean Confidence', 'Below Threshold Count', 
#             'Below Threshold %', 'Avg Percentile'
#         ]].copy()
        
#         st.dataframe(
#             review_display.style.format({
#                 'Mean (μ)': '{:.2f}',
#                 'Std Dev (σ)': '{:.2f}',
#                 'Threshold': '{:.2f}',
#                 'Mean Confidence': '{:.2f}',
#                 'Below Threshold %': '{:.1f}%',
#                 'Avg Percentile': '{:.1f}%'
#             }).background_gradient(subset=['Below Threshold %'], cmap='Reds'),
#             use_container_width=True
#         )
#         st.markdown('</div>', unsafe_allow_html=True)
#     else:
#         st.success("✅ All files passed the quality threshold!")
    
#     # Display files passing review
#     if len(files_ok) > 0:
#         st.subheader("✅ Files Passing Quality Threshold")
#         st.markdown('<div class="review-ok">', unsafe_allow_html=True)
        
#         ok_display = files_ok[[
#             'Filename', 'Total Records', 'Mean (μ)', 'Std Dev (σ)', 
#             'Threshold', 'Mean Confidence', 'Below Threshold Count', 
#             'Below Threshold %', 'Avg Percentile'
#         ]].copy()
        
#         st.dataframe(
#             ok_display.style.format({
#                 'Mean (μ)': '{:.2f}',
#                 'Std Dev (σ)': '{:.2f}',
#                 'Threshold': '{:.2f}',
#                 'Mean Confidence': '{:.2f}',
#                 'Below Threshold %': '{:.1f}%',
#                 'Avg Percentile': '{:.1f}%'
#             }).background_gradient(subset=['Mean Confidence'], cmap='Greens'),
#             use_container_width=True
#         )
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     # Complete summary table
#     st.subheader("📊 Complete Summary Table")
    
#     st.dataframe(
#         summary_df.style.format({
#             'Mean (μ)': '{:.2f}',
#             'Std Dev (σ)': '{:.2f}',
#             'Threshold': '{:.2f}',
#             'Threshold Percentile': '{:.1f}%',
#             'Mean Confidence': '{:.2f}',
#             'Below Threshold %': '{:.1f}%',
#             'Avg Percentile': '{:.1f}%'
#         }),
#         use_container_width=True
#     )
    
#     # Download complete summary
#     csv_buffer = io.StringIO()
#     summary_df.to_csv(csv_buffer, index=False)
    
#     st.download_button(
#         label="📥 Download Complete Summary (CSV)",
#         data=csv_buffer.getvalue(),
#         file_name=f"ocr_analysis_summary_{threshold_method.replace(' ', '_')}.csv",
#         mime="text/csv"
#     )
    
#     # Key Insights
#     st.header("💡 Key Insights")
    
#     insights = [
#         f"**Analysis Method**: Each file analyzed independently with its own distribution",
#         f"**Threshold Method**: {threshold_method}",
#         f"**Files Analyzed**: {len(all_file_data)} files processed",
#         f"**Review Rate**: {len(files_need_review)} out of {len(all_file_data)} files require review ({len(files_need_review)/len(all_file_data)*100:.1f}%)",
#     ]
    
#     if len(files_need_review) > 0:
#         worst_file = files_need_review.iloc[0]
#         insights.append(
#             f"**Highest Priority**: '{worst_file['Filename']}' has {worst_file['Below Threshold %']:.1f}% records below its threshold"
#         )
#         insights.append(
#             f"**Recommendation**: Review files marked with ⚠️ YES, starting with highest 'Below Threshold %'"
#         )
#     else:
#         insights.append(
#             f"**Recommendation**: All files meet quality standards - no immediate review needed"
#         )
    
#     for insight in insights:
#         st.info(insight)
    
#     # Comparison visualization
#     st.header("📊 Comparison Across Files")
    
#     fig_comparison = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=('Mean Confidence by File', '% Below Threshold by File')
#     )
    
#     # Mean confidence comparison
#     fig_comparison.add_trace(
#         go.Bar(
#             x=summary_df['Filename'],
#             y=summary_df['Mean Confidence'],
#             name='Mean Confidence',
#             marker_color=['red' if x == '⚠️ YES' else 'green' for x in summary_df['Review Required']]
#         ),
#         row=1, col=1
#     )
    
#     # Below threshold percentage comparison
#     fig_comparison.add_trace(
#         go.Bar(
#             x=summary_df['Filename'],
#             y=summary_df['Below Threshold %'],
#             name='% Below Threshold',
#             marker_color=['red' if x == '⚠️ YES' else 'green' for x in summary_df['Review Required']]
#         ),
#         row=1, col=2
#     )
    
#     fig_comparison.update_layout(
#         height=500,
#         showlegend=False,
#         title_text="File Comparison"
#     )
    
#     fig_comparison.update_xaxes(title_text="File", row=1, col=1)
#     fig_comparison.update_xaxes(title_text="File", row=1, col=2)
#     fig_comparison.update_yaxes(title_text="Mean Confidence", row=1, col=1)
#     fig_comparison.update_yaxes(title_text="% Below Threshold", row=1, col=2)
    
#     st.plotly_chart(fig_comparison, use_container_width=True)

# if __name__ == "__main__":
#     main()


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

def analyze_file(data, filename, threshold):
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
    
    # Determine if review is needed (if more than 10% below threshold)
    results['review_needed'] = results['below_threshold_pct'] > 10
    
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
        
        # Common threshold slider
        st.sidebar.subheader("🎯 Common Threshold Configuration")
        threshold = st.sidebar.slider(
            "Set common confidence threshold:",
            min_value=float(global_min),
            max_value=float(global_max),
            value=float(global_mean - global_std),
            step=0.1,
            help="This threshold will be applied to all files"
        )
        
        st.sidebar.info(f"Threshold: {threshold:.2f}")
        
        # Analyze all files
        for filename, data in files_data.items():
            results = analyze_file(data, filename, threshold)
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
            f"Review is {'recommended' if selected_result['review_needed'] else 'not necessary'} for this file (>10% threshold)"
        ]
        
        for insight in insights:
            st.info(f"• {insight}")

if __name__ == "__main__":
    main()