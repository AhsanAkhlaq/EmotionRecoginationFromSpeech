import streamlit as st
import numpy as np
import librosa
from joblib import load
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from Utilities.utils import extract_features



#   Page Configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

#   Custom CSS for Professional UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .emotion-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar-content {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .audio-player {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .success-message {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .warning-message {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .emotion-emoji {
        font-size: 4rem;
        margin: 1rem 0;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .confidence-bar {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .file-details {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

#   Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'results' not in st.session_state:
    st.session_state.results = None

#  Load model and preprocessors (with error handling)
@st.cache_resource
def load_models():
    try:
        MODEL_PATH = "model/best_lstm_model.keras"
        SCALER_PATH = "./model/scaler.joblib"
        ENCODER_PATH = "model/encoder.joblib"
        
        model = load_model(MODEL_PATH)
        scaler = load(SCALER_PATH)
        encoder = load(ENCODER_PATH)
        
        return model, scaler, encoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

    
#   Emotion mapping with emojis and colors
emotion_mapping = {
    'angry': {'emoji': '😠', 'color': '#FF4444'},
    'disgust': {'emoji': '🤢', 'color': '#4CAF50'},
    'fear': {'emoji': '😨', 'color': '#FF9800'},
    'happy': {'emoji': '😊', 'color': '#4CAF50'},
    'neutral': {'emoji': '😐', 'color': '#9E9E9E'},
    'sad': {'emoji': '😢', 'color': '#2196F3'},
    'surprise': {'emoji': '😲', 'color': '#9C27B0'}
}

#   Visualization functions
def create_bar_chart(emotions, probabilities):
    """Create horizontal bar chart using matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by probability
    sorted_data = sorted(zip(emotions, probabilities), key=lambda x: x[1])
    sorted_emotions, sorted_probs = zip(*sorted_data)
    
    # Create color map
    colors = [emotion_mapping.get(emotion, {'color': '#666666'})['color'] for emotion in sorted_emotions]
    
    bars = ax.barh(range(len(sorted_emotions)), sorted_probs, color=colors, alpha=0.7)
    
    # Customize the plot
    ax.set_yticks(range(len(sorted_emotions)))
    ax.set_yticklabels([f"{emotion_mapping.get(e, {'emoji': '❓'})['emoji']} {e.capitalize()}" 
                        for e in sorted_emotions])
    ax.set_xlabel('Confidence Score')
    ax.set_title('Emotion Probability Distribution', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, sorted_probs)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1%}', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_radar_chart(emotions, probabilities):
    """Create radar chart using matplotlib"""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Calculate angles for each emotion
    angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()
    
    # Close the plot by adding the first value at the end
    probabilities = list(probabilities) + [probabilities[0]]
    angles += angles[:1]
    emotions_labels = list(emotions) + [emotions[0]]
    
    # Plot
    ax.plot(angles, probabilities, 'o-', linewidth=2, color='#667eea')
    ax.fill(angles, probabilities, color='#667eea', alpha=0.25)
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{emotion_mapping.get(e, {'emoji': '❓'})['emoji']} {e.capitalize()}" 
                        for e in emotions])
    ax.set_ylim(0, 1)
    ax.set_title('Emotion Confidence Radar', fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True)
    
    return fig

def create_donut_chart(emotions, probabilities):
    """Create donut chart for top emotions"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get top 5 emotions
    top_indices = np.argsort(probabilities)[-5:][::-1]
    top_emotions = [emotions[i] for i in top_indices]
    top_probs = [probabilities[i] for i in top_indices]
    
    # Colors for the chart
    colors = [emotion_mapping.get(emotion, {'color': '#666666'})['color'] for emotion in top_emotions]
    
    # Create donut chart
    wedges, texts, autotexts = ax.pie(top_probs, 
                                     labels=[f"{emotion_mapping.get(e, {'emoji': '❓'})['emoji']} {e.capitalize()}" 
                                            for e in top_emotions],
                                     colors=colors,
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     pctdistance=0.85)
    
    # Add a circle at the center to make it a donut
    centre_circle = Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    
    ax.set_title('Top 5 Emotions Distribution', fontsize=16, fontweight='bold')
    
    return fig

#   Main Application Header
st.markdown("""
<div class="main-header">
    <h1>🎤 Speech Emotion Recognition</h1>
    <p>AI-Powered Emotion Analysis from Speech Patterns</p>
    <p>Upload an audio file to discover the emotional tone using advanced deep learning</p>
</div>
""", unsafe_allow_html=True)

#   Load models
model, scaler, encoder = load_models()

if model is None:
    st.error("⚠️ Models could not be loaded. Please check your model paths.")
    st.stop()

#   Sidebar for additional controls and information
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h3>🎯 How it Works</h3>
        <p>Our AI analyzes audio features like:</p>
        <ul>
            <li>🎵 Spectral characteristics</li>
            <li>🔊 Voice energy patterns</li>
            <li>📊 Frequency distributions</li>
            <li>⚡ Temporal dynamics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-content">
        <h3>📋 Supported Formats</h3>
        <p>• WAV files (recommended)</p>
        <p>• Duration: 1-3 seconds</p>
        <p>• Sample rate: 16kHz+</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Reset Analysis", help="Clear all results and start over"):
        st.session_state.analysis_complete = False
        st.session_state.audio_file = None
        st.session_state.results = None
        st.rerun()

#   Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="emotion-card">
        <h2>📁 Upload Audio File</h2>
        <p>Choose a clear audio recording with speech to analyze emotional content</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["wav"],
        help="Select an audio file containing speech for emotion analysis"
    )
    
    if uploaded_file is not None:
        st.session_state.audio_file = uploaded_file
        
        # Display file information
        file_details = f"""
        <div class="file-details">
            <h4>📄 File Information</h4>
            <p><strong>Name:</strong> {uploaded_file.name}</p>
            <p><strong>Size:</strong> {uploaded_file.size / 1024:.1f} KB</p>
            <p><strong>Type:</strong> {uploaded_file.type}</p>
        </div>
        """
        st.markdown(file_details, unsafe_allow_html=True)
        
        # Audio player
        st.markdown("""
        <div class="audio-player">
            <h4>🎵 Audio Preview</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.audio(uploaded_file, format='audio/wav')
        
        # Analysis button
        if st.button("🧠 Analyze Emotion", type="primary", help="Process the audio and predict emotion"):
            with st.spinner("🔍 Analyzing audio features..."):
                progress_bar = st.progress(0)
                
                try:
                    # Load audio with progress updates
                    progress_bar.progress(20)
                    y, sr = librosa.load(uploaded_file, duration=2.5, offset=0.6)
                    
                    progress_bar.progress(40)
                    # Extract features
                    features = extract_features(y, sr).reshape(1, -1)
                    
                    progress_bar.progress(60)
                    # Scale features
                    features = scaler.transform(features)
                    
                    progress_bar.progress(80)
                    # Add dimension for LSTM input
                    features = np.expand_dims(features, axis=2)
                    
                    # Predict emotion
                    prediction = model.predict(features)
                    predicted_class = np.argmax(prediction)
                    predicted_label = encoder.categories_[0][predicted_class]
                    
                    progress_bar.progress(100)
                    
                    # Store results
                    st.session_state.results = {
                        'predicted_emotion': predicted_label,
                        'probabilities': prediction[0],
                        'emotions': encoder.categories_[0]
                    }
                    st.session_state.analysis_complete = True
                    
                    # Success message
                    st.markdown("""
                    <div class="success-message">
                          Analysis Complete! Results are ready.
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    progress_bar.empty()

with col2:
    if st.session_state.analysis_complete and st.session_state.results:
        results = st.session_state.results
        predicted_emotion = results['predicted_emotion']
        probabilities = results['probabilities']
        emotions = results['emotions']
        
        # Main result display
        emotion_info = emotion_mapping.get(predicted_emotion, {'emoji': '❓', 'color': '#666666'})
        
        result_html = f"""
        <div class="result-card">
            <div class="emotion-emoji">{emotion_info['emoji']}</div>
            <h2>Predicted Emotion</h2>
            <h1 style="font-size: 2.5rem; margin: 1rem 0;">{predicted_emotion.upper()}</h1>
            <p style="font-size: 1.2rem;">Confidence: {probabilities[np.argmax(probabilities)]:.1%}</p>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)
        
        # Confidence metrics
        st.markdown("""
        <div class="emotion-card">
            <h3>🎯 Confidence Scores</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create confidence bars
        for emotion, prob in zip(emotions, probabilities):
            emotion_info = emotion_mapping.get(emotion, {'emoji': '❓', 'color': '#666666'})
            
            col_emoji, col_name, col_bar, col_percent = st.columns([1, 2, 4, 1])
            
            with col_emoji:
                st.markdown(f"<div style='font-size: 2rem; text-align: center;'>{emotion_info['emoji']}</div>", unsafe_allow_html=True)
            
            with col_name:
                st.markdown(f"**{emotion.capitalize()}**")
            
            with col_bar:
                st.progress(float(prob))
            
            with col_percent:
                st.markdown(f"**{prob:.1%}**")

#   Results visualization (full width)
if st.session_state.analysis_complete and st.session_state.results:
    st.markdown("---")
    st.markdown("## 📊 Detailed Analysis")
    
    results = st.session_state.results
    emotions = results['emotions']
    probabilities = results['probabilities']
    
    # Create visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Probability Distribution", "🎯 Confidence Radar", "🍩 Top Emotions", "📈 Emotion Ranking"])
    
    with tab1:
        st.markdown("### Horizontal Bar Chart")
        fig = create_bar_chart(emotions, probabilities)
        st.pyplot(fig)
        plt.close(fig)
    
    with tab2:
        st.markdown("### Radar Chart")
        fig = create_radar_chart(emotions, probabilities)
        st.pyplot(fig)
        plt.close(fig)
    
    with tab3:
        st.markdown("### Donut Chart - Top 5 Emotions")
        fig = create_donut_chart(emotions, probabilities)
        st.pyplot(fig)
        plt.close(fig)
    
    with tab4:
        # Ranking table
        st.markdown("### Emotion Ranking Table")
        df_ranking = pd.DataFrame({
            'Rank': range(1, len(emotions) + 1),
            'Emotion': emotions,
            'Confidence': probabilities,
            'Emoji': [emotion_mapping.get(e, {'emoji': '❓'})['emoji'] for e in emotions]
        }).sort_values('Confidence', ascending=False).reset_index(drop=True)
        
        df_ranking['Rank'] = range(1, len(df_ranking) + 1)
        df_ranking['Confidence'] = df_ranking['Confidence'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            df_ranking,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("🏆 Rank", width="small"),
                "Emoji": st.column_config.TextColumn("", width="small"),
                "Emotion": st.column_config.TextColumn("Emotion", width="medium"),
                "Confidence": st.column_config.TextColumn("Confidence", width="medium")
            }
        )
        
        # Additional metrics
        st.markdown("### 📈 Statistical Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Highest Confidence", f"{np.max(probabilities):.1%}")
        
        with col2:
            st.metric("Lowest Confidence", f"{np.min(probabilities):.1%}")
        
        with col3:
            st.metric("Average Confidence", f"{np.mean(probabilities):.1%}")
        
        with col4:
            st.metric("Std. Deviation", f"{np.std(probabilities):.1%}")

#   Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🧠 Powered by Deep Learning • 🎤 Speech Emotion Recognition • 🚀 Built with Streamlit</p>
    <p>Upload clear audio recordings for best results</p>
</div>
""", unsafe_allow_html=True)