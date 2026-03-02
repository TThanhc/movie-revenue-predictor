import streamlit as st
import pandas as pd
import numpy as np
from prediction_utils import MovieRevenuePredictor
from poster_analyzer import PosterAnalyzer
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="🎬 Movie Revenue Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-container h1 {
        margin-bottom: 10px;
        font-size: 2.5em;
    }
    
    /* Form sections */
    .form-section {
        background-color: white;
        padding: 25px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .section-title {
        color: #667eea;
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #667eea;
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
    }
    
    .result-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 15px 0;
    }
    
    .result-label {
        font-size: 1em;
        opacity: 0.9;
        margin-bottom: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 12px 30px !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: bold !important;
        font-size: 1.1em !important;
        width: 100% !important;
        transition: transform 0.2s !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.02) !important;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 15px;
    }
    
    .example-box {
        background-color: #f0f2f6;
        padding: 12px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.9em;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = MovieRevenuePredictor(model_dir="models")
if 'poster_analyzer' not in st.session_state:
    st.session_state.poster_analyzer = PosterAnalyzer()

predictor = st.session_state.predictor
poster_analyzer = st.session_state.poster_analyzer

# Header
st.markdown("""
    <div class="header-container">
        <h1>🎬 Movie Revenue Predictor</h1>
        <p>Enter your movie details in their original format - we'll handle the rest!</p>
    </div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "ℹ️ How It Works", "💡 Examples"])

with tab1:
    st.markdown("""
        <div class="info-box">
            <strong>💡 Tip:</strong> Enter your movie information as you would find it in a database. 
            Use comma-separated values for lists (genres, cast, etc.). Upload a poster image for automatic visual feature extraction!
        </div>
    """, unsafe_allow_html=True)
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="form-section"><div class="section-title">🎬 Basic Movie Information</div>', unsafe_allow_html=True)
        
        title = st.text_input(
            "🎥 Movie Title",
            value="Moana 2",
            help="The title of your movie"
        )
        
        release_date = st.date_input(
            "📅 Release Date",
            value=datetime(2024, 11, 21),
            help="When will the movie be released?"
        )
        
        col_budget, col_runtime = st.columns(2)
        
        with col_budget:
            budget = st.number_input(
                "💰 Budget (USD)",
                min_value=100_000,
                max_value=500_000_000,
                value=150_000_000,
                step=1_000_000,
                help="Total production budget"
            )
        
        with col_runtime:
            runtime = st.number_input(
                "⏱️ Runtime (minutes)",
                min_value=60,
                max_value=240,
                value=100,
                step=1,
                help="Movie length in minutes"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Creative details
        st.markdown('<div class="form-section"><div class="section-title">🎭 Creative Details</div>', unsafe_allow_html=True)
        
        genres = st.text_input(
            "🎨 Genres (comma-separated)",
            value="Family, Comedy, Adventure, Animation, Fantasy",
            help="List all genres separated by commas"
        )
        st.markdown('<div class="example-box">Example: Action, Drama, Thriller</div>', unsafe_allow_html=True)
        
        cast = st.text_area(
            "👥 Cast (comma-separated)",
            value="Auli'i Cravalho, Dwayne Johnson, Hualalai Chung, Rose Matafeo, David Fane",
            height=80,
            help="Main cast members separated by commas"
        )
        st.markdown('<div class="example-box">Example: Tom Hanks, Meryl Streep, Brad Pitt</div>', unsafe_allow_html=True)
        
        director = st.text_input(
            "🎥 Directors (comma-separated)",
            value="David G. Derrick Jr., Jason Hand, Dana Ledoux Miller",
            help="Director(s) separated by commas"
        )
        st.markdown('<div class="example-box">Example: Christopher Nolan</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="form-section"><div class="section-title">🎨 Poster Image</div>', unsafe_allow_html=True)
        
        uploaded_poster = st.file_uploader(
            "Upload Movie Poster",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload poster image for automatic visual feature extraction"
        )
        
        if uploaded_poster:
            # Display uploaded image
            image = Image.open(uploaded_poster)
            st.image(image, caption="Uploaded Poster", width='stretch')
            
            # Analyze poster
            with st.spinner("Analyzing poster..."):
                poster_features = poster_analyzer.analyze_image(uploaded_poster)
            
            st.success("✓ Poster analyzed!")
            
            # Show extracted features
            with st.expander("📊 Extracted Visual Features"):
                st.write(f"Brightness: {poster_features['poster_brightness']:.1f}")
                st.write(f"Saturation: {poster_features['poster_saturation']:.1f}")
                st.write(f"Dominant Color RGB: ({poster_features['poster_dom_r']:.0f}, {poster_features['poster_dom_g']:.0f}, {poster_features['poster_dom_b']:.0f})")
                
                # Show dominant color
                dom_color = f"rgb({poster_features['poster_dom_r']:.0f}, {poster_features['poster_dom_g']:.0f}, {poster_features['poster_dom_b']:.0f})"
                st.markdown(f'<div style="width: 100%; height: 40px; background-color: {dom_color}; border-radius: 5px; margin-top: 10px;"></div>', unsafe_allow_html=True)
        else:
            st.info("📤 No poster uploaded. Default visual features will be used.")
            poster_features = poster_analyzer.get_default_features()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Expected rating
        st.markdown('<div class="form-section"><div class="section-title">⭐ Rating</div>', unsafe_allow_html=True)
        
        rating = st.slider(
            "Expected IMDb Rating",
            min_value=1.0,
            max_value=10.0,
            value=7.0,
            step=0.1,
            help="What rating do you expect this movie to receive?"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Production details
    st.markdown('<div class="form-section"><div class="section-title">🏢 Production Information</div>', unsafe_allow_html=True)
    
    col_prod1, col_prod2 = st.columns(2)
    
    with col_prod1:
        production_companies = st.text_input(
            "🏢 Production Companies (comma-separated)",
            value="Walt Disney Animation Studios",
            help="Companies involved in production"
        )
        st.markdown('<div class="example-box">Example: Warner Bros., Legendary Pictures</div>', unsafe_allow_html=True)
    
    with col_prod2:
        production_countries = st.text_input(
            "🌍 Production Countries (comma-separated)",
            value="United States of America",
            help="Countries where the movie was produced"
        )
        st.markdown('<div class="example-box">Example: United States, United Kingdom</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional details
    st.markdown('<div class="form-section"><div class="section-title">🏷️ Additional Details</div>', unsafe_allow_html=True)
    
    col_add1, col_add2 = st.columns(2)
    
    with col_add1:
        keywords = st.text_input(
            "🔑 Keywords (comma-separated)",
            value="sea, ocean, villain, musical, sequel, coming of age, hopeful",
            help="Keywords describing the movie's themes and content"
        )
        st.markdown('<div class="example-box">Example: superhero, revenge, redemption</div>', unsafe_allow_html=True)
    
    with col_add2:
        collection = st.text_input(
            "📚 Collection/Franchise",
            value="Moana Collection",
            help="Leave empty if not part of a franchise, otherwise enter the franchise name"
        )
        st.markdown('<div class="example-box">Example: Marvel Cinematic Universe</div>', unsafe_allow_html=True)
    
    col_lang, col_model = st.columns(2)
    
    with col_lang:
        original_language = st.selectbox(
            "🗣️ Original Language",
            options=["English", "Spanish", "French", "Mandarin", "Japanese", "Korean", "Other"],
            index=0,
            help="Primary language of the movie"
        )
    
    with col_model:
        model_choice = st.selectbox(
            "🤖 Prediction Model",
            options=list(predictor.models.keys()),
            index=0,
            help="Select which trained model to use"
        )
    
    show_all_models = st.checkbox(
        "📊 Compare all models",
        value=False,
        help="Show predictions from all available models"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔮 Predict Revenue", width='stretch')
    
    if predict_button:
        # Prepare raw input
        raw_input = {
            'title': title,
            'release_date': release_date.strftime('%Y-%m-%d'),
            'budget': budget,
            'runtime': runtime,
            'genres': genres,
            'cast': cast,
            'director': director,
            'production_companies': production_companies,
            'production_countries': production_countries,
            'keywords': keywords,
            'collection': collection,
            'original_language': original_language.lower() if original_language != "Other" else "",
            'rating': rating,
            **poster_features  # Add poster features from analysis
        }
        
        # Make predictions
        with st.spinner("🎬 Analyzing your movie and predicting revenue..."):
            if show_all_models:
                st.markdown("---")
                st.subheader("📊 Predictions from All Models")
                
                all_predictions = predictor.predict_all_models_from_raw(raw_input)
                
                # Create comparison visualization
                col_vis1, col_vis2 = st.columns([2, 1])
                
                with col_vis1:
                    # Bar chart of predictions
                    models_list = list(all_predictions.keys())
                    values = [all_predictions[m] for m in models_list]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=models_list,
                            y=values,
                            marker=dict(
                                color=['#667eea', '#764ba2', '#f093fb', '#4facfe'],
                                line=dict(color='white', width=2)
                            ),
                            text=[f'${v:,.0f}' for v in values],
                            textposition='outside',
                        )
                    ])
                    
                    fig.update_layout(
                        title=f"Revenue Predictions for '{title}'",
                        yaxis_title="Predicted Revenue ($)",
                        xaxis_title="Model",
                        hovermode='x unified',
                        template='plotly_white',
                        height=400,
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                
                with col_vis2:
                    # Average prediction
                    avg_prediction = np.mean(values)
                    st.markdown(f"""
                        <div class="result-card">
                            <div class="result-label">Average Prediction</div>
                            <div class="result-value">${avg_prediction:,.0f}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            else:
                # Single model prediction
                prediction = predictor.predict_from_raw(raw_input, model_choice)
                revenue = prediction['revenue_actual']
                
                st.markdown("---")
                st.subheader(f"🎯 Prediction for '{title}'")
                
                col_result1, col_result2 = st.columns([1, 1])
                
                with col_result1:
                    st.markdown(f"""
                        <div class="result-card">
                            <div class="result-label">Predicted Revenue</div>
                            <div class="result-value">${revenue:,.0f}</div>
                            <div style="font-size: 0.9em; opacity: 0.8; margin-top: 10px;">
                                Model: {prediction['model_used']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col_result2:
                    # Financial insights
                    roi = ((revenue - budget) / budget) * 100
                    profit = revenue - budget
                    
                    roi_color = "#4CAF50" if roi > 0 else "#f44336"
                    
                    st.markdown(f"""
                        <div class="result-card">
                            <div class="result-label">Estimated ROI</div>
                            <div class="result-value" style="color: {roi_color};">{roi:+.1f}%</div>
                            <div style="font-size: 0.9em; opacity: 0.8; margin-top: 10px;">
                                {'Profit' if profit > 0 else 'Loss'}: ${abs(profit):,.0f}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
        
        # Movie summary
        st.markdown("---")
        st.subheader("📋 Movie Input Summary")
        
        # Parse input to show counts
        parsed = predictor.parse_raw_input(raw_input)
        
        summary_data = {
            '📊 Category': [
                '🎬 Title',
                '💰 Budget',
                '⏱️ Runtime',
                '📅 Release Date',
                '🎨 Genres',
                '👥 Cast Members',
                '🎥 Directors',
                '🏢 Production Companies',
                '🌍 Production Countries',
                '🏷️ Keywords',
                '⭐ Expected Rating',
                '📚 Franchise',
                '🗣️ Language',
            ],
            '💬 Details': [
                title,
                f'${budget:,}',
                f'{runtime} minutes',
                release_date.strftime('%B %d, %Y'),
                f'{parsed["num_genres"]} genres: {genres[:50]}...' if len(genres) > 50 else genres,
                f'{parsed["num_cast"]} cast members',
                f'{parsed["num_directors"]} director(s)',
                f'{parsed["num_production_companies"]} companies',
                f'{parsed["num_production_countries"]} countries',
                f'{parsed["num_keywords"]} keywords',
                f'{rating}/10',
                '✅ Yes' if collection else '❌ No',
                '🇺🇸 English' if parsed['is_english'] else '🌍 Other'
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width='stretch', hide_index=True)


with tab2:
    st.markdown("""
        <div class="form-section">
        <h2>How Does This Work?</h2>
        
        <h3>📝 Input Format</h3>
        <p>This app accepts movie data in its <strong>original, raw format</strong> - just like you'd find in a database or spreadsheet:</p>
        <ul>
            <li><strong>Text fields:</strong> Movie title, collection name</li>
            <li><strong>Comma-separated lists:</strong> Genres, cast, directors, companies, countries, keywords</li>
            <li><strong>Numbers:</strong> Budget (in USD), runtime (in minutes)</li>
            <li><strong>Date:</strong> Release date (calendar picker)</li>
            <li><strong>Image:</strong> Poster upload for automatic visual feature extraction</li>
        </ul>
        
        <h3>🖼️ Poster Image Analysis</h3>
        <p>When you upload a poster image, the app automatically extracts:</p>
        <ul>
            <li><strong>Brightness:</strong> Overall lightness of the poster</li>
            <li><strong>Saturation:</strong> Color intensity and vibrancy</li>
            <li><strong>Dominant Color (RGB):</strong> The primary color scheme</li>
        </ul>
        <p>These visual features correlate with marketing appeal and audience attention!</p>
        
        <h3>⚙️ Feature Engineering</h3>
        <p>The app automatically:</p>
        <ul>
            <li>Counts items in comma-separated lists (number of genres, cast, etc.)</li>
            <li>Extracts date components (year, month, season)</li>
            <li>Detects specific genres (Action, Drama, etc.)</li>
            <li>Identifies English-language films</li>
            <li>Determines franchise/collection status</li>
            <li>Calculates budget-per-minute efficiency</li>
            <li>Creates seasonal indicators (summer release, holidays)</li>
        </ul>
        
        <h3>🧠 Machine Learning Models</h3>
        <p>Choose from 4 trained models:</p>
        <ul>
            <li><strong>Best Model:</strong> Top performer selected from validation</li>
            <li><strong>XGBoost:</strong> Gradient boosting for complex patterns</li>
            <li><strong>LightGBM:</strong> Fast and efficient boosting algorithm</li>
            <li><strong>Random Forest:</strong> Robust ensemble method</li>
        </ul>
        
        <h3>📊 What Gets Analyzed</h3>
        <p>The models consider 31+ features including:</p>
        <ul>
            <li>Budget and production scale</li>
            <li>Team size (cast, directors)</li>
            <li>Genre mix and themes</li>
            <li>Release timing (season, month, year)</li>
            <li>Franchise status</li>
            <li>Language and market reach</li>
            <li>Poster visual appeal</li>
            <li>Keywords and themes</li>
        </ul>
        
        <h3>🎯 Model Accuracy</h3>
        <ul>
            <li>R² Score: ~0.59 (explains 59% of revenue variance)</li>
            <li>Trained on thousands of movies (2000-2024)</li>
            <li>Minimal overfitting with strong generalization</li>
        </ul>
        
        </div>
    """, unsafe_allow_html=True)


with tab3:
    st.markdown("""
        <div class="form-section">
        <h2>💡 Example Movie Inputs</h2>
        
        <h3>🎬 Example 1: Big Budget Sequel (Moana 2)</h3>
        <p><strong>Expected Revenue: ~$500M - $800M</strong></p>
        </div>
        
        <div class="form-section">
        <strong>Title:</strong> Moana 2<br>
        <strong>Release Date:</strong> 2024-11-21<br>
        <strong>Budget:</strong> $150,000,000<br>
        <strong>Runtime:</strong> 100 minutes<br>
        <strong>Genres:</strong> Family, Comedy, Adventure, Animation, Fantasy<br>
        <strong>Cast:</strong> Auli'i Cravalho, Dwayne Johnson, Hualalai Chung, Rose Matafeo, David Fane<br>
        <strong>Director:</strong> David G. Derrick Jr., Jason Hand, Dana Ledoux Miller<br>
        <strong>Production Companies:</strong> Walt Disney Animation Studios<br>
        <strong>Production Countries:</strong> United States of America<br>
        <strong>Keywords:</strong> sea, ocean, villain, musical, sequel, coming of age, hopeful<br>
        <strong>Collection:</strong> Moana Collection<br>
        <strong>Language:</strong> English
        </div>
        
        <div class="form-section">
        <h3>🎭 Example 2: Mid-Budget Drama</h3>
        <p><strong>Expected Revenue: ~$80M - $150M</strong></p>
        </div>
        
        <div class="form-section">
        <strong>Title:</strong> The Last Dance<br>
        <strong>Release Date:</strong> 2024-11-15<br>
        <strong>Budget:</strong> $35,000,000<br>
        <strong>Runtime:</strong> 125 minutes<br>
        <strong>Genres:</strong> Drama, Romance<br>
        <strong>Cast:</strong> Emma Stone, Ryan Gosling, Olivia Colman, Willem Dafoe<br>
        <strong>Director:</strong> Damien Chazelle<br>
        <strong>Production Companies:</strong> Lionsgate, Plan B Entertainment<br>
        <strong>Production Countries:</strong> United States of America<br>
        <strong>Keywords:</strong> love, music, rivalry, redemption, emotional<br>
        <strong>Collection:</strong> <em>(leave empty)</em><br>
        <strong>Language:</strong> English
        </div>
        
        <div class="form-section">
        <h3>🦸 Example 3: Superhero Blockbuster</h3>
        <p><strong>Expected Revenue: ~$800M - $1.2B</strong></p>
        </div>
        
        <div class="form-section">
        <strong>Title:</strong> Avengers: New Dawn<br>
        <strong>Release Date:</strong> 2025-06-15<br>
        <strong>Budget:</strong> $350,000,000<br>
        <strong>Runtime:</strong> 165 minutes<br>
        <strong>Genres:</strong> Action, Adventure, Science Fiction<br>
        <strong>Cast:</strong> Chris Hemsworth, Brie Larson, Tom Holland, Zendaya, Anthony Mackie, Florence Pugh<br>
        <strong>Director:</strong> Ryan Coogler<br>
        <strong>Production Companies:</strong> Marvel Studios, Walt Disney Pictures<br>
        <strong>Production Countries:</strong> United States of America<br>
        <strong>Keywords:</strong> superhero, team, villain, multiverse, action, sequel, franchise<br>
        <strong>Collection:</strong> Marvel Cinematic Universe<br>
        <strong>Language:</strong> English
        </div>
        
        <div class="form-section">
        <h3>😄 Example 4: Comedy</h3>
        <p><strong>Expected Revenue: ~$50M - $120M</strong></p>
        </div>
        
        <div class="form-section">
        <strong>Title:</strong> Wedding Crashers 2<br>
        <strong>Release Date:</strong> 2024-07-12<br>
        <strong>Budget:</strong> $45,000,000<br>
        <strong>Runtime:</strong> 105 minutes<br>
        <strong>Genres:</strong> Comedy, Romance<br>
        <strong>Cast:</strong> Owen Wilson, Vince Vaughn, Rachel McAdams, Isla Fisher<br>
        <strong>Director:</strong> David Dobkin<br>
        <strong>Production Companies:</strong> New Line Cinema, Warner Bros.<br>
        <strong>Production Countries:</strong> United States of America<br>
        <strong>Keywords:</strong> wedding, comedy, friendship, romance, sequel<br>
        <strong>Collection:</strong> Wedding Crashers Collection<br>
        <strong>Language:</strong> English
        </div>
        
        <div class="form-section">
        <h3>💡 Tips for Best Results</h3>
        <ul>
            <li><strong>Be specific with genres:</strong> List all that apply</li>
            <li><strong>Include notable cast:</strong> Well-known actors help predictions</li>
            <li><strong>Upload a poster:</strong> Visual features matter!</li>
            <li><strong>Consider timing:</strong> Summer and holiday releases often perform better</li>
            <li><strong>Franchise matters:</strong> Sequels typically have higher potential</li>
            <li><strong>Keywords help:</strong> Descriptive keywords improve accuracy</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 20px; color: #666;">
        <p>🎬 Movie Revenue Predictor v2.0 | Raw Input Edition</p>
        <p style="font-size: 0.9em;">Powered by Machine Learning • Automatic Feature Engineering • Poster Analysis</p>
    </div>
""", unsafe_allow_html=True)
