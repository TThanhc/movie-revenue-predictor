import os
import time
import csv
import requests
import logging
import numpy as np
from io import BytesIO
from PIL import Image
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ==============================================================================
# CONFIGURATION (CONFIG)
# ==============================================================================
# Replace with your API Key or set it as an environment variable
API_KEY = os.getenv("TMDB_API_KEY", "YOUR_API_KEY_HERE")

OUTPUT_FILE = "data/movies_dataset_revenue.csv"
START_YEAR = 2000
END_YEAR = 2024
PAGES_PER_YEAR = 25   # 25 pages * 20 movies = 500 highest-revenue movies per year
MAX_WORKERS = 10      # Number of parallel workers
IMG_BASE_URL = "https://image.tmdb.org/t/p/w185" # Poster image URL (w185 - small size for faster processing)

# Setup logging for easy tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# IMAGE PROCESSING FUNCTIONS (POSTER FEATURES EXTRACTION)
# ==============================================================================
def extract_poster_features(img_url):
    """
    Load image from URL and extract color/brightness features.
    Return dictionary of features, or None if error.
    """
    try:
        response = requests.get(img_url, timeout=5)
        if response.status_code != 200:
            return None
        
        # Open image from bytes in RAM (no need to save to disk)
        img = Image.open(BytesIO(response.content))
        img = img.convert("RGB") # Ensure RGB color space
        
        # Resize to calculate faster (50x75 pixels)
        img_small = img.resize((50, 75)) 
        img_array = np.array(img_small)
        
        # 1. Calculate brightness and saturation
        # Convert from RGB to HSV for more accurate calculation
        img_hsv = img_small.convert("HSV")
        hsv_array = np.array(img_hsv)
        
        # Channel V (Value/Brightness) is index 2, S (Saturation) is index 1
        saturation = hsv_array[:, :, 1].mean()
        brightness = hsv_array[:, :, 2].mean()

        # 2. Find dominant color using K-Means Clustering
        pixels = img_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=1, n_init=10)
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_[0] # [R, G, B]
        
        return {
            "poster_brightness": round(brightness, 2),
            "poster_saturation": round(saturation, 2),
            "poster_dom_r": int(dominant_color[0]),
            "poster_dom_g": int(dominant_color[1]),
            "poster_dom_b": int(dominant_color[2])
        }

    except Exception as e:
        logging.debug(f"Error extracting poster features from {img_url}: {e}")
        return None

# ==============================================================================
# SAFE API CALL FUNCTION (ROBUST REQUEST)
# ==============================================================================
def safe_get(url, params=None, max_retries=5):
    """
    Send request with retry mechanism if network error or Rate Limit (429).
    """
    if params is None:
        params = {}
    params["api_key"] = API_KEY

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 429:  # Rate Limit
                retry_after = int(response.headers.get("Retry-After", 1))
                logging.warning(f"Rate limit hit. Sleeping {retry_after}s...")
                time.sleep(retry_after + 0.5)
                continue
            
            else:
                # Other 4xx, 5xx errors
                logging.error(f"Request failed: {response.status_code} - {url}")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Network error: {e}. Retrying {attempt+1}/{max_retries}...")
            time.sleep(1)
    
    return None

# ==============================================================================
# 1. GET MOVIE IDS BY YEAR (STRATEGY: YEARLY REVENUE)
# ==============================================================================
def fetch_movie_ids_by_year(year):
    """
    Get list of movie IDs in a year, sorted by revenue descending.
    Purpose: Only get movies with revenue data to train the model.
    """
    movie_ids = []
    base_url = "https://api.themoviedb.org/3/discover/movie"
    
    for page in range(1, PAGES_PER_YEAR + 1):
        params = {
            "primary_release_year": year,
            "sort_by": "revenue.desc",  # IMPORTANT: Prioritize movies with revenue data
            "page": page,
            "vote_count.gte": 10        # Filter out trash movies
        }
        data = safe_get(base_url, params)
        
        if not data or "results" not in data:
            break
            
        for item in data["results"]:
            movie_ids.append(item["id"])
            
        # Light delay between discover pages
        time.sleep(0.2)
        
    logging.info(f"Year {year}: Found {len(movie_ids)} potential movies.")
    return movie_ids

# ==============================================================================
# 2. GET MOVIE DETAILS (FEATURE ENGINEERING FOR REVENUE PREDICTION + POSTER FEATURES)
# ==============================================================================
def fetch_movie_details(movie_id):
    """
    Get all required features in a single request using 'append_to_response'.
    Also get poster features (brightness, saturation, dominant colors).
    """
    # Technique to batch fetch: credits (actors), keywords (keywords), release_dates (release dates)
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"append_to_response": "credits,keywords,release_dates"}
    
    data = safe_get(url, params)
    
    if not data:
        return None

    # --- Filter garbage data ---
    revenue = data.get("revenue", 0)
    budget = data.get("budget", 0)
    
    # --- Extract basic features ---
    
    # 1. Director & Cast (Top 5)
    credits = data.get("credits", {})
    directors = [m["name"] for m in credits.get("crew", []) if m["job"] == "Director"]
    cast = [m["name"] for m in credits.get("cast", [])[:5]] # Get top 5 star power
    
    # 2. Keywords (Critical for content-based)
    keywords = [k["name"] for k in data.get("keywords", {}).get("keywords", [])]
    
    # 3. Production information
    production_companies = [c["name"] for c in data.get("production_companies", [])]
    production_countries = [c["name"] for c in data.get("production_countries", [])]
    genres = [g["name"] for g in data.get("genres", [])]
    
    # 4. Official release date
    release_date = data.get("release_date", "")
    
    # Create basic dict
    result = {
        "id": data.get("id"),
        "title": data.get("title"),
        "release_date": release_date,
        
        # Target Variables
        "budget": budget,
        "revenue": revenue,
        
        # Numeric Features
        "runtime": data.get("runtime"),
        "rating": data.get("vote_average"),
        "vote_count": data.get("vote_count"),
        "popularity": data.get("popularity"),
        
        # Categorical / Text Features
        "genres": ", ".join(genres),
        "production_companies": ", ".join(production_companies),
        "production_countries": ", ".join(production_countries),
        "director": ", ".join(directors),
        "cast": ", ".join(cast),
        "keywords": ", ".join(keywords),
        "original_language": data.get("original_language"),
        
        # Movie series (Harry Potter, Marvel...) greatly impact revenue
        "collection": data.get("belongs_to_collection", {}).get("name") if data.get("belongs_to_collection") else None
    }
    
    # --- Extract Poster Features ---
    poster_features = {
        "poster_brightness": np.nan,
        "poster_saturation": np.nan,
        "poster_dom_r": np.nan,
        "poster_dom_g": np.nan,
        "poster_dom_b": np.nan
    }
    
    poster_path = data.get("poster_path")
    if poster_path:
        full_url = IMG_BASE_URL + poster_path
        extracted = extract_poster_features(full_url)
        if extracted:
            poster_features = extracted
    
    # Merge poster features into result
    result.update(poster_features)
    
    return result

# ==============================================================================
# 3. CSV SAVING FUNCTION
# ==============================================================================

def save_to_csv(data_list, filename, mode='a'):
    if not data_list:
        return
    
    file_exists = os.path.isfile(filename)
    keys = data_list[0].keys()
    
    with open(filename, mode, newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists or mode == 'w':
            writer.writeheader()
        writer.writerows(data_list)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print(f"=== START CRAWLING DATA ({START_YEAR}-{END_YEAR}) ===")
    
    # Delete old file if you want to restart from beginning
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    total_collected = 0

    for year in range(START_YEAR, END_YEAR + 1):
        logging.info(f"--> Processing year: {year}")
        
        # B1: Get ID list
        movie_ids = fetch_movie_ids_by_year(year)
        if not movie_ids:
            continue
            
        year_data = []
        
        # B2: Get details with multithreading
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_id = {executor.submit(fetch_movie_details, mid): mid for mid in movie_ids}
            
            for future in as_completed(future_to_id):
                try:
                    result = future.result()
                    # Only get movies with Revenue > 0 (For better training)
                    if result and result['revenue'] > 0:
                        year_data.append(result)
                except Exception as e:
                    logging.error(f"Error fetching movie: {e}")

        # B3: Save immediately after each year (Checkpoint)
        if year_data:
            save_to_csv(year_data, OUTPUT_FILE, mode='a')
            count = len(year_data)
            total_collected += count
            logging.info(f"    Saved {count} movies of year {year}. Total: {total_collected}")
        
        # Rest a bit between years to allow API to breathe
        time.sleep(1)

    print(f"\n=== COMPLETE. TOTAL MOVIES: {total_collected} ===")
    print(f"Data file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()