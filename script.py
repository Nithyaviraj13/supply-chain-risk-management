import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
import warnings
import sys
import io
import heapq
from collections import defaultdict
import requests

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="India Supply Chain Intelligence",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== API CONFIGURATION ====================
GNEWS_API_KEY = "8d4dd89ea3a4943b8b85f881ea4ee57b"
NEWSAPI_KEY = "06d67138eb1d4943ae51cd8c527d95a3"
WEATHER_API_KEY = "7cb31ec50a0bb82a328c4f3e3fbe9176"

GNEWS_BASE_URL = "https://gnews.io/api/v4"
NEWSAPI_BASE_URL = "https://newsapi.org/v2"
WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

CITY_COORDINATES = {
    'Mumbai Port': 'Mumbai', 'JNPT Mumbai': 'Mumbai', 'Chennai Port': 'Chennai',
    'Kolkata Port': 'Kolkata', 'Visakhapatnam Port': 'Visakhapatnam', 'Cochin Port': 'Kochi',
    'Kandla Port': 'Kandla', 'Mangalore Port': 'Mangalore', 'Pune': 'Pune',
    'Ahmedabad': 'Ahmedabad', 'Coimbatore': 'Coimbatore', 'Gurgaon': 'Gurgaon',
    'Noida': 'Noida', 'Bangalore': 'Bangalore', 'Hyderabad': 'Hyderabad',
    'Surat': 'Surat', 'Faridabad': 'Faridabad', 'Rajkot': 'Rajkot',
    'Delhi NCR': 'Delhi', 'Mumbai Central': 'Mumbai', 'Chennai Central': 'Chennai',
    'Kolkata Central': 'Kolkata', 'Bangalore Central': 'Bangalore', 'Jaipur': 'Jaipur',
    'Lucknow': 'Lucknow', 'Chandigarh': 'Chandigarh', 'Indore': 'Indore',
    'Bhopal': 'Bhopal', 'Nagpur': 'Nagpur', 'Vadodara': 'Vadodara',
    'Nashik': 'Nashik', 'Aurangabad': 'Aurangabad', 'Vijayawada': 'Vijayawada',
    'Madurai': 'Madurai', 'Trichy': 'Trichy', 'Bhubaneswar': 'Bhubaneswar',
    'Guwahati': 'Guwahati', 'Patna': 'Patna', 'Ranchi': 'Ranchi',
    'Kakinada': 'Kakinada', 'Guntur': 'Guntur', 'Warangal': 'Warangal',
}

def get_weather_data(location):
    try:
        city_name = CITY_COORDINATES.get(location, location)
        params = {'q': city_name, 'appid': WEATHER_API_KEY, 'units': 'metric'}
        response = requests.get(f"{WEATHER_BASE_URL}/weather", params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {'success': True, 'data': {
            'temperature': data.get('main', {}).get('temp'),
            'humidity': data.get('main', {}).get('humidity'),
            'pressure': data.get('main', {}).get('pressure'),
            'wind_speed': data.get('wind', {}).get('speed'),
            'description': data.get('weather', [{}])[0].get('description'),
            'condition': data.get('weather', [{}])[0].get('main'),
            'city': data.get('name'),
            'country': data.get('sys', {}).get('country')
        }}
    except Exception as e:
        return {'success': False, 'error': f'❌ API Error: {str(e)}'}

def get_news_from_gnews(location):
    try:
        city_name = CITY_COORDINATES.get(location, location)
        params = {
            'q': f'{city_name} supply chain logistics disruption',
            'token': GNEWS_API_KEY, 'lang': 'en', 'max': 10, 'sortby': 'publishedAt'
        }
        response = requests.get(f"{GNEWS_BASE_URL}/search", params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('articles'):
            return {'success': True, 'source': 'GNews', 'articles': [
                {'title': article.get('title'), 'description': article.get('description'),
                 'source': article.get('source', {}).get('name'), 'url': article.get('url'),
                 'published_at': article.get('publishedAt')} for article in data.get('articles', [])[:5]
            ]}
        return {'success': False, 'error': '❌ No articles from GNews', 'articles': []}
    except Exception as e:
        return {'success': False, 'error': f'❌ GNews Error: {str(e)}', 'articles': []}

def get_news_from_newsapi(location):
    try:
        city_name = CITY_COORDINATES.get(location, location)
        params = {
            'q': f'{city_name} supply chain logistics',
            'apiKey': NEWSAPI_KEY, 'language': 'en', 'sortBy': 'publishedAt', 'pageSize': 10
        }
        response = requests.get(f"{NEWSAPI_BASE_URL}/everything", params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('articles'):
            return {'success': True, 'source': 'NewsAPI', 'articles': [
                {'title': article.get('title'), 'description': article.get('description'),
                 'source': article.get('source', {}).get('name'), 'url': article.get('url'),
                 'published_at': article.get('publishedAt')} for article in data.get('articles', [])[:5]
            ]}
        return {'success': False, 'error': '❌ No articles from NewsAPI', 'articles': []}
    except Exception as e:
        return {'success': False, 'error': f'❌ NewsAPI Error: {str(e)}', 'articles': []}

def predict_disruption_probability(weather_data, news_gnews, news_newsapi, cost_increase_pct, location):
    factors = {}
    weather_score, weather_reasons = 0, []
    if weather_data['success']:
        w = weather_data['data']
        if w['temperature'] > 40: weather_score += 0.25; weather_reasons.append(f"🔥 Heat: {w['temperature']}°C")
        elif w['temperature'] < 10: weather_score += 0.15; weather_reasons.append(f"❄️ Cold: {w['temperature']}°C")
        if w['wind_speed'] > 30: weather_score += 0.30; weather_reasons.append(f"💨 Wind: {w['wind_speed']} km/h")
        elif w['wind_speed'] > 20: weather_score += 0.15
        cond = w['condition'].lower()
        if 'rain' in cond: weather_score += 0.20; weather_reasons.append("🌧️ Rain")
        elif 'storm' in cond: weather_score += 0.35; weather_reasons.append("⛈️ Storm")
        elif 'fog' in cond: weather_score += 0.15; weather_reasons.append("🌫️ Fog")
        if w['humidity'] > 85: weather_score += 0.10
        weather_score = min(weather_score, 1.0)
        factors['Weather'] = {'score': weather_score, 'weight': 0.40, 'reasons': weather_reasons if weather_reasons else ['✅ Good weather']}
    else:
        factors['Weather'] = {'score': 0.2, 'weight': 0.40, 'reasons': ['⚠️ Data unavailable']}
    
    news_score, news_reasons = 0, []
    all_news = []
    if news_gnews['success']: all_news.extend(news_gnews['articles'])
    if news_newsapi['success']: all_news.extend(news_newsapi['articles'])
    if all_news:
        keywords = ['strike', 'protest', 'accident', 'traffic', 'disruption', 'closed', 'blocked', 'delayed']
        for article in all_news:
            text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            for kw in keywords:
                if kw in text: news_score += 0.08; news_reasons.append(f"📰 '{kw}' detected"); break
        news_score = min(news_score, 1.0)
    factors['News'] = {'score': news_score, 'weight': 0.35, 'reasons': news_reasons if news_reasons else ['✅ No alerts']}
    
    cost_score = min(cost_increase_pct / 100, 1.0)
    factors['Cost'] = {'score': cost_score, 'weight': 0.15, 'reasons': [f"💰 +{cost_increase_pct}% increase"]}
    
    region_risk = {'North': 0.3, 'West': 0.4, 'South': 0.35, 'East': 0.25, 'Central': 0.2, 'Northeast': 0.35}
    region = INDIAN_LOCATIONS.get(location, {}).get('region', 'Central')
    location_score = region_risk.get(region, 0.2)
    factors['Location'] = {'score': location_score, 'weight': 0.10, 'reasons': [f"📍 {region} region"]}
    
    total = sum(f['score'] * f['weight'] for f in factors.values())
    severity = 'Low' if total < 0.2 else 'Medium' if total < 0.4 else 'High' if total < 0.7 else 'Critical'
    emoji = {'Low': '🟢', 'Medium': '🟡', 'High': '🟠', 'Critical': '🔴'}[severity]
    return {'probability': total, 'severity': severity, 'emoji': emoji, 'factors': factors, 'location': location}

INDIAN_LOCATIONS = {
    'Mumbai Port': {'lat': 18.9583, 'lon': 72.8347, 'type': 'port', 'region': 'West', 'capacity': 95},
    'JNPT Mumbai': {'lat': 18.9485, 'lon': 72.9483, 'type': 'port', 'region': 'West', 'capacity': 100},
    'Chennai Port': {'lat': 13.0827, 'lon': 80.2707, 'type': 'port', 'region': 'South', 'capacity': 85},
    'Kolkata Port': {'lat': 22.5726, 'lon': 88.3639, 'type': 'port', 'region': 'East', 'capacity': 80},
    'Visakhapatnam Port': {'lat': 17.6868, 'lon': 83.2185, 'type': 'port', 'region': 'East', 'capacity': 75},
    'Cochin Port': {'lat': 9.9312, 'lon': 76.2673, 'type': 'port', 'region': 'South', 'capacity': 70},
    'Kandla Port': {'lat': 23.0330, 'lon': 70.2167, 'type': 'port', 'region': 'West', 'capacity': 65},
    'Mangalore Port': {'lat': 12.9141, 'lon': 74.8560, 'type': 'port', 'region': 'South', 'capacity': 60},
    'Pune': {'lat': 18.5204, 'lon': 73.8567, 'type': 'manufacturing', 'region': 'West', 'capacity': 90},
    'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'type': 'manufacturing', 'region': 'West', 'capacity': 85},
    'Coimbatore': {'lat': 11.0168, 'lon': 76.9558, 'type': 'manufacturing', 'region': 'South', 'capacity': 80},
    'Gurgaon': {'lat': 28.4595, 'lon': 77.0266, 'type': 'manufacturing', 'region': 'North', 'capacity': 95},
    'Noida': {'lat': 28.5355, 'lon': 77.3910, 'type': 'manufacturing', 'region': 'North', 'capacity': 90},
    'Bangalore': {'lat': 12.9716, 'lon': 77.5946, 'type': 'manufacturing', 'region': 'South', 'capacity': 100},
    'Hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'type': 'manufacturing', 'region': 'South', 'capacity': 95},
    'Surat': {'lat': 21.1702, 'lon': 72.8311, 'type': 'manufacturing', 'region': 'West', 'capacity': 85},
    'Faridabad': {'lat': 28.4089, 'lon': 77.3178, 'type': 'manufacturing', 'region': 'North', 'capacity': 80},
    'Rajkot': {'lat': 22.3039, 'lon': 70.8022, 'type': 'manufacturing', 'region': 'West', 'capacity': 75},
    'Delhi NCR': {'lat': 28.7041, 'lon': 77.1025, 'type': 'distribution', 'region': 'North', 'capacity': 100},
    'Mumbai Central': {'lat': 19.0760, 'lon': 72.8777, 'type': 'distribution', 'region': 'West', 'capacity': 100},
    'Chennai Central': {'lat': 13.0827, 'lon': 80.2707, 'type': 'distribution', 'region': 'South', 'capacity': 95},
    'Kolkata Central': {'lat': 22.5726, 'lon': 88.3639, 'type': 'distribution', 'region': 'East', 'capacity': 90},
    'Bangalore Central': {'lat': 12.9716, 'lon': 77.5946, 'type': 'distribution', 'region': 'South', 'capacity': 95},
    'Jaipur': {'lat': 26.9124, 'lon': 75.7873, 'type': 'warehouse', 'region': 'North', 'capacity': 85},
    'Lucknow': {'lat': 26.8467, 'lon': 80.9462, 'type': 'warehouse', 'region': 'North', 'capacity': 80},
    'Chandigarh': {'lat': 30.7333, 'lon': 76.7794, 'type': 'warehouse', 'region': 'North', 'capacity': 75},
    'Indore': {'lat': 22.7196, 'lon': 75.8577, 'type': 'warehouse', 'region': 'Central', 'capacity': 80},
    'Bhopal': {'lat': 23.2599, 'lon': 77.4126, 'type': 'warehouse', 'region': 'Central', 'capacity': 75},
    'Nagpur': {'lat': 21.1458, 'lon': 79.0882, 'type': 'warehouse', 'region': 'Central', 'capacity': 85},
    'Vadodara': {'lat': 22.3072, 'lon': 73.1812, 'type': 'warehouse', 'region': 'West', 'capacity': 75},
    'Nashik': {'lat': 19.9975, 'lon': 73.7898, 'type': 'warehouse', 'region': 'West', 'capacity': 70},
    'Aurangabad': {'lat': 19.8762, 'lon': 75.3433, 'type': 'warehouse', 'region': 'West', 'capacity': 70},
    'Vijayawada': {'lat': 16.5062, 'lon': 80.6480, 'type': 'warehouse', 'region': 'South', 'capacity': 75},
    'Madurai': {'lat': 9.9252, 'lon': 78.1198, 'type': 'warehouse', 'region': 'South', 'capacity': 70},
    'Trichy': {'lat': 10.7905, 'lon': 78.7047, 'type': 'warehouse', 'region': 'South', 'capacity': 70},
    'Bhubaneswar': {'lat': 20.2961, 'lon': 85.8245, 'type': 'warehouse', 'region': 'East', 'capacity': 75},
    'Guwahati': {'lat': 26.1445, 'lon': 91.7362, 'type': 'warehouse', 'region': 'Northeast', 'capacity': 70},
    'Patna': {'lat': 25.5941, 'lon': 85.1376, 'type': 'warehouse', 'region': 'East', 'capacity': 75},
    'Ranchi': {'lat': 23.3441, 'lon': 85.3096, 'type': 'warehouse', 'region': 'East', 'capacity': 70},
    'Kakinada': {'lat': 16.9891, 'lon': 82.2475, 'type': 'warehouse', 'region': 'South', 'capacity': 65},
    'Guntur': {'lat': 16.3067, 'lon': 80.4365, 'type': 'warehouse', 'region': 'South', 'capacity': 65},
    'Warangal': {'lat': 17.9689, 'lon': 79.5941, 'type': 'warehouse', 'region': 'South', 'capacity': 65},
}

ROUTE_NETWORK = {
    'Mumbai Port': {'Mumbai Central': {'distance': 15, 'time': 1.5, 'highway': 'Local'}, 'Pune': {'distance': 150, 'time': 3.5, 'highway': 'NH48'}, 'Nashik': {'distance': 165, 'time': 4, 'highway': 'NH160'}, 'Surat': {'distance': 265, 'time': 5, 'highway': 'NH48'}, 'Aurangabad': {'distance': 335, 'time': 6.5, 'highway': 'NH211'}},
    'JNPT Mumbai': {'Mumbai Central': {'distance': 50, 'time': 2, 'highway': 'NH348B'}, 'Pune': {'distance': 120, 'time': 3, 'highway': 'NH48'}, 'Mumbai Port': {'distance': 35, 'time': 1.5, 'highway': 'Coastal Road'}},
    'Mumbai Central': {'Delhi NCR': {'distance': 1400, 'time': 20, 'highway': 'NH48'}, 'Pune': {'distance': 150, 'time': 3.5, 'highway': 'NH48'}, 'Ahmedabad': {'distance': 525, 'time': 8, 'highway': 'NH48'}, 'Surat': {'distance': 265, 'time': 5, 'highway': 'NH48'}, 'Indore': {'distance': 580, 'time': 10, 'highway': 'NH52'}, 'Nagpur': {'distance': 785, 'time': 13, 'highway': 'NH160'}},
    'Pune': {'Mumbai Central': {'distance': 150, 'time': 3.5, 'highway': 'NH48'}, 'Bangalore': {'distance': 840, 'time': 14, 'highway': 'NH48'}, 'Hyderabad': {'distance': 560, 'time': 10, 'highway': 'NH65'}, 'Aurangabad': {'distance': 235, 'time': 5, 'highway': 'NH211'}, 'Nashik': {'distance': 210, 'time': 4.5, 'highway': 'NH60'}, 'Gurgaon': {'distance': 1450, 'time': 22, 'highway': 'NH48'}},
    'Delhi NCR': {'Gurgaon': {'distance': 30, 'time': 1, 'highway': 'NH48'}, 'Noida': {'distance': 25, 'time': 1, 'highway': 'DND Flyway'}, 'Jaipur': {'distance': 280, 'time': 5, 'highway': 'NH48'}, 'Chandigarh': {'distance': 245, 'time': 5, 'highway': 'NH44'}, 'Lucknow': {'distance': 555, 'time': 9, 'highway': 'NH24'}, 'Indore': {'distance': 715, 'time': 12, 'highway': 'NH48'}, 'Mumbai Central': {'distance': 1400, 'time': 20, 'highway': 'NH48'}},
    'Gurgaon': {'Delhi NCR': {'distance': 30, 'time': 1, 'highway': 'NH48'}, 'Noida': {'distance': 40, 'time': 1.5, 'highway': 'NH48'}, 'Jaipur': {'distance': 265, 'time': 5, 'highway': 'NH48'}, 'Faridabad': {'distance': 25, 'time': 1, 'highway': 'Local'}},
    'Noida': {'Delhi NCR': {'distance': 25, 'time': 1, 'highway': 'DND'}, 'Gurgaon': {'distance': 40, 'time': 1.5, 'highway': 'NH48'}, 'Lucknow': {'distance': 540, 'time': 8.5, 'highway': 'Yamuna Expressway'}},
    'Bangalore': {'Bangalore Central': {'distance': 10, 'time': 0.5, 'highway': 'Local'}, 'Chennai Central': {'distance': 350, 'time': 6, 'highway': 'NH48'}, 'Hyderabad': {'distance': 575, 'time': 10, 'highway': 'NH44'}, 'Coimbatore': {'distance': 360, 'time': 7, 'highway': 'NH209'}, 'Cochin Port': {'distance': 540, 'time': 10, 'highway': 'NH47'}, 'Mangalore Port': {'distance': 350, 'time': 7, 'highway': 'NH75'}, 'Pune': {'distance': 840, 'time': 14, 'highway': 'NH48'}},
    'Bangalore Central': {'Bangalore': {'distance': 10, 'time': 0.5, 'highway': 'Local'}, 'Chennai Central': {'distance': 350, 'time': 6, 'highway': 'NH48'}, 'Hyderabad': {'distance': 575, 'time': 10, 'highway': 'NH44'}, 'Coimbatore': {'distance': 360, 'time': 7, 'highway': 'NH209'}},
    'Chennai Port': {'Chennai Central': {'distance': 5, 'time': 0.5, 'highway': 'Local'}, 'Bangalore': {'distance': 350, 'time': 6, 'highway': 'NH48'}, 'Vijayawada': {'distance': 270, 'time': 5, 'highway': 'NH16'}, 'Trichy': {'distance': 320, 'time': 6, 'highway': 'NH45'}},
    'Chennai Central': {'Chennai Port': {'distance': 5, 'time': 0.5, 'highway': 'Local'}, 'Bangalore Central': {'distance': 350, 'time': 6, 'highway': 'NH48'}, 'Hyderabad': {'distance': 625, 'time': 11, 'highway': 'NH44'}, 'Coimbatore': {'distance': 505, 'time': 8.5, 'highway': 'NH544'}, 'Vijayawada': {'distance': 270, 'time': 5, 'highway': 'NH16'}, 'Madurai': {'distance': 460, 'time': 8, 'highway': 'NH38'}},
    'Hyderabad': {'Bangalore': {'distance': 575, 'time': 10, 'highway': 'NH44'}, 'Chennai Central': {'distance': 625, 'time': 11, 'highway': 'NH44'}, 'Vijayawada': {'distance': 275, 'time': 5, 'highway': 'NH65'}, 'Nagpur': {'distance': 500, 'time': 9, 'highway': 'NH44'}, 'Pune': {'distance': 560, 'time': 10, 'highway': 'NH65'}, 'Visakhapatnam Port': {'distance': 615, 'time': 11, 'highway': 'NH16'}, 'Warangal': {'distance': 145, 'time': 3, 'highway': 'NH163'}, 'Guntur': {'distance': 305, 'time': 5.5, 'highway': 'NH44'}, 'Kakinada': {'distance': 465, 'time': 8.5, 'highway': 'NH216'}},
    'Warangal': {'Hyderabad': {'distance': 145, 'time': 3, 'highway': 'NH163'}, 'Vijayawada': {'distance': 240, 'time': 4.5, 'highway': 'NH163'}, 'Kakinada': {'distance': 280, 'time': 5, 'highway': 'SH9'}, 'Nagpur': {'distance': 430, 'time': 8, 'highway': 'NH44'}},
    'Guntur': {'Hyderabad': {'distance': 305, 'time': 5.5, 'highway': 'NH44'}, 'Vijayawada': {'distance': 35, 'time': 1, 'highway': 'NH16'}, 'Chennai Central': {'distance': 330, 'time': 6, 'highway': 'NH16'}},
    'Kakinada': {'Hyderabad': {'distance': 465, 'time': 8.5, 'highway': 'NH216'}, 'Vijayawada': {'distance': 170, 'time': 3.5, 'highway': 'NH216'}, 'Visakhapatnam Port': {'distance': 180, 'time': 3.5, 'highway': 'NH16'}, 'Warangal': {'distance': 280, 'time': 5, 'highway': 'SH9'}},
    'Kolkata Port': {'Kolkata Central': {'distance': 5, 'time': 0.5, 'highway': 'Local'}, 'Bhubaneswar': {'distance': 445, 'time': 8, 'highway': 'NH16'}, 'Patna': {'distance': 585, 'time': 10, 'highway': 'NH19'}},
    'Kolkata Central': {'Kolkata Port': {'distance': 5, 'time': 0.5, 'highway': 'Local'}, 'Bhubaneswar': {'distance': 445, 'time': 8, 'highway': 'NH16'}, 'Patna': {'distance': 585, 'time': 10, 'highway': 'NH19'}, 'Guwahati': {'distance': 985, 'time': 18, 'highway': 'NH27'}, 'Ranchi': {'distance': 415, 'time': 8, 'highway': 'NH19'}},
    'Ahmedabad': {'Mumbai Central': {'distance': 525, 'time': 8, 'highway': 'NH48'}, 'Surat': {'distance': 265, 'time': 4.5, 'highway': 'NH48'}, 'Rajkot': {'distance': 215, 'time': 4, 'highway': 'NH27'}, 'Kandla Port': {'distance': 310, 'time': 6, 'highway': 'NH8A'}, 'Vadodara': {'distance': 105, 'time': 2, 'highway': 'NH48'}, 'Indore': {'distance': 410, 'time': 7, 'highway': 'NH47'}, 'Jaipur': {'distance': 655, 'time': 11, 'highway': 'NH48'}},
    'Surat': {'Mumbai Central': {'distance': 265, 'time': 5, 'highway': 'NH48'}, 'Ahmedabad': {'distance': 265, 'time': 4.5, 'highway': 'NH48'}, 'Vadodara': {'distance': 140, 'time': 2.5, 'highway': 'NH48'}, 'Nashik': {'distance': 190, 'time': 4, 'highway': 'NH48'}},
    'Jaipur': {'Delhi NCR': {'distance': 280, 'time': 5, 'highway': 'NH48'}, 'Ahmedabad': {'distance': 655, 'time': 11, 'highway': 'NH48'}, 'Indore': {'distance': 480, 'time': 8.5, 'highway': 'NH52'}},
    'Indore': {'Mumbai Central': {'distance': 580, 'time': 10, 'highway': 'NH52'}, 'Ahmedabad': {'distance': 410, 'time': 7, 'highway': 'NH47'}, 'Bhopal': {'distance': 195, 'time': 3.5, 'highway': 'NH46'}, 'Nagpur': {'distance': 515, 'time': 9, 'highway': 'NH44'}, 'Delhi NCR': {'distance': 715, 'time': 12, 'highway': 'NH48'}, 'Jaipur': {'distance': 480, 'time': 8.5, 'highway': 'NH52'}},
    'Nagpur': {'Mumbai Central': {'distance': 785, 'time': 13, 'highway': 'NH160'}, 'Hyderabad': {'distance': 500, 'time': 9, 'highway': 'NH44'}, 'Indore': {'distance': 515, 'time': 9, 'highway': 'NH44'}, 'Bhopal': {'distance': 355, 'time': 6.5, 'highway': 'NH44'}, 'Warangal': {'distance': 430, 'time': 8, 'highway': 'NH44'}},
    'Cochin Port': {'Bangalore': {'distance': 540, 'time': 10, 'highway': 'NH47'}, 'Coimbatore': {'distance': 190, 'time': 4, 'highway': 'NH544'}, 'Madurai': {'distance': 260, 'time': 5.5, 'highway': 'NH85'}},
    'Coimbatore': {'Bangalore': {'distance': 360, 'time': 7, 'highway': 'NH209'}, 'Chennai Central': {'distance': 505, 'time': 8.5, 'highway': 'NH544'}, 'Cochin Port': {'distance': 190, 'time': 4, 'highway': 'NH544'}, 'Madurai': {'distance': 215, 'time': 4.5, 'highway': 'NH209'}},
    'Visakhapatnam Port': {'Hyderabad': {'distance': 615, 'time': 11, 'highway': 'NH16'}, 'Vijayawada': {'distance': 350, 'time': 6.5, 'highway': 'NH16'}, 'Bhubaneswar': {'distance': 445, 'time': 8, 'highway': 'NH16'}, 'Kakinada': {'distance': 180, 'time': 3.5, 'highway': 'NH16'}},
    'Vijayawada': {'Chennai Port': {'distance': 270, 'time': 5, 'highway': 'NH16'}, 'Hyderabad': {'distance': 275, 'time': 5, 'highway': 'NH65'}, 'Visakhapatnam Port': {'distance': 350, 'time': 6.5, 'highway': 'NH16'}, 'Warangal': {'distance': 240, 'time': 4.5, 'highway': 'NH163'}, 'Guntur': {'distance': 35, 'time': 1, 'highway': 'NH16'}, 'Kakinada': {'distance': 170, 'time': 3.5, 'highway': 'NH216'}, 'Chennai Central': {'distance': 270, 'time': 5, 'highway': 'NH16'}},
    'Bhubaneswar': {'Kolkata Central': {'distance': 445, 'time': 8, 'highway': 'NH16'}, 'Visakhapatnam Port': {'distance': 445, 'time': 8, 'highway': 'NH16'}},
    'Lucknow': {'Delhi NCR': {'distance': 555, 'time': 9, 'highway': 'NH24'}, 'Noida': {'distance': 540, 'time': 8.5, 'highway': 'Yamuna Expressway'}, 'Patna': {'distance': 535, 'time': 10, 'highway': 'NH19'}},
    'Chandigarh': {'Delhi NCR': {'distance': 245, 'time': 5, 'highway': 'NH44'}},
    'Vadodara': {'Ahmedabad': {'distance': 105, 'time': 2, 'highway': 'NH48'}, 'Surat': {'distance': 140, 'time': 2.5, 'highway': 'NH48'}},
    'Nashik': {'Mumbai Port': {'distance': 165, 'time': 4, 'highway': 'NH160'}, 'Surat': {'distance': 190, 'time': 4, 'highway': 'NH48'}, 'Pune': {'distance': 210, 'time': 4.5, 'highway': 'NH60'}},
    'Kandla Port': {'Ahmedabad': {'distance': 310, 'time': 6, 'highway': 'NH8A'}, 'Rajkot': {'distance': 230, 'time': 4.5, 'highway': 'NH8A'}},
    'Rajkot': {'Ahmedabad': {'distance': 215, 'time': 4, 'highway': 'NH27'}, 'Kandla Port': {'distance': 230, 'time': 4.5, 'highway': 'NH8A'}},
    'Mangalore Port': {'Bangalore': {'distance': 350, 'time': 7, 'highway': 'NH75'}},
    'Madurai': {'Chennai Central': {'distance': 460, 'time': 8, 'highway': 'NH38'}, 'Coimbatore': {'distance': 215, 'time': 4.5, 'highway': 'NH209'}, 'Cochin Port': {'distance': 260, 'time': 5.5, 'highway': 'NH85'}},
    'Trichy': {'Chennai Port': {'distance': 320, 'time': 6, 'highway': 'NH45'}},
    'Bhopal': {'Indore': {'distance': 195, 'time': 3.5, 'highway': 'NH46'}, 'Nagpur': {'distance': 355, 'time': 6.5, 'highway': 'NH44'}},
    'Aurangabad': {'Mumbai Port': {'distance': 335, 'time': 6.5, 'highway': 'NH211'}, 'Pune': {'distance': 235, 'time': 5, 'highway': 'NH211'}},
    'Patna': {'Kolkata Central': {'distance': 585, 'time': 10, 'highway': 'NH19'}, 'Lucknow': {'distance': 535, 'time': 10, 'highway': 'NH19'}},
    'Ranchi': {'Kolkata Central': {'distance': 415, 'time': 8, 'highway': 'NH19'}},
    'Guwahati': {'Kolkata Central': {'distance': 985, 'time': 18, 'highway': 'NH27'}},
    'Faridabad': {'Gurgaon': {'distance': 25, 'time': 1, 'highway': 'Local'}},
}

ACTIVE_SHIPMENTS = [
    {'id': 'SHP001', 'origin': 'Mumbai Port', 'destination': 'Delhi NCR', 'cargo': 'Electronics', 'value': 5000000, 'status': 'In Transit'},
    {'id': 'SHP002', 'origin': 'Chennai Port', 'destination': 'Bangalore', 'cargo': 'Auto Parts', 'value': 3500000, 'status': 'In Transit'},
    {'id': 'SHP003', 'origin': 'JNPT Mumbai', 'destination': 'Pune', 'cargo': 'Machinery', 'value': 8000000, 'status': 'In Transit'},
    {'id': 'SHP004', 'origin': 'Kolkata Port', 'destination': 'Patna', 'cargo': 'Pharmaceuticals', 'value': 2500000, 'status': 'In Transit'},
    {'id': 'SHP005', 'origin': 'Kandla Port', 'destination': 'Ahmedabad', 'cargo': 'Chemicals', 'value': 4200000, 'status': 'In Transit'},
    {'id': 'SHP006', 'origin': 'Bangalore', 'destination': 'Chennai Central', 'cargo': 'IT Equipment', 'value': 6500000, 'status': 'In Transit'},
    {'id': 'SHP007', 'origin': 'Hyderabad', 'destination': 'Vijayawada', 'cargo': 'Textiles', 'value': 1800000, 'status': 'In Transit'},
    {'id': 'SHP008', 'origin': 'Pune', 'destination': 'Mumbai Central', 'cargo': 'FMCG', 'value': 2200000, 'status': 'In Transit'},
    {'id': 'SHP009', 'origin': 'Ahmedabad', 'destination': 'Surat', 'cargo': 'Diamonds', 'value': 15000000, 'status': 'In Transit'},
    {'id': 'SHP010', 'origin': 'Delhi NCR', 'destination': 'Jaipur', 'cargo': 'Consumer Goods', 'value': 3000000, 'status': 'In Transit'},
    {'id': 'SHP011', 'origin': 'Cochin Port', 'destination': 'Bangalore', 'cargo': 'Spices', 'value': 1500000, 'status': 'In Transit'},
    {'id': 'SHP012', 'origin': 'Visakhapatnam Port', 'destination': 'Hyderabad', 'cargo': 'Steel', 'value': 7500000, 'status': 'In Transit'},
    {'id': 'SHP013', 'origin': 'Mumbai Central', 'destination': 'Indore', 'cargo': 'Plastics', 'value': 2800000, 'status': 'In Transit'},
    {'id': 'SHP014', 'origin': 'Chennai Central', 'destination': 'Coimbatore', 'cargo': 'Automotive', 'value': 4500000, 'status': 'In Transit'},
    {'id': 'SHP015', 'origin': 'Gurgaon', 'destination': 'Noida', 'cargo': 'Electronics', 'value': 1200000, 'status': 'In Transit'},
    {'id': 'SHP016', 'origin': 'Kolkata Central', 'destination': 'Bhubaneswar', 'cargo': 'Fish Products', 'value': 1000000, 'status': 'In Transit'},
    {'id': 'SHP017', 'origin': 'Surat', 'destination': 'Mumbai Central', 'cargo': 'Textiles', 'value': 3300000, 'status': 'In Transit'},
    {'id': 'SHP018', 'origin': 'Indore', 'destination': 'Bhopal', 'cargo': 'Pharmaceuticals', 'value': 2100000, 'status': 'In Transit'},
    {'id': 'SHP019', 'origin': 'Bangalore', 'destination': 'Hyderabad', 'cargo': 'Software', 'value': 5500000, 'status': 'In Transit'},
    {'id': 'SHP020', 'origin': 'Jaipur', 'destination': 'Delhi NCR', 'cargo': 'Handicrafts', 'value': 900000, 'status': 'In Transit'},
    {'id': 'SHP021', 'origin': 'Nagpur', 'destination': 'Mumbai Central', 'cargo': 'Oranges', 'value': 800000, 'status': 'In Transit'},
    {'id': 'SHP022', 'origin': 'Coimbatore', 'destination': 'Bangalore', 'cargo': 'Machinery', 'value': 4000000, 'status': 'In Transit'},
    {'id': 'SHP023', 'origin': 'Lucknow', 'destination': 'Delhi NCR', 'cargo': 'Food Grains', 'value': 1600000, 'status': 'In Transit'},
    {'id': 'SHP024', 'origin': 'Vadodara', 'destination': 'Ahmedabad', 'cargo': 'Petrochemicals', 'value': 6000000, 'status': 'In Transit'},
    {'id': 'SHP025', 'origin': 'Nashik', 'destination': 'Pune', 'cargo': 'Wine', 'value': 1100000, 'status': 'In Transit'},
    {'id': 'SHP026', 'origin': 'Chennai Port', 'destination': 'Madurai', 'cargo': 'Rice', 'value': 1300000, 'status': 'In Transit'},
    {'id': 'SHP027', 'origin': 'Rajkot', 'destination': 'Kandla Port', 'cargo': 'Export Goods', 'value': 4800000, 'status': 'In Transit'},
    {'id': 'SHP028', 'origin': 'Patna', 'destination': 'Kolkata Central', 'cargo': 'Agricultural', 'value': 1400000, 'status': 'In Transit'},
    {'id': 'SHP029', 'origin': 'Chandigarh', 'destination': 'Delhi NCR', 'cargo': 'Electronics', 'value': 2700000, 'status': 'In Transit'},
    {'id': 'SHP030', 'origin': 'Mangalore Port', 'destination': 'Bangalore', 'cargo': 'Coffee', 'value': 950000, 'status': 'In Transit'},
]

DISRUPTION_TYPES = {
    'weather': {'monsoon_flooding': "Heavy monsoon rainfall causing road flooding", 'cyclone': "Tropical cyclone affecting coastal regions", 'fog': "Dense fog reducing visibility on highways", 'landslide': "Monsoon-induced landslides blocking routes", 'heat_wave': "Extreme heat affecting vehicle operations"},
    'infrastructure': {'road_construction': "Highway maintenance and construction work", 'bridge_repair': "Bridge repair causing route closure", 'port_congestion': "Port congestion due to high cargo volume", 'warehouse_full': "Warehouse at maximum capacity", 'power_outage': "Power outage affecting operations"},
    'social': {'strike': "Labor strike affecting logistics operations", 'protest': "Road blockade due to protests", 'festival': "Major festival causing traffic delays", 'bandh': "Regional bandh/shutdown"},
    'security': {'accident': "Major road accident blocking highway", 'vehicle_breakdown': "Fleet vehicle breakdown", 'theft': "Cargo theft incident reported"},
    'policy': {'toll_increase': "Sudden toll tax increase", 'permit_issue': "Interstate permit problems", 'customs_delay': "Customs clearance delays at port"}
}

FUEL_COST_PER_KM, DELAY_COST_PER_HOUR = 35, 2500

class RouteOptimizer:
    def __init__(self, network):
        self.network = network; self.graph = self._build_graph()
    def _build_graph(self):
        graph = defaultdict(dict)
        for source, destinations in self.network.items():
            for dest, info in destinations.items():
                graph[source][dest] = info['time']
                if source not in graph[dest]: graph[dest][source] = info['time']
        return graph
    def find_shortest_path(self, start, end, blocked_nodes=None):
        if blocked_nodes is None: blocked_nodes = set()
        if start == end: return [start], 0
        blocked_nodes = blocked_nodes - {start, end}
        pq = [(0, start, [start])]
        visited = set()
        while pq:
            cost, current, path = heapq.heappop(pq)
            if current in visited: continue
            visited.add(current)
            if current == end: return path, cost
            if current not in self.graph: continue
            for neighbor, edge_cost in self.graph[current].items():
                if neighbor not in visited and neighbor not in blocked_nodes:
                    new_cost = cost + edge_cost; new_path = path + [neighbor]
                    heapq.heappush(pq, (new_cost, neighbor, new_path))
        return None, float('inf')
    
    def find_k_shortest_paths(self, start, end, k=5, blocked_nodes=None):
        """Find k shortest alternative paths"""
        if blocked_nodes is None: blocked_nodes = set()
        paths = []
        temp_blocked = set()
        for i in range(k):
            path, cost = self.find_shortest_path(start, end, blocked_nodes.union(temp_blocked))
            if path is None or cost == float('inf'): break
            paths.append({'path': path, 'cost': cost, 'distance': self._calculate_path_distance(path)})
            if len(path) > 2: temp_blocked.add(path[len(path)//2])
        return paths
    
    def _calculate_path_distance(self, path):
        total_distance = 0
        for i in range(len(path) - 1):
            source, dest = path[i], path[i + 1]
            if source in self.network and dest in self.network[source]: total_distance += self.network[source][dest]['distance']
            elif dest in self.network and source in self.network[dest]: total_distance += self.network[dest][source]['distance']
        return total_distance
    def get_route_details(self, path):
        if not path or len(path) < 2: return []
        details = []
        for i in range(len(path) - 1):
            source, dest = path[i], path[i + 1]
            info = None
            if source in self.network and dest in self.network[source]: info = self.network[source][dest]
            elif dest in self.network and source in self.network[dest]: info = self.network[dest][source]
            if info: details.append({'from': source, 'to': dest, 'distance': info['distance'], 'time': info['time'], 'highway': info['highway']})
        return details
    def calculate_route_cost(self, path):
        details = self.get_route_details(path)
        total_distance, total_time = sum(d['distance'] for d in details), sum(d['time'] for d in details)
        fuel_cost, time_cost, total_cost = total_distance * FUEL_COST_PER_KM, total_time * DELAY_COST_PER_HOUR, (total_distance * FUEL_COST_PER_KM) + (total_time * DELAY_COST_PER_HOUR)
        return {'distance': total_distance, 'time': total_time, 'fuel_cost': fuel_cost, 'time_cost': time_cost, 'total_cost': total_cost}

def generate_realistic_disruption():
    category = np.random.choice(list(DISRUPTION_TYPES.keys()), p=[0.35, 0.25, 0.2, 0.15, 0.05])
    subcategory = np.random.choice(list(DISRUPTION_TYPES[category].keys()))
    location = np.random.choice(list(INDIAN_LOCATIONS.keys()))
    severity = np.random.choice(['Low', 'Medium', 'High', 'Critical'], p=[0.2, 0.3, 0.35, 0.15])
    impact_duration = {'Low': np.random.randint(1, 4), 'Medium': np.random.randint(3, 8), 'High': np.random.randint(6, 24), 'Critical': np.random.randint(12, 72)}
    return {'location': location, 'category': category, 'subcategory': subcategory, 'description': DISRUPTION_TYPES[category][subcategory], 'severity': severity, 'impact_duration_hrs': impact_duration[severity], 'affected_radius_km': np.random.randint(10, 200), 'timestamp': datetime.now()}

def create_india_map(locations, active_disruptions=None, shipments=None, highlight_shipment=None, user_route=None):
    fig = go.Figure()
    added_routes = set()
    for source, destinations in ROUTE_NETWORK.items():
        if source in locations:
            for dest in destinations.keys():
                if dest in locations:
                    route_key = tuple(sorted([source, dest]))
                    if route_key not in added_routes:
                        added_routes.add(route_key)
                        fig.add_trace(go.Scattergeo(lon=[locations[source]['lon'], locations[dest]['lon']], lat=[locations[source]['lat'], locations[dest]['lat']], mode='lines', line=dict(width=0.8, color='rgba(100, 150, 200, 0.3)'), hoverinfo='skip', showlegend=False))
    if shipments:
        optimizer = RouteOptimizer(ROUTE_NETWORK)
        for shipment in shipments:
            origin, destination = shipment['origin'], shipment['destination']
            if origin in locations and destination in locations:
                path, _ = optimizer.find_shortest_path(origin, destination)
                if path and len(path) >= 2:
                    lats, lons = [locations[city]['lat'] for city in path if city in locations], [locations[city]['lon'] for city in path if city in locations]
                    is_highlighted = (highlight_shipment and shipment['id'] == highlight_shipment)
                    fig.add_trace(go.Scattergeo(lon=lons, lat=lats, mode='lines', line=dict(width=4 if is_highlighted else 2, color='#00ff00' if is_highlighted else 'rgba(52, 152, 219, 0.6)'), name=f"{shipment['id']}: {origin} → {destination}" if is_highlighted else '', showlegend=is_highlighted, hoverinfo='text', hovertext=f"<b>{shipment['id']}</b><br>{origin} → {destination}<br>Cargo: {shipment['cargo']}<br>Value: ₹{shipment['value']:,}"))
    if user_route and 'path' in user_route:
        path = user_route['path']
        lats, lons = [locations[city]['lat'] for city in path if city in locations], [locations[city]['lon'] for city in path if city in locations]
        fig.add_trace(go.Scattergeo(lon=lons, lat=lats, mode='lines+markers', line=dict(width=5, color='#9b59b6'), marker=dict(size=8, color='#9b59b6'), name='Your Route', showlegend=True, hoverinfo='text', hovertext=f"Your Route: {' → '.join(path)}"))
    for loc_name, loc_data in locations.items():
        color_map = {'port': '#3498db', 'manufacturing': '#e74c3c', 'distribution': '#2ecc71', 'warehouse': '#f39c12'}
        size_map = {'port': 15, 'manufacturing': 12, 'distribution': 14, 'warehouse': 10}
        is_disrupted, disruption_info = False, ""
        if active_disruptions:
            for disr in active_disruptions:
                if disr['location'] == loc_name: is_disrupted = True; disruption_info = f"<br><b>⚠️ DISRUPTION</b><br>{disr['description']}<br>Severity: {disr['severity']}"; break
        fig.add_trace(go.Scattergeo(lon=[loc_data['lon']], lat=[loc_data['lat']], mode='markers+text', marker=dict(size=size_map[loc_data['type']] + (10 if is_disrupted else 0), color='red' if is_disrupted else color_map[loc_data['type']], line=dict(width=3 if is_disrupted else 1, color='white'), symbol='x' if is_disrupted else 'circle'), text=loc_name if is_disrupted else '', textposition='top center', textfont=dict(size=9, color='white', family='Arial Black'), hovertext=f"<b>{loc_name}</b><br>Type: {loc_data['type']}<br>Region: {loc_data['region']}{disruption_info}", hoverinfo='text', showlegend=False))
    fig.update_geos(scope='asia', center=dict(lat=22.5, lon=79), projection_scale=4.5, showland=True, landcolor='rgb(20, 25, 30)', coastlinecolor='rgb(50, 50, 70)', showocean=True, oceancolor='rgb(10, 15, 20)', showcountries=True, countrycolor='rgb(40, 40, 50)')
    fig.update_layout(height=700, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title=dict(text='🇮🇳 India Supply Chain Network - Live Monitoring', font=dict(size=20, color='white'), x=0.5, xanchor='center'), geo=dict(bgcolor='rgba(0,0,0,0)'))
    return fig

def check_affected_shipments(disruption, shipments, optimizer):
    affected, disrupted_location = [], disruption['location']
    for shipment in shipments:
        path, _ = optimizer.find_shortest_path(shipment['origin'], shipment['destination'])
        if path and disrupted_location in path: affected.append({'shipment': shipment, 'original_path': path})
    return affected

def main():
    st.markdown("""<style>.main {background-color: #0e1117;} .stMetric {background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #2e3140;} h1, h2, h3 {color: #ffffff !important;} .alert-critical {background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); padding: 20px; border-radius: 10px; color: white; border-left: 5px solid #fff;}</style>""", unsafe_allow_html=True)
    st.title("🇮🇳 India Supply Chain Intelligence System")
    st.markdown("*AI-Powered Disruption Prediction, Detection & Automatic Route Re-Optimization*")
    optimizer = RouteOptimizer(ROUTE_NETWORK)
    if 'active_disruptions' not in st.session_state: st.session_state.active_disruptions = []
    if 'active_shipments' not in st.session_state: st.session_state.active_shipments = ACTIVE_SHIPMENTS.copy()
    if 'user_routes' not in st.session_state: st.session_state.user_routes = []
    if 'affected_analysis' not in st.session_state: st.session_state.affected_analysis = []
    if 'predictions' not in st.session_state: st.session_state.predictions = []
    
    tabs = st.tabs(["🔮 Predict", "🗺️ Network", "📦 Shipments", "➕ Route", "🚨 Alerts", "📊 Analytics"])
    
    with tabs[0]:
        st.header("🔮 AI Disruption Prediction")
        col1, col2 = st.columns([2, 1])
        with col1:
            location = st.selectbox("📍 Location", list(INDIAN_LOCATIONS.keys()), key='pred_loc')
            cost = st.slider("💰 Cost (%)", 0, 100, 10, 5)
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🌦️ Weather", use_container_width=True):
                    weather = get_weather_data(location); st.session_state.weather = weather
                    if weather['success']: w = weather['data']; st.success(f"✅ {w['city']}, {w['country']}"); st.info(f"🌡️ {w['temperature']}°C | {w['description'].title()}\n💨 {w['wind_speed']} km/h | 💧 {w['humidity']}%")
                    else: st.error(weather['error'])
            with c2:
                if st.button("📰 GNews", use_container_width=True):
                    gnews = get_news_from_gnews(location); st.session_state.gnews = gnews
                    if gnews['success']: st.success(f"✅ {len(gnews['articles'])} articles"); [st.caption(f"📌 {a['title'][:70]}...") for a in gnews['articles'][:3]]
                    else: st.error(gnews['error'])
            with c3:
                if st.button("📰 NewsAPI", use_container_width=True):
                    newsapi = get_news_from_newsapi(location); st.session_state.newsapi = newsapi
                    if newsapi['success']: st.success(f"✅ {len(newsapi['articles'])} articles"); [st.caption(f"📌 {a['title'][:70]}...") for a in newsapi['articles'][:3]]
                    else: st.error(newsapi['error'])
            st.markdown("---")
            if st.button("🤖 PREDICT", use_container_width=True, type="primary"):
                weather = st.session_state.get('weather', get_weather_data(location))
                gnews = st.session_state.get('gnews', get_news_from_gnews(location))
                newsapi = st.session_state.get('newsapi', get_news_from_newsapi(location))
                pred = predict_disruption_probability(weather, gnews, newsapi, cost, location)
                st.session_state.predictions.append(pred); st.rerun()
        with col2:
            st.metric("Predictions", len(st.session_state.predictions))
            if st.session_state.predictions: latest = st.session_state.predictions[-1]; st.metric("Severity", latest['severity']); st.metric("Risk", f"{latest['probability']*100:.1f}%")
        if st.session_state.predictions:
            st.markdown("---")
            p = st.session_state.predictions[-1]
            st.markdown(f"<div class='alert-critical'><h2>{p['emoji']} {p['severity'].upper()}</h2><h3>{p['location']} | {p['probability']*100:.1f}%</h3></div>", unsafe_allow_html=True)
            st.markdown("### 🔍 Factors")
            fcols = st.columns(len(p['factors']))
            for idx, (fn, fd) in enumerate(p['factors'].items()):
                with fcols[idx]: st.markdown(f"*{fn}*\n- Score: {fd['score']*100:.0f}%\n- Weight: {fd['weight']*100:.0f}%"); [st.caption(r) for r in fd['reasons'][:2]]
    
    with tabs[1]:
        st.header("🗺️ Live Network")
        col1, col2 = st.columns([3, 1])
        with col1:
            hl = st.selectbox("Highlight", ['None'] + [s['id'] for s in st.session_state.active_shipments], 0)
            user_route_display = st.session_state.user_routes[-1] if st.session_state.user_routes else None
            map_fig = create_india_map(INDIAN_LOCATIONS, st.session_state.active_disruptions, st.session_state.active_shipments, None if hl == 'None' else hl, user_route_display)
            st.plotly_chart(map_fig, use_container_width=True)
        with col2:
            st.subheader("🎛️ Control Panel")
            if st.button("🔴 Random Disruption", use_container_width=True, type="primary"):
                nd = generate_realistic_disruption(); st.session_state.active_disruptions.append(nd)
                aff = check_affected_shipments(nd, st.session_state.active_shipments, optimizer)
                aff_u = [ur for ur in st.session_state.user_routes if nd['location'] in ur['path']]
                if aff or aff_u: st.session_state.affected_analysis = {'disruption': nd, 'affected_shipments': aff, 'affected_user_routes': aff_u}
                st.success(f"✅ {nd['location']}"); st.rerun()
            
            st.markdown("---")
            st.subheader("✏️ Manual Disruption")
            man_loc = st.selectbox("Location", list(INDIAN_LOCATIONS.keys()), key='man_loc')
            man_cat = st.selectbox("Category", list(DISRUPTION_TYPES.keys()), key='man_cat')
            man_subcat = st.selectbox("Type", list(DISRUPTION_TYPES[st.session_state.get('man_cat', 'weather')].keys()), key='man_sub')
            man_sev = st.selectbox("Severity", ['Low', 'Medium', 'High', 'Critical'], key='man_sev')
            man_hrs = st.slider("Duration (hrs)", 1, 72, 6, key='man_hrs')
            
            if st.button("➕ Add Disruption", use_container_width=True, type="secondary"):
                md = {
                    'location': man_loc,
                    'category': st.session_state.man_cat,
                    'subcategory': st.session_state.man_sub,
                    'description': DISRUPTION_TYPES[st.session_state.man_cat][st.session_state.man_sub],
                    'severity': st.session_state.man_sev,
                    'impact_duration_hrs': st.session_state.man_hrs,
                    'affected_radius_km': np.random.randint(10, 200),
                    'timestamp': datetime.now()
                }
                st.session_state.active_disruptions.append(md)
                aff = check_affected_shipments(md, st.session_state.active_shipments, optimizer)
                aff_u = [ur for ur in st.session_state.user_routes if md['location'] in ur['path']]
                if aff or aff_u: st.session_state.affected_analysis = {'disruption': md, 'affected_shipments': aff, 'affected_user_routes': aff_u}
                st.success(f"✅ Added at {man_loc}"); st.rerun()
            
            if st.button("🔄 Clear All", use_container_width=True):
                st.session_state.active_disruptions = []; st.session_state.affected_analysis = []; st.rerun()
            
            st.markdown("---")
            st.metric("🚚 Shipments", len(st.session_state.active_shipments))
            st.metric("⚠️ Disruptions", len(st.session_state.active_disruptions))
            st.metric("📍 Routes", len(st.session_state.user_routes))
    
    with tabs[2]:
        st.header("📦 Shipments")
        c1, c2, c3, c4 = st.columns(4)
        tv = sum(s['value'] for s in st.session_state.active_shipments)
        c1.metric("Value", f"₹{tv/10000000:.1f}Cr")
        c2.metric("Origins", len(set(s['origin'] for s in st.session_state.active_shipments)))
        c3.metric("Destinations", len(set(s['destination'] for s in st.session_state.active_shipments)))
        ac = len(st.session_state.affected_analysis.get('affected_shipments', [])) if st.session_state.affected_analysis else 0
        c4.metric("⚠️ Affected", ac)
        st.markdown("---")
        df = pd.DataFrame([{'ID': s['id'], 'Origin': s['origin'], 'Destination': s['destination'], 'Cargo': s['cargo'], 'Value (₹)': f"₹{s['value']:,}", 'Status': s['status']} for s in st.session_state.active_shipments])
        st.dataframe(df, use_container_width=True, height=500)
    
    with tabs[3]:
        st.header("➕ Plan Route")
        col1, col2 = st.columns(2)
        with col1:
            origin = st.selectbox("Origin", list(INDIAN_LOCATIONS.keys()), key='o')
            destination = st.selectbox("Destination", list(INDIAN_LOCATIONS.keys()), key='d')
            cargo_type = st.text_input("Cargo", "General Cargo")
            cargo_value = st.number_input("Value (₹)", 1000000, step=100000)
            if st.button("🔍 Find Routes", use_container_width=True, type="primary"):
                if origin != destination:
                    with st.spinner("Computing..."):
                        blocked = {d['location'] for d in st.session_state.active_disruptions}
                        alt_paths = optimizer.find_k_shortest_paths(origin, destination, k=5, blocked_nodes=blocked)
                        
                        if alt_paths:
                            st.success(f"✅ Found {len(alt_paths)} routes!")
                            st.session_state.alt_paths = alt_paths
                            st.session_state.route_params = {'origin': origin, 'destination': destination, 'cargo_type': cargo_type, 'cargo_value': cargo_value}
                        else:
                            st.error("❌ No routes found!")
                else:
                    st.warning("⚠️ Different locations!")
        
        with col2:
            if 'alt_paths' in st.session_state and st.session_state.alt_paths:
                st.subheader("🛣️ Available Routes")
                for idx, alt_route in enumerate(st.session_state.alt_paths, 1):
                    with st.expander(f"Route {idx}: {' → '.join(alt_route['path'][:3])}..."):
                        cost = optimizer.calculate_route_cost(alt_route['path'])
                        st.markdown(f"""
                        *Path:* {' → '.join(alt_route['path'])}
                        - Distance: {cost['distance']:.0f} km
                        - Time: {cost['time']:.1f} hrs
                        - Fuel: ₹{cost['fuel_cost']:,.0f}
                        - Total Cost: ₹{cost['total_cost']:,.0f}
                        """)
                        if st.button(f"✅ Select Route {idx}", key=f"select_{idx}", use_container_width=True):
                            rp = st.session_state.route_params
                            user_route = {
                                'id': f"USER{len(st.session_state.user_routes)+1:03d}",
                                'origin': rp['origin'],
                                'destination': rp['destination'],
                                'cargo': rp['cargo_type'],
                                'value': rp['cargo_value'],
                                'path': alt_route['path'],
                                'cost_details': cost,
                                'timestamp': datetime.now()
                            }
                            st.session_state.user_routes.append(user_route)
                            st.success(f"✅ Route saved!"); st.rerun()
            elif st.session_state.user_routes:
                st.subheader("✅ Selected Route")
                r = st.session_state.user_routes[-1]
                st.markdown(f"### {r['origin']} → {r['destination']}\n*{r['cargo']}* | ₹{r['value']:,}")
                c = r['cost_details']
                ca, cb = st.columns(2)
                ca.metric("Distance", f"{c['distance']:.0f} km"); ca.metric("Fuel", f"₹{c['fuel_cost']:,.0f}")
                cb.metric("Time", f"{c['time']:.1f} hrs"); cb.metric("Total", f"₹{c['total_cost']:,.0f}")
            else:
                st.info("Find routes to see alternatives")
    
    with tabs[4]:
        st.header("🚨 Disruption Alerts")
        if st.session_state.affected_analysis:
            ana = st.session_state.affected_analysis; d = ana['disruption']
            sev_col = {'Low': '🟢', 'Medium': '🟡', 'High': '🟠', 'Critical': '🔴'}
            st.markdown(f"<div class='alert-critical'><h2>{sev_col[d['severity']]} {d['severity'].upper()} - {d['location']}</h2><p>{d['description']} | {d['impact_duration_hrs']}h | {d['affected_radius_km']}km</p></div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            tav = sum(a['shipment']['value'] for a in ana.get('affected_shipments', [])) + sum(r['value'] for r in ana.get('affected_user_routes', []))
            c1.metric("Affected Ships", len(ana.get('affected_shipments', [])))
            c2.metric("Affected Routes", len(ana.get('affected_user_routes', [])))
            c3.metric("Value Risk", f"₹{tav/10000000:.2f}Cr")
            
            for aff in ana.get('affected_shipments', []):
                with st.expander(f"🚛 {aff['shipment']['id']}: {aff['shipment']['cargo']}"):
                    orig_cost = optimizer.calculate_route_cost(aff['original_path'])
                    alt_paths = optimizer.find_k_shortest_paths(aff['shipment']['origin'], aff['shipment']['destination'], k=3, blocked_nodes={d['location']})
                    
                    col_orig, col_alt = st.columns(2)
                    with col_orig:
                        st.markdown(f"*❌ Original (BLOCKED)*\n{' → '.join(aff['original_path'])}\n- Distance: {orig_cost['distance']:.0f} km\n- Cost: ₹{orig_cost['total_cost']:,.0f}")
                    
                    if alt_paths:
                        with col_alt:
                            st.markdown(f"*✅ Alternatives*")
                            for i, ap in enumerate(alt_paths, 1):
                                ac = optimizer.calculate_route_cost(ap['path'])
                                extra = ac['total_cost'] - orig_cost['total_cost']
                                st.button(f"Route {i}: +₹{extra:,.0f}", key=f"ship_route_{aff['shipment']['id']}_{i}", use_container_width=True)
                                st.caption(f"{' → '.join(ap['path'][:4])}...")
        else:
            st.success("✅ No disruptions affecting routes")
    
    with tabs[5]:
        st.header("📊 Analytics")
        c1, c2, c3, c4 = st.columns(4)
        tv = sum(s['value'] for s in st.session_state.active_shipments)
        c1.metric("Nodes", len(INDIAN_LOCATIONS)); c2.metric("Routes", sum(len(d) for d in ROUTE_NETWORK.values()))
        c3.metric("Value", f"₹{tv/10000000:.1f}Cr"); c4.metric("Disruptions", len(st.session_state.active_disruptions))

main()