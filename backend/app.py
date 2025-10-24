from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re
import numpy as np
from datetime import datetime
import time
import random
import threading
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
import json

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Configuration
GOOGLE_FACT_CHECK_API_KEY = "AIzaSyA-BFhMPN9048puVOHEmYtSNQLkYDldUz4"  # Replace with your actual API key
FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Global statistics
stats = {
    'total_analyzed': 0,
    'suspicious_count': 0,
    'avg_credibility': 0,
    'trend': 'stable',
    'avg_manipulation_index': 0,
    'avg_emotional_density': 0,
    'fact_checked_claims': 0
}

analysis_history = []

# 1️⃣ Load datasets
print("📊 Loading datasets...")
true_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

true_news["label"] = "real"
fake_news["label"] = "fake"

# Combine and shuffle
data = pd.concat([true_news, fake_news], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# 2️⃣ Preprocess
def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

data["text"] = data["title"] + " " + data["text"]
data["text"] = data["text"].apply(clean_text)

# 3️⃣ Train model
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test_vec)) * 100
print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}%")

# GOOGLE FACT CHECK API INTEGRATION
def google_fact_check(text):
    """Check claims against Google Fact Check API"""
    if not GOOGLE_FACT_CHECK_API_KEY or GOOGLE_FACT_CHECK_API_KEY == "YOUR_GOOGLE_API_KEY":
        return {
            'available': False,
            'message': 'Google Fact Check API key not configured'
        }
    
    try:
        # Extract potential claims from text
        claims = extract_claims_from_text(text)
        fact_check_results = []
        
        for claim in claims[:3]:  # Limit to 3 claims to avoid rate limiting
            params = {
                'key': GOOGLE_FACT_CHECK_API_KEY,
                'query': claim,
                'languageCode': 'en'
            }
            
            response = requests.get(FACT_CHECK_API_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'claims' in data and data['claims']:
                    for claim_result in data['claims'][:2]:  # Get top 2 results per claim
                        fact_check_results.append({
                            'claim': claim,
                            'claimant': claim_result.get('claimant', 'Unknown'),
                            'claim_date': claim_result.get('claimDate', ''),
                            'text': claim_result.get('text', ''),
                            'rating': get_claim_rating(claim_result),
                            'url': claim_result.get('claimReview', [{}])[0].get('url', '') if claim_result.get('claimReview') else ''
                        })
                        stats['fact_checked_claims'] += 1
        
        return {
            'available': True,
            'results': fact_check_results,
            'total_claims_checked': len(fact_check_results)
        }
    
    except Exception as e:
        return {
            'available': False,
            'error': str(e),
            'message': 'Fact check service temporarily unavailable'
        }

def extract_claims_from_text(text):
    """Extract potential factual claims from text"""
    # Simple claim extraction - can be enhanced with NLP
    sentences = re.split(r'[.!?]+', text)
    claims = []
    
    # Patterns that often indicate factual claims
    claim_indicators = [
        r'\b(studies? show|research indicates|according to|data shows|experts say)\b',
        r'\b(proven|demonstrated|found that|reveals|confirmed)\b',
        r'\b(percentage|statistics|data|numbers|figures)\b'
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) > 5:  # Reasonable length for a claim
            # Check if sentence contains claim indicators
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in claim_indicators):
                claims.append(sentence)
            # Also include sentences that look like factual statements
            elif looks_like_factual_claim(sentence):
                claims.append(sentence)
    
    return claims[:5]  # Return top 5 potential claims

def looks_like_factual_claim(sentence):
    """Heuristic to identify potential factual claims"""
    words = sentence.lower().split()
    
    # Sentences with numbers often contain facts
    if re.search(r'\d+', sentence):
        return True
    
    # Sentences with certain verbs often make claims
    claim_verbs = ['is', 'are', 'was', 'were', 'has', 'have', 'shows', 'proves', 'demonstrates']
    if any(verb in words for verb in claim_verbs):
        return True
    
    return False

def get_claim_rating(claim_result):
    """Extract and normalize claim rating"""
    if not claim_result.get('claimReview'):
        return 'Unknown'
    
    review = claim_result['claimReview'][0]
    rating = review.get('textualRating', 'Unknown')
    
    # Normalize ratings
    rating = rating.lower()
    if any(word in rating for word in ['false', 'incorrect', 'wrong', 'fabricated']):
        return 'False'
    elif any(word in rating for word in ['true', 'correct', 'accurate']):
        return 'True'
    elif any(word in rating for word in ['misleading', 'exaggerated', 'out of context']):
        return 'Misleading'
    elif any(word in rating for word in ['unproven', 'unverified', 'unsupported']):
        return 'Unverified'
    else:
        return rating.title()

# NOVEL APPROACH 1: Cognitive Bias Detection
def detect_psychological_manipulation(text):
    """Detect psychological manipulation techniques"""
    manipulation_techniques = []
    
    # Urgency and scarcity
    urgency_words = ['urgent', 'immediately', 'now', 'breaking', 'last chance', 'limited time']
    if any(word in text.lower() for word in urgency_words):
        manipulation_techniques.append("Creates false urgency")
    
    # Social proof manipulation
    social_proof_words = ['everyone knows', 'people are saying', 'everybody', 'the world']
    if any(word in text.lower() for word in social_proof_words):
        manipulation_techniques.append("Uses vague social proof")
    
    # Authority appeals
    authority_words = ['experts say', 'scientists prove', 'doctors recommend', 'official sources']
    if any(word in text.lower() for word in authority_words) and not re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', text):
        manipulation_techniques.append("Appeals to unnamed authority")
    
    # Emotional triggering
    emotional_words = ['shocking', 'amazing', 'unbelievable', 'horrifying', 'outrageous']
    emotional_count = sum(1 for word in emotional_words if word in text.lower())
    if emotional_count > 2:
        manipulation_techniques.append(f"Uses {emotional_count} emotional trigger words")
    
    # Either-or fallacy detection
    either_or_patterns = [r'either [^,.]+ or [^,.]+ disaster', r'if we don\'t [^,.]+, then [^,.]+']
    if any(re.search(pattern, text.lower()) for pattern in either_or_patterns):
        manipulation_techniques.append("Uses false dichotomy")
    
    return manipulation_techniques

# NOVEL APPROACH 2: Information vs Emotional Density
def calculate_manipulation_metrics(text):
    """Calculate quantitative manipulation metrics"""
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        return {
            'emotional_density': 0,
            'factual_density': 0,
            'manipulation_index': 0,
            'sentiment_polarity': 0
        }
    
    # Emotional words dictionary
    emotional_words = {
        'shocking', 'amazing', 'unbelievable', 'horrifying', 'outrageous', 'incredible',
        'astounding', 'stunning', 'devastating', 'heartbreaking', 'miraculous'
    }
    
    # Factual indicator words
    factual_indicators = {
        'according', 'study', 'research', 'data', 'statistics', 'report', 'analysis',
        'findings', 'evidence', 'showed', 'demonstrated', 'confirmed'
    }
    
    emotional_count = sum(1 for word in words if word.lower() in emotional_words)
    factual_count = sum(1 for word in words if word.lower() in factual_indicators)
    
    emotional_density = emotional_count / total_words
    factual_density = factual_count / total_words
    
    # Avoid division by zero
    manipulation_index = emotional_density / (factual_density + 0.001)
    
    # Sentiment analysis
    sentiment = sia.polarity_scores(text)
    
    return {
        'emotional_density': round(emotional_density, 4),
        'factual_density': round(factual_density, 4),
        'manipulation_index': round(manipulation_index, 2),
        'sentiment_polarity': round(sentiment['compound'], 3),
        'emotional_words_count': emotional_count,
        'factual_indicators_count': factual_count
    }

# NOVEL APPROACH 3: Writing Style Inconsistency
def detect_style_inconsistency(text):
    """Detect inconsistencies in writing style suggesting mixed authorship"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if len(sentences) < 3:
        return {'style_shifts': 0, 'inconsistency_score': 0, 'issues': []}
    
    style_metrics = []
    issues = []
    
    for sentence in sentences:
        words = sentence.split()
        if len(words) > 0:
            # Sentence length variation
            sentence_length = len(words)
            
            # Complexity metrics
            avg_word_length = sum(len(word) for word in words) / len(words)
            complexity = sentence_length * avg_word_length
            
            style_metrics.append({
                'length': sentence_length,
                'complexity': complexity,
                'avg_word_length': avg_word_length
            })
    
    # Analyze variations
    if len(style_metrics) > 1:
        lengths = [m['length'] for m in style_metrics]
        complexities = [m['complexity'] for m in style_metrics]
        
        length_std = np.std(lengths)
        complexity_std = np.std(complexities)
        
        # Detect significant shifts
        style_shifts = 0
        if length_std > np.mean(lengths) * 0.5:
            style_shifts += 1
            issues.append("High variation in sentence lengths")
        
        if complexity_std > np.mean(complexities) * 0.6:
            style_shifts += 1
            issues.append("Inconsistent writing complexity")
        
        inconsistency_score = min(100, (length_std + complexity_std) * 10)
        
        return {
            'style_shifts': style_shifts,
            'inconsistency_score': round(inconsistency_score, 2),
            'issues': issues,
            'length_variation': round(length_std, 2),
            'complexity_variation': round(complexity_std, 2)
        }
    
    return {'style_shifts': 0, 'inconsistency_score': 0, 'issues': []}

# NOVEL APPROACH 4: Narrative Template Matching
def detect_narrative_patterns(text):
    """Detect known disinformation narrative templates"""
    narrative_templates = {
        'conspiracy': [
            r'[^.!?]*secret[^.!?]*government[^.!?]*hidden[^.!?]*',
            r'[^.!?]*they don\'t want you to know[^.!?]*',
            r'[^.!?]*mainstream media[^.!?]*lying[^.!?]*'
        ],
        'urgency': [
            r'[^.!?]*act now[^.!?]*before[^.!?]*',
            r'[^.!?]*limited time[^.!?]*offer[^.!?]*',
            r'[^.!?]*breaking[^.!?]*emergency[^.!?]*'
        ],
        'miracle_cure': [
            r'[^.!?]*one simple trick[^.!?]*',
            r'[^.!?]*doctors hate this[^.!?]*',
            r'[^.!?]*cure[^.!?]*secret[^.!?]*'
        ]
    }
    
    matched_templates = []
    for template_type, patterns in narrative_templates.items():
        for pattern in patterns:
            if re.search(pattern, text.lower()):
                matched_templates.append(template_type)
                break  # Only count each template once
    
    return {
        'matched_templates': list(set(matched_templates)),
        'template_count': len(set(matched_templates))
    }

# Enhanced analysis functions
def analyze_text_features(text):
    """Enhanced text feature analysis with novel approaches"""
    analysis_points = []
    
    # Traditional features
    if len(re.findall(r'[A-Z]{5,}', text)) > 2:
        analysis_points.append("Excessive use of capitalization")
    
    if text.count('!') > 3:
        analysis_points.append("Excessive exclamation marks")
    
    word_count = len(text.split())
    if word_count < 20:
        analysis_points.append("Very short content")
    elif word_count > 500:
        analysis_points.append("Unusually long content")
    
    # Novel approach analyses
    manipulation_techniques = detect_psychological_manipulation(text)
    analysis_points.extend(manipulation_techniques)
    
    style_analysis = detect_style_inconsistency(text)
    if style_analysis['style_shifts'] > 0:
        analysis_points.extend(style_analysis['issues'])
    
    narrative_analysis = detect_narrative_patterns(text)
    if narrative_analysis['template_count'] > 0:
        analysis_points.append(f"Matches {narrative_analysis['template_count']} known disinformation templates")
    
    return analysis_points

def get_verdict(score, manipulation_index, style_inconsistency, fact_check_results):
    """Enhanced verdict system with fact checking"""
    # Adjust score based on fact check results
    fact_check_penalty = 0
    if fact_check_results.get('available') and fact_check_results.get('results'):
        false_claims = sum(1 for result in fact_check_results['results'] if result['rating'] in ['False', 'Misleading'])
        if false_claims > 0:
            fact_check_penalty = false_claims * 10
    
    adjusted_score = max(0, score - fact_check_penalty)
    
    # Base verdict on adjusted credibility score
    if adjusted_score >= 80 and manipulation_index < 2 and style_inconsistency < 30:
        return {"status": "Highly Credible", "color": "#10B981", "icon": "✅"}
    elif adjusted_score >= 65 and manipulation_index < 3:
        return {"status": "Credible", "color": "#34D399", "icon": "✓"}
    elif adjusted_score >= 50:
        return {"status": "Neutral", "color": "#F59E0B", "icon": "⚖️"}
    elif adjusted_score >= 35:
        return {"status": "Suspicious", "color": "#EF4444", "icon": "⚠️"}
    else:
        return {"status": "Highly Suspicious", "color": "#DC2626", "icon": "🚨"}

def update_stats(credibility_score, manipulation_index, emotional_density):
    """Update global statistics"""
    stats['total_analyzed'] += 1
    
    if credibility_score < 50:
        stats['suspicious_count'] += 1
    
    # Update averages
    if stats['total_analyzed'] == 1:
        stats['avg_credibility'] = credibility_score
        stats['avg_manipulation_index'] = manipulation_index
        stats['avg_emotional_density'] = emotional_density
    else:
        stats['avg_credibility'] = (
            (stats['avg_credibility'] * (stats['total_analyzed'] - 1) + credibility_score) 
            / stats['total_analyzed']
        )
        stats['avg_manipulation_index'] = (
            (stats['avg_manipulation_index'] * (stats['total_analyzed'] - 1) + manipulation_index) 
            / stats['total_analyzed']
        )
        stats['avg_emotional_density'] = (
            (stats['avg_emotional_density'] * (stats['total_analyzed'] - 1) + emotional_density) 
            / stats['total_analyzed']
        )

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Clean and predict
        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = np.max(model.predict_proba(vectorized))

        # Calculate credibility score (0-100)
        if prediction == "real":
            credibility_score = 70 + (confidence * 30)
        else:
            credibility_score = 30 - (confidence * 30)
        
        credibility_score = max(0, min(100, round(credibility_score, 1)))
        
        # Run enhanced analyses
        analysis_points = analyze_text_features(text)
        manipulation_metrics = calculate_manipulation_metrics(text)
        style_analysis = detect_style_inconsistency(text)
        narrative_analysis = detect_narrative_patterns(text)
        
        # Google Fact Check API
        fact_check_results = google_fact_check(text)
        
        # Get enhanced verdict
        verdict = get_verdict(
            credibility_score, 
            manipulation_metrics['manipulation_index'],
            style_analysis['inconsistency_score'],
            fact_check_results
        )
        
        # Create comprehensive analysis result
        analysis = {
            "id": f"analysis_{int(time.time())}_{random.randint(1000, 9999)}",
            "text": text,
            "credibility_score": credibility_score,
            "verdict": verdict,
            "prediction": prediction,
            "confidence": round(confidence * 100, 1),
            "word_count": len(text.split()),
            "analysis_points": analysis_points,
            "timestamp": datetime.now().isoformat(),
            "model_accuracy": f"{accuracy:.2f}%",
            
            # Enhanced analysis data
            "enhanced_analysis": {
                "psychological_manipulation": detect_psychological_manipulation(text),
                "manipulation_metrics": manipulation_metrics,
                "style_analysis": style_analysis,
                "narrative_patterns": narrative_analysis,
                "fact_check": fact_check_results
            }
        }
        
        # Update statistics and history
        analysis_history.append(analysis)
        update_stats(
            credibility_score, 
            manipulation_metrics['manipulation_index'],
            manipulation_metrics['emotional_density']
        )
        
        # Emit real-time update
        socketio.emit('new_analysis', analysis)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(stats)

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify(analysis_history[-10:])

@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "model_accuracy": f"{accuracy:.2f}%",
        "dataset_size": len(data),
        "real_news_count": len(true_news),
        "fake_news_count": len(fake_news),
        "features": [
            "Machine Learning Classification",
            "Psychological Manipulation Detection", 
            "Writing Style Analysis",
            "Narrative Pattern Matching",
            "Google Fact Check API Integration",
            "Emotional vs Factual Density Analysis"
        ]
    })

@app.route('/')
def home():
    return jsonify({
        "message": "Advanced Fake News Detection API",
        "status": "running",
        "model_accuracy": f"{accuracy:.2f}%",
        "features": "ML + Psychological Analysis + Fact Checking",
        "note": "Set GOOGLE_FACT_CHECK_API_KEY environment variable for fact checking"
    })

# Simulate real-time analysis for demo
def simulate_real_time_analysis():
    """Simulate real-time analysis for demo purposes"""
    sample_texts = [
        "Scientists discover breakthrough in renewable energy that could transform global power systems.",
        "BREAKING: Celebrities involved in secret government conspiracy exposed! Shocking details!",
        "Economic indicators show positive growth and stable market conditions according to latest reports.",
        "One simple trick to become millionaire overnight revealed by financial experts!",
        "Medical research confirms benefits of healthy diet and regular exercise for long-term health.",
        "URGENT WARNING: Government hiding truth about alien contact for decades! Whistleblower reveals all!"
    ]
    
    while True:
        time.sleep(25)
        if random.random() < 0.6:
            text = random.choice(sample_texts)
            
            # Clean and predict
            cleaned = clean_text(text)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            confidence = np.max(model.predict_proba(vectorized))

            # Calculate credibility score
            if prediction == "real":
                credibility_score = 70 + (confidence * 30)
            else:
                credibility_score = 30 - (confidence * 30)
            
            credibility_score = max(0, min(100, round(credibility_score, 1)))
            
            # Enhanced analyses
            analysis_points = analyze_text_features(text)
            manipulation_metrics = calculate_manipulation_metrics(text)
            style_analysis = detect_style_inconsistency(text)
            fact_check_results = google_fact_check(text)
            
            verdict = get_verdict(
                credibility_score,
                manipulation_metrics['manipulation_index'],
                style_analysis['inconsistency_score'],
                fact_check_results
            )
            
            analysis = {
                "id": f"demo_{int(time.time())}_{random.randint(1000, 9999)}",
                "text": text,
                "credibility_score": credibility_score,
                "verdict": verdict,
                "prediction": prediction,
                "confidence": round(confidence * 100, 1),
                "word_count": len(text.split()),
                "analysis_points": analysis_points,
                "timestamp": datetime.now().isoformat(),
                "model_accuracy": f"{accuracy:.2f}%",
                "enhanced_analysis": {
                    "psychological_manipulation": detect_psychological_manipulation(text),
                    "manipulation_metrics": manipulation_metrics,
                    "style_analysis": style_analysis,
                    "fact_check": fact_check_results
                }
            }
            
            analysis_history.append(analysis)
            update_stats(
                credibility_score,
                manipulation_metrics['manipulation_index'],
                manipulation_metrics['emotional_density']
            )
            socketio.emit('new_analysis', analysis)

# Start simulation in background thread
simulation_thread = threading.Thread(target=simulate_real_time_analysis, daemon=True)
simulation_thread.start()

if __name__ == "__main__":
    print("🚀 Advanced Fake News Detection API Started!")
    print(f"📊 Model Accuracy: {accuracy:.2f}%")
    print(f"📁 Dataset: {len(true_news)} real, {len(fake_news)} fake articles")
    print("🔗 Server: http://localhost:5000")
    print("📡 WebSocket: Ready for real-time updates")
    print("🧠 Features: ML + Psychological Analysis + Style Detection + Fact Checking")
    
    if GOOGLE_FACT_CHECK_API_KEY == "AIzaSyA-BFhMPN9048puVOHEmYtSNQLkYDldUz4":
        print("⚠️  Google Fact Check API: Not configured - set GOOGLE_FACT_CHECK_API_KEY environment variable")
    else:
        print("✅ Google Fact Check API: Configured and ready")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)