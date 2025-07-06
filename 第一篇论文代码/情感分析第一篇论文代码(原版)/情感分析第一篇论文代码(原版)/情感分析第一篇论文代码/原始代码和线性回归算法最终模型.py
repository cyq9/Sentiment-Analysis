import os
import re
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

# Ignore irrelevant warnings
warnings.filterwarnings('ignore')

# Set font and style for proper Chinese display (labels in English)
sns.set(font="SimHei", style="whitegrid")  
plt.rcParams["axes.unicode_minus"] = False  # Correctly display minus sign


# ---------------------------
# 1. Utility Functions: File and Text Processing
# ---------------------------
def load_file(file_path, default_content=None, encoding='utf-8'):
    """General file loading function with exception handling"""
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist, using default content")
        return default_content or []
    try:
        with open(file_path, 'r', encoding=encoding) as fr:
            return [line.strip() for line in fr if line.strip()]
    except Exception as e:
        print(f"Failed to read file {file_path}: {e}")
        return default_content or []


def tokenize_sentence(sentence, stopwords):
    """Tokenize a sentence and filter out stopwords"""
    if not sentence:
        return []
    seg_list = jieba.cut(sentence)
    return [word.strip() for word in seg_list if word.strip() and word not in stopwords]


# ---------------------------
# 2. Sentiment Lexicons and Rule-based Scoring
# ---------------------------
def load_sentiment_resources():
    """Load sentiment analysis lexicons (stopwords, negation words, degree adverbs, sentiment words)"""
    stopwords = set(load_file('stopwords.txt', default_content=[]))
    negation_words = load_file('negation_words.txt', default_content=["不", "没", "没有", "别", "莫", "勿"])
    degree_dict = defaultdict(float)
    for line in load_file('degree_adverbs.txt', default_content=[]):
        try:
            word, weight = line.split(',', 1)
            degree_dict[word.strip()] = float(weight.strip())
        except:
            print(f"Ignoring invalid degree adverb entry: {line}")
    sentiment_dict = defaultdict(float)
    for line in load_file('BosonNLP_sentiment_score.txt', default_content=[]):
        try:
            word, score = line.split(' ', 1)
            sentiment_dict[word.strip()] = float(score.strip())
        except:
            print(f"Ignoring invalid sentiment word entry: {line}")
    print(f"Lexicons loaded: Stopwords({len(stopwords)}) | Negation Words({len(negation_words)}) | "
          f"Degree Adverbs({len(degree_dict)}) | Sentiment Words({len(sentiment_dict)})")
    return stopwords, negation_words, degree_dict, sentiment_dict


def rule_based_sentiment_score(sentence, stopwords, negation_words, degree_dict, sentiment_dict):
    """Calculate sentiment score using rule-based method with lexicons"""
    tokens = tokenize_sentence(sentence, stopwords)
    if not tokens:
        return 0.0

    sentiment_word_scores = defaultdict(float)
    negation_markers = defaultdict(int)
    degree_modifiers = defaultdict(float)

    for i, token in enumerate(tokens):
        if token in sentiment_dict:
            sentiment_word_scores[i] = sentiment_dict[token]
        elif token in negation_words:
            negation_markers[i] = -1
        elif token in degree_dict:
            degree_modifiers[i] = degree_dict[token]

    total_score = 0.0
    current_weight = 1.0
    sentiment_indices = sorted(sentiment_word_scores.keys())

    for i in range(len(tokens)):
        if i in sentiment_word_scores:
            total_score += current_weight * sentiment_word_scores[i]
            next_sentiment_idx = None
            for idx in sentiment_indices:
                if idx > i:
                    next_sentiment_idx = idx
                    break
            next_sentiment_idx = next_sentiment_idx or len(tokens)
            for j in range(i + 1, next_sentiment_idx):
                if j in negation_markers:
                    current_weight *= -1
                elif j in degree_modifiers:
                    current_weight *= degree_modifiers[j]
            current_weight = 1.0
    return total_score


# ---------------------------
# 3. Linear Regression Model Training and Prediction
# ---------------------------
def load_dataset(dataset_file='sentiment_dataset.tsv'):
    """Load dataset with text and corresponding continuous sentiment scores (TSV format)"""
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file {dataset_file} not found")
        print("Expected dataset format (TSV):")
        print("text\tsentiment_score")
        print("This movie is amazing!\t0.8")
        print("Poor service attitude\t-0.6")
        return None
    try:
        df = pd.read_csv(dataset_file, sep='\t')
        print(f"Successfully loaded dataset: {len(df)} records")
        return df
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None


def train_linear_regression(stopwords, dataset_df):
    """Train a linear regression model for sentiment score prediction"""
    if dataset_df is None:
        return None, None
    
    # Preprocess text: tokenize and convert to string
    dataset_df['tokenized_text'] = dataset_df['text'].apply(lambda x: ' '.join(tokenize_sentence(x, stopwords)))
    
    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(dataset_df['tokenized_text'])
    y = dataset_df['sentiment'].astype(float)  # Target: continuous sentiment scores
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\n=== Linear Regression Model Evaluation ===")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")
    return model, vectorizer


def predict_with_linear_regression(model, vectorizer, sentence, stopwords):
    """Predict sentiment score using the trained linear regression model"""
    if not sentence:
        return 0.0
    tokenized_text = ' '.join(tokenize_sentence(sentence, stopwords))
    try:
        X = vectorizer.transform([tokenized_text])
        return model.predict(X)[0]
    except Exception as e:
        print(f"Prediction error: {e}, input sentence: {sentence}")
        return 0.0


# ---------------------------
# 4. Visualization Functions
# ---------------------------
def plot_bar_comparison(sentences, rule_scores, lr_scores, true_scores, title_suffix=""):
    """Plot bar chart to compare scores from three methods"""
    valid_mask = [bool(s.strip()) for s in sentences]
    valid_sentences = [s for s, v in zip(sentences, valid_mask) if v]
    valid_rule = [s for s, v in zip(rule_scores, valid_mask) if v]
    valid_lr = [s for s, v in zip(lr_scores, valid_mask) if v]
    valid_true = [s for s, v in zip(true_scores, valid_mask) if v]
    
    if not valid_sentences:
        print("No valid test sentences, skipping bar chart.")
        return
    
    short_sentences = [s[:15] + '...' if len(s) > 15 else s for s in valid_sentences]
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(valid_sentences))
    width = 0.25
    ax.bar(x - width, valid_rule, width, label='Rule-based', color='skyblue')
    ax.bar(x, valid_lr, width, label='Linear Regression', color='salmon')
    ax.bar(x + width, valid_true, width, label='True Score', color='lightgreen')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(short_sentences, rotation=45, ha='right')
    ax.set_ylabel('Sentiment Score')
    ax.set_title(f'Comparison of Sentiment Scores{title_suffix}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'sentiment_bar{title_suffix}.png', dpi=300)
    print(f"Bar chart saved as sentiment_bar{title_suffix}.png")
    plt.show()
    time.sleep(1)


def plot_line_trend(sentences, rule_scores, lr_scores, true_scores, title_suffix=""):
    """Plot line chart to show score trends across methods"""
    valid_mask = [bool(s.strip()) for s in sentences]
    valid_sentences = [s for s, v in zip(sentences, valid_mask) if v]
    valid_rule = [s for s, v in zip(rule_scores, valid_mask) if v]
    valid_lr = [s for s, v in zip(lr_scores, valid_mask) if v]
    valid_true = [s for s, v in zip(true_scores, valid_mask) if v]
    
    if not valid_sentences:
        print("No valid test sentences, skipping line chart.")
        return
    
    short_sentences = [s[:15] + '...' if len(s) > 15 else s for s in valid_sentences]
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(short_sentences, valid_rule, marker='o', label='Rule-based', color='skyblue')
    ax.plot(short_sentences, valid_lr, marker='s', label='Linear Regression', color='salmon')
    ax.plot(short_sentences, valid_true, marker='^', label='True Score', color='lightgreen')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xticklabels(short_sentences, rotation=45, ha='right')
    ax.set_ylabel('Sentiment Score')
    ax.set_title(f'Sentiment Score Trends{title_suffix}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'sentiment_line{title_suffix}.png', dpi=300)
    print(f"Line chart saved as sentiment_line{title_suffix}.png")
    plt.show()
    time.sleep(1)


def plot_scatter_correlation(sentences, rule_scores, lr_scores, true_scores, title_suffix=""):
    """Plot scatter chart to show correlation between predicted and true scores"""
    valid_mask = [bool(s.strip()) for s in sentences]
    valid_sentences = [s for s, v in zip(sentences, valid_mask) if v]
    valid_rule = [s for s, v in zip(rule_scores, valid_mask) if v]
    valid_lr = [s for s, v in zip(lr_scores, valid_mask) if v]
    valid_true = [s for s, v in zip(true_scores, valid_mask) if v]
    
    if not valid_sentences:
        print("No valid test sentences, skipping scatter chart.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.scatter(valid_true, valid_rule, color='skyblue', alpha=0.7, label='Rule-based vs True')
    ax.scatter(valid_true, valid_lr, color='salmon', alpha=0.7, label='Linear Regression vs True')
    min_val = min(min(valid_true), min(valid_rule), min(valid_lr)) - 0.1
    max_val = max(max(valid_true), max(valid_rule), max(valid_lr)) + 0.1
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    ax.set_xlabel('True Score')
    ax.set_ylabel('Predicted Score')
    ax.set_title(f'Correlation Between Predicted and True Scores{title_suffix}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'sentiment_scatter{title_suffix}.png', dpi=300)
    print(f"Scatter chart saved as sentiment_scatter{title_suffix}.png")
    plt.show()
    time.sleep(1)


def plot_error_comparison(sentences, rule_scores, lr_scores, true_scores, title_suffix=""):
    """Plot bar chart to compare prediction errors of two methods"""
    valid_mask = [bool(s.strip()) for s in sentences]
    valid_sentences = [s for s, v in zip(sentences, valid_mask) if v]
    valid_rule = [s for s, v in zip(rule_scores, valid_mask) if v]
    valid_lr = [s for s, v in zip(lr_scores, valid_mask) if v]
    valid_true = [s for s, v in zip(true_scores, valid_mask) if v]
    
    if not valid_sentences:
        print("No valid test sentences, skipping error chart.")
        return
    
    short_sentences = [s[:15] + '...' if len(s) > 15 else s for s in valid_sentences]
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(valid_sentences))
    width = 0.35
    rule_errors = [abs(d - l) for d, l in zip(valid_rule, valid_true)]
    lr_errors = [abs(n - l) for n, l in zip(valid_lr, valid_true)]
    ax.bar(x - width/2, rule_errors, width, label='Rule-based Error', color='skyblue')
    ax.bar(x + width/2, lr_errors, width, label='Linear Regression Error', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(short_sentences, rotation=45, ha='right')
    ax.set_ylabel('Absolute Error')
    ax.set_title(f'Comparison of Prediction Errors{title_suffix}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'sentiment_error{title_suffix}.png', dpi=300)
    print(f"Error chart saved as sentiment_error{title_suffix}.png")
    plt.show()


# ---------------------------
# 5. Main Execution Flow
# ---------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("Sentiment Analysis System (Linear Regression Model)")
    print("=" * 50)
    
    # Step 1: Load sentiment lexicons
    print("\n=== Loading Sentiment Resources ===")
    stopwords, negation_words, degree_dict, sentiment_dict = load_sentiment_resources()
    
    # Step 2: Load dataset (expects continuous sentiment scores)
    print("\n=== Loading Dataset ===")
    dataset_df = load_dataset('sentiment_dataset.tsv')
    if dataset_df is None:
        print("Dataset loading failed. Exiting program.")
        exit(1)
    
    # Step 3: Train linear regression model
    print("\n=== Training Linear Regression Model ===")
    model, vectorizer = train_linear_regression(stopwords, dataset_df)
    if not model or not vectorizer:
        print("Model training failed. Exiting program.")
        exit(1)
    
    # Step 4: Prepare test sentences (random sample from dataset)
    print("\n=== Preparing Test Sentences ===")
    test_df = dataset_df.sample(10, random_state=42)
    test_sentences = test_df['text'].tolist()
    true_scores = test_df['sentiment'].tolist()
    print("Test Sentences (True Scores):")
    for i, (sentence, score) in enumerate(zip(test_sentences, true_scores)):
        print(f"  {i+1}. Score: {score:.4f} | Text: {sentence[:70]}...")
    
    # Step 5: Calculate scores using both methods
    print("\n=== Calculating Sentiment Scores ===")
    rule_based_scores, lr_scores = [], []
    for i, sentence in enumerate(test_sentences):
        print(f"Processing sentence {i+1}/{len(test_sentences)}: {sentence[:50]}...")
        
        # Rule-based score
        rule_score = rule_based_sentiment_score(sentence, stopwords, negation_words, degree_dict, sentiment_dict)
        
        # Linear regression score
        lr_score = predict_with_linear_regression(model, vectorizer, sentence, stopwords)
        
        # Collect scores
        rule_based_scores.append(rule_score)
        lr_scores.append(lr_score)
        
        print(f"  Rule-based: {rule_score:.4f} | Linear Regression: {lr_score:.4f} | True: {true_scores[i]:.4f}")
    
    # Step 6: Calculate average errors
    rule_avg_error = np.mean([abs(d - l) for d, l in zip(rule_based_scores, true_scores)])
    lr_avg_error = np.mean([abs(n - l) for n, l in zip(lr_scores, true_scores)])
    print(f"\nAverage Error - Rule-based: {rule_avg_error:.4f}")
    print(f"Average Error - Linear Regression: {lr_avg_error:.4f}")
    
    # Determine better method
    better_method = "Rule-based" if rule_avg_error < lr_avg_error else "Linear Regression"
    print(f"\n{better_method} has lower average error ({min(rule_avg_error, lr_avg_error):.4f})")
    
    # Step 7: Generate comparison charts
    print("\n=== Generating Comparison Charts ===")
    print("(Please close each chart window to proceed to the next one)")
    
    plot_bar_comparison(test_sentences, rule_based_scores, lr_scores, true_scores)
    plot_line_trend(test_sentences, rule_based_scores, lr_scores, true_scores)
    plot_scatter_correlation(test_sentences, rule_based_scores, lr_scores, true_scores)
    plot_error_comparison(test_sentences, rule_based_scores, lr_scores, true_scores)
    
    # Step 8: Interactive sentiment analysis
    print("\n=== Interactive Sentiment Analysis ===")
    print("Enter custom sentences for sentiment analysis (type 'q' to quit)")
    
    while True:
        user_input = input("\nEnter sentence: ")
        if user_input.lower() == 'q':
            break
        
        # Calculate scores for user input
        rule_score = rule_based_sentiment_score(user_input, stopwords, negation_words, degree_dict, sentiment_dict)
        lr_score = predict_with_linear_regression(model, vectorizer, user_input, stopwords)
        
        # Determine sentiment labels
        rule_sentiment = "Positive" if rule_score > 0 else "Negative"
        lr_sentiment = "Positive" if lr_score > 0 else "Negative"
        
        # Calculate confidence
        rule_confidence = min(abs(rule_score) * 100, 100.0)
        lr_confidence = min(abs(lr_score) * 100, 100.0)
        
        print("\nAnalysis Results:")
        print(f"Rule-based Method: Score={rule_score:.4f}, Sentiment={rule_sentiment}, Confidence={rule_confidence:.1f}%")
        print(f"Linear Regression: Score={lr_score:.4f}, Sentiment={lr_sentiment}, Confidence={lr_confidence:.1f}%")
        
        # Provide combined result
        combined_score = (rule_score + lr_score) / 2
        combined_sentiment = "Positive" if combined_score > 0 else "Negative"
        combined_confidence = min(abs(combined_score) * 100, 100.0)
        print(f"Combined Result: Score={combined_score:.4f}, Sentiment={combined_sentiment}, Confidence={combined_confidence:.1f}%")
    
    print("\n=" * 50)
    print("Sentiment Analysis System Execution Completed")
    print("=" * 50)