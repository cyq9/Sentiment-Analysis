import os
import re
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

# Ignore irrelevant warnings
warnings.filterwarnings('ignore')

# Set Chinese font and style (still use SimHei for proper display, but labels will be in English)
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


def seg_word(sentence, stopwords):
    """Tokenize sentence and filter stopwords"""
    if not sentence:
        return []
    seg_list = jieba.cut(sentence)
    return [word.strip() for word in seg_list if word.strip() and word not in stopwords]


# ---------------------------
# 2. Sentiment Lexicons and Rule-based Method
# ---------------------------
def load_sentiment_resources():
    """Load sentiment analysis lexicons"""
    stopwords = set(load_file('stopwords.txt', default_content=[]))
    negation_words = load_file('否定词.txt', default_content=["不", "没", "没有", "别", "莫", "勿"])  # Keep Chinese for actual words
    degree_dict = defaultdict(float)
    for line in load_file('程度副词.txt', default_content=[]):
        try:
            word, weight = line.split(',', 1)
            degree_dict[word.strip()] = float(weight.strip())
        except:
            print(f"Ignoring invalid degree adverb: {line}")
    sentiment_dict = defaultdict(float)
    for line in load_file('BosonNLP_sentiment_score.txt', default_content=[]):
        try:
            word, score = line.split(' ', 1)
            sentiment_dict[word.strip()] = float(score.strip())
        except:
            print(f"Ignoring invalid sentiment word: {line}")
    print(f"Lexicons loaded: Stopwords({len(stopwords)}) | Negation Words({len(negation_words)}) | Degree Adverbs({len(degree_dict)}) | Sentiment Words({len(sentiment_dict)})")
    return stopwords, negation_words, degree_dict, sentiment_dict


def sentiment_score_by_rule(sentence, stopwords, negation_words, degree_dict, sentiment_dict):
    """Rule-based sentiment scoring using lexicons"""
    token_list = seg_word(sentence, stopwords)
    if not token_list:
        return 0.0

    sentiment_words = defaultdict(float)
    negation_marks = defaultdict(int)
    degree_marks = defaultdict(float)
    for i, token in enumerate(token_list):
        if token in sentiment_dict:
            sentiment_words[i] = sentiment_dict[token]
        elif token in negation_words:
            negation_marks[i] = -1
        elif token in degree_dict:
            degree_marks[i] = degree_dict[token]

    score = 0.0
    weight = 1.0
    sentiment_indices = sorted(sentiment_words.keys())
    for i in range(len(token_list)):
        if i in sentiment_words:
            score += weight * sentiment_words[i]
            next_sen_idx = None
            for idx in sentiment_indices:
                if idx > i:
                    next_sen_idx = idx
                    break
            next_sen_idx = next_sen_idx or len(token_list)
            for j in range(i + 1, next_sen_idx):
                if j in negation_marks:
                    weight *= -1
                elif j in degree_marks:
                    weight *= degree_marks[j]
            weight = 1.0
    return score


# ---------------------------
# 3. Logistic Regression Model
# ---------------------------
def load_50_dataset(dataset_file='sentiment_dataset.tsv'):
    """Load 50-sentence sentiment analysis dataset (TSV format required)"""
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file {dataset_file} not found")
        print("Example dataset format (text\t sentiment):")
        print("This movie is amazing!\t1")
        print("Poor service attitude\t0")
        return None
    try:
        df = pd.read_csv(dataset_file, sep='\t')
        print(f"Successfully loaded dataset: {len(df)} records")
        return df
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None


def train_logistic_regression_model(stopwords, dataset_df):
    """Train logistic regression model using 50-sentence dataset"""
    if dataset_df is None:
        return None, None
    
    # Preprocess: tokenize and join as text
    dataset_df['tokenized_text'] = dataset_df['text'].apply(lambda x: ' '.join(seg_word(x, stopwords)))
    
    # Feature extraction: TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(dataset_df['tokenized_text'])
    y = dataset_df['sentiment'].astype(int)
    
    # Split into training and test sets (80% training, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("\n=== Logistic Regression Model Evaluation ===")
    print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    return model, vectorizer


def predict_sentiment_score(model, vectorizer, sentence, stopwords):
    """Predict sentiment score using logistic regression model (mapped to [-1, 1] range)"""
    if not sentence:
        return 0.0
    tokenized_text = ' '.join(seg_word(sentence, stopwords))
    try:
        X = vectorizer.transform([tokenized_text])
        proba = model.predict_proba(X)[0][1]  # Probability of positive sentiment
        return 2 * proba - 1  # Map to [-1, 1]
    except Exception as e:
        print(f"Prediction error: {e}, sentence: {sentence}")
        return 0.0


# ---------------------------
# 4. Single Plot Functions
# ---------------------------
def label_to_score(label):
    """Convert annotation label (0/1) to sentiment score (-1/1)"""
    return 2 * label - 1


def plot_bar_chart(sentences, dict_scores, lr_scores, label_scores, title_suffix=""):
    """Plot bar chart comparing three methods' scores"""
    valid_mask = [bool(s.strip()) for s in sentences]
    valid_sentences = [s for s, v in zip(sentences, valid_mask) if v]
    valid_dict = [s for s, v in zip(dict_scores, valid_mask) if v]
    valid_lr = [s for s, v in zip(lr_scores, valid_mask) if v]
    valid_labels = [s for s, v in zip(label_scores, valid_mask) if v]
    
    if not valid_sentences:
        print("No valid test sentences, skipping chart")
        return
    
    short_sentences = [s[:15] + '...' if len(s) > 15 else s for s in valid_sentences]
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(valid_sentences))
    width = 0.25
    ax.bar(x - width, valid_dict, width, label='Lexicon-based', color='skyblue')
    ax.bar(x, valid_lr, width, label='Logistic Regression', color='salmon')
    ax.bar(x + width, valid_labels, width, label='Annotated Score', color='lightgreen')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(short_sentences, rotation=45, ha='right')
    ax.set_ylabel('Sentiment Score')
    ax.set_title(f'Comparison of Sentiment Scores by Three Methods{title_suffix}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'sentiment_bar{title_suffix}.png', dpi=300)
    print(f"Chart 1/4 saved as sentiment_bar{title_suffix}.png")
    plt.show()
    time.sleep(1)  # Pause to view


def plot_line_chart(sentences, dict_scores, lr_scores, label_scores, title_suffix=""):
    """Plot line chart showing score trends"""
    valid_mask = [bool(s.strip()) for s in sentences]
    valid_sentences = [s for s, v in zip(sentences, valid_mask) if v]
    valid_dict = [s for s, v in zip(dict_scores, valid_mask) if v]
    valid_lr = [s for s, v in zip(lr_scores, valid_mask) if v]
    valid_labels = [s for s, v in zip(label_scores, valid_mask) if v]
    
    if not valid_sentences:
        print("No valid test sentences, skipping chart")
        return
    
    short_sentences = [s[:15] + '...' if len(s) > 15 else s for s in valid_sentences]
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(short_sentences, valid_dict, marker='o', label='Lexicon-based', color='skyblue')
    ax.plot(short_sentences, valid_lr, marker='s', label='Logistic Regression', color='salmon')
    ax.plot(short_sentences, valid_labels, marker='^', label='Annotated Score', color='lightgreen')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xticklabels(short_sentences, rotation=45, ha='right')
    ax.set_ylabel('Sentiment Score')
    ax.set_title(f'Sentiment Score Trends by Three Methods{title_suffix}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'sentiment_line{title_suffix}.png', dpi=300)
    print(f"Chart 2/4 saved as sentiment_line{title_suffix}.png")
    plt.show()
    time.sleep(1)


def plot_scatter_chart(sentences, dict_scores, lr_scores, label_scores, title_suffix=""):
    """Plot scatter chart for prediction-annotation correlation"""
    valid_mask = [bool(s.strip()) for s in sentences]
    valid_sentences = [s for s, v in zip(sentences, valid_mask) if v]
    valid_dict = [s for s, v in zip(dict_scores, valid_mask) if v]
    valid_lr = [s for s, v in zip(lr_scores, valid_mask) if v]
    valid_labels = [s for s, v in zip(label_scores, valid_mask) if v]
    
    if not valid_sentences:
        print("No valid test sentences, skipping chart")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.scatter(valid_labels, valid_dict, color='skyblue', alpha=0.7, label='Lexicon vs Annotation')
    ax.scatter(valid_labels, valid_lr, color='salmon', alpha=0.7, label='Logistic Regression vs Annotation')
    min_val = min(min(valid_labels), min(valid_dict), min(valid_lr)) - 0.1
    max_val = max(max(valid_labels), max(valid_dict), max(valid_lr)) + 0.1
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    ax.set_xlabel('Annotated Score')
    ax.set_ylabel('Predicted Score')
    ax.set_title(f'Correlation Between Predicted and Annotated Scores{title_suffix}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'sentiment_scatter{title_suffix}.png', dpi=300)
    print(f"Chart 3/4 saved as sentiment_scatter{title_suffix}.png")
    plt.show()
    time.sleep(1)


def plot_error_chart(sentences, dict_scores, lr_scores, label_scores, title_suffix=""):
    """Plot bar chart comparing prediction errors"""
    valid_mask = [bool(s.strip()) for s in sentences]
    valid_sentences = [s for s, v in zip(sentences, valid_mask) if v]
    valid_dict = [s for s, v in zip(dict_scores, valid_mask) if v]
    valid_lr = [s for s, v in zip(lr_scores, valid_mask) if v]
    valid_labels = [s for s, v in zip(label_scores, valid_mask) if v]
    
    if not valid_sentences:
        print("No valid test sentences, skipping chart")
        return
    
    short_sentences = [s[:15] + '...' if len(s) > 15 else s for s in valid_sentences]
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(valid_sentences))
    width = 0.35
    dict_errors = [abs(d - l) for d, l in zip(valid_dict, valid_labels)]
    lr_errors = [abs(n - l) for n, l in zip(valid_lr, valid_labels)]
    ax.bar(x - width/2, dict_errors, width, label='Lexicon Error', color='skyblue')
    ax.bar(x + width/2, lr_errors, width, label='Logistic Regression Error', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(short_sentences, rotation=45, ha='right')
    ax.set_ylabel('Absolute Error')
    ax.set_title(f'Comparison of Prediction Errors by Two Methods{title_suffix}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'sentiment_error{title_suffix}.png', dpi=300)
    print(f"Chart 4/4 saved as sentiment_error{title_suffix}.png")
    plt.show()


# ---------------------------
# 5. Main Program
# ---------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("Sentiment Analysis System (Sequential Plot Display)")
    print("=" * 50)
    
    # 1. Load sentiment lexicons
    stopwords, negation_words, degree_dict, sentiment_dict = load_sentiment_resources()
    
    # 2. Load 50-sentence dataset (prepare sentiment_50.tsv in advance)
    print("\n=== Loading Dataset ===")
    dataset_df = load_50_dataset('sentiment_dataset.tsv')
    if dataset_df is None:
        print("Dataset loading failed, exiting program")
        exit(1)
    
    # 3. Train logistic regression model
    print("\n=== Training Logistic Regression Model ===")
    model, vectorizer = train_logistic_regression_model(stopwords, dataset_df)
    if not model or not vectorizer:
        print("Model training failed, exiting program")
        exit(1)
    
    # 4. Prepare test sentences (randomly select 10 from dataset)
    print("\n=== Preparing Test Sentences ===")
    test_df = dataset_df.sample(10, random_state=42)
    test_sentences = test_df['text'].tolist()
    true_labels = test_df['sentiment'].tolist()
    print("Test Sentences (Annotated Labels):")
    for i, (s, l) in enumerate(zip(test_sentences, true_labels)):
        print(f"  {i+1}. [{l}] {s}")
    
    # 5. Calculate scores by three methods
    print("\n=== Calculating Sentiment Scores ===")
    dict_scores, lr_scores, label_scores = [], [], []
    for i, (sentence, label) in enumerate(zip(test_sentences, true_labels)):
        print(f"Processing sentence {i+1}/{len(test_sentences)}: {sentence[:30]}...")
        
        # Lexicon-based score
        dict_score = sentiment_score_by_rule(sentence, stopwords, negation_words, degree_dict, sentiment_dict)
        
        # Logistic regression score
        lr_score = predict_sentiment_score(model, vectorizer, sentence, stopwords)
        
        # Annotated score (0→-1, 1→1)
        label_score = label_to_score(label)
        
        dict_scores.append(dict_score)
        lr_scores.append(lr_score)
        label_scores.append(label_score)
        
        print(f"  Lexicon-based: {dict_score:.4f} | Logistic Regression: {lr_score:.4f} | Annotated: {label_score:.4f}")
    
    # 6. Calculate and display average errors
    dict_avg_error = np.mean([abs(d - l) for d, l in zip(dict_scores, label_scores)])
    lr_avg_error = np.mean([abs(n - l) for n, l in zip(lr_scores, label_scores)])
    print(f"\nAverage error of lexicon-based method: {dict_avg_error:.4f}")
    print(f"Average error of logistic regression: {lr_avg_error:.4f}")
    
    # Determine better method
    better_method = "Lexicon-based" if dict_avg_error < lr_avg_error else "Logistic Regression"
    print(f"\n{better_method} has lower average error ({'%.4f' % min(dict_avg_error, lr_avg_error)})")
    
    # 7. Generate comparison charts
    print("\n=== Generating Comparison Charts ===")
    print("(Please close each chart window to proceed to the next one)")
    
    plot_bar_chart(test_sentences, dict_scores, lr_scores, label_scores)
    plot_line_chart(test_sentences, dict_scores, lr_scores, label_scores)
    plot_scatter_chart(test_sentences, dict_scores, lr_scores, label_scores)
    plot_error_chart(test_sentences, dict_scores, lr_scores, label_scores)
    
    # 8. Interactive sentiment analysis
    print("\n=== Interactive Sentiment Analysis ===")
    print("Enter custom sentences for sentiment analysis (type 'q' to quit)")
    
    while True:
        user_input = input("\nEnter sentence: ")
        if user_input.lower() == 'q':
            break
        
        # Calculate scores for user input
        dict_score = sentiment_score_by_rule(user_input, stopwords, negation_words, degree_dict, sentiment_dict)
        lr_score = predict_sentiment_score(model, vectorizer, user_input, stopwords)
        
        # Determine sentiment
        dict_sentiment = "Positive" if dict_score > 0 else "Negative"
        lr_sentiment = "Positive" if lr_score > 0 else "Negative"
        
        # Calculate confidence
        dict_confidence = abs(dict_score) * 100
        lr_confidence = abs(lr_score) * 100
        
        print("\nAnalysis Results:")
        print(f"Lexicon-based method: score={dict_score:.4f}, sentiment={dict_sentiment}, confidence={dict_confidence:.1f}%")
        print(f"Logistic regression: score={lr_score:.4f}, sentiment={lr_sentiment}, confidence={lr_confidence:.1f}%")
        
        # Provide combined result
        combined_score = (dict_score + lr_score) / 2
        combined_sentiment = "Positive" if combined_score > 0 else "Negative"
        combined_confidence = abs(combined_score) * 100
        print(f"Combined result: score={combined_score:.4f}, sentiment={combined_sentiment}, confidence={combined_confidence:.1f}%")
    
    print("\n=" * 50)
    print("Sentiment Analysis System execution completed")
    print("=" * 50)