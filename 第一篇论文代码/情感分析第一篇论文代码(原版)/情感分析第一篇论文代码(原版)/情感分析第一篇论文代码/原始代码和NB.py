import os
import re
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error,
    r2_score, classification_report, confusion_matrix
)
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from io import StringIO

# Ignore irrelevant warnings
warnings.filterwarnings('ignore')

# Set font for plotting (SimHei for Chinese characters, labels in English)
sns.set(font="SimHei", style="whitegrid")  
plt.rcParams["axes.unicode_minus"] = False  # Ensure minus sign displays correctly

# Check if running in a Jupyter environment
try:
    get_ipython
    IN_JUPYTER = True
except NameError:
    IN_JUPYTER = False
    plt.switch_backend('TkAgg')  


# ---------------------------
# 1. Utility Functions: File and Text Processing
# ---------------------------
def load_file(file_path, default_content=None, encoding='utf-8'):
    """File loading function with error handling"""
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
    """Tokenize text and filter stopwords"""
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
    negation_words = load_file('negation_words.txt', default_content=["不", "没", "没有", "别", "莫", "勿"])
    degree_dict = defaultdict(float)
    
    # Load degree adverbs with weights
    for line in load_file('degree_adverbs.txt', default_content=[]):
        try:
            parts = line.split(',', 1)
            if len(parts) == 2:
                word, weight = parts
                degree_dict[word.strip()] = float(weight.strip())
            else:
                degree_dict[line.strip()] = 1.5  # Default intensity weight
        except Exception as e:
            print(f"Ignoring invalid degree adverb: {line}, Error: {e}")
    
    sentiment_dict = defaultdict(float)
    # Support both "word score" and "word,score" formats
    for line in load_file('BosonNLP_sentiment_score.txt', default_content=[]):
        try:
            if ',' in line:
                word, score = line.split(',', 1)
            else:
                word, score = line.split(' ', 1)
            sentiment_dict[word.strip()] = float(score.strip())
        except Exception as e:
            print(f"Ignoring invalid sentiment word: {line}, Error: {e}")
    
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
    
    # Mark positions and weights of sentiment words, negations, and degree adverbs
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
    
    # Analyze sentiment words and their surrounding negations/degrees
    for i in range(len(token_list)):
        if i in sentiment_words:
            # Apply current weight to sentiment word
            score += weight * sentiment_words[i]
            
            # Find next sentiment word to determine influence range
            next_sen_idx = None
            for idx in sentiment_indices:
                if idx > i:
                    next_sen_idx = idx
                    break
            next_sen_idx = next_sen_idx or len(token_list)
            
            # Reset weight for the next sentiment word
            weight = 1.0
        else:
            # Update weight for non-sentiment words (negations/degrees)
            if i in negation_marks:
                weight *= negation_marks[i]  # Negation flips weight
            elif i in degree_marks:
                weight *= degree_marks[i]  # Degree adjusts weight
    
    return score


# ---------------------------
# 3. Multinomial Naive Bayes Model
# ---------------------------
def load_custom_dataset(dataset_file='sentiment_dataset.tsv'):
    """Load the custom sentiment analysis dataset with enhanced error handling"""
    if not os.path.exists(dataset_file):
        print(f"Custom dataset file {dataset_file} not found, creating demo dataset")
        return create_demo_dataset()
    
    try:
        # Try to load the dataset with different possible columns
        df = pd.read_csv(dataset_file, sep='\t')
        
        # Validate required columns
        required_columns = {'text', 'sentiment'}
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            print(f"Error: Dataset is missing required columns: {', '.join(missing_columns)}")
            print("Using demo dataset instead")
            return create_demo_dataset()
        
        # Check label distribution
        label_counts = df['sentiment'].value_counts()
        print(f"Dataset loaded: {len(df)} records")
        print("Label distribution:")
        for label, count in label_counts.items():
            print(f"  Label {label}: {count} ({count/len(df):.2%})")
        
        # Check for empty texts
        empty_texts = df['text'].isna().sum()
        if empty_texts > 0:
            print(f"Warning: {empty_texts} records have empty text, removing them")
            df = df.dropna(subset=['text'])
        
        return df
    
    except Exception as e:
        print(f"Failed to load custom dataset: {e}")
        print("Using demo dataset instead")
        return create_demo_dataset()


def create_demo_dataset():
    """Create a more diverse demo dataset with different domains"""
    demo_data = """The movie plot is wonderful, and the actors' performances are great    1
The restaurant service is poor, and the dishes are not fresh    0
The hotel environment is comfortable, and the staff are friendly    1
The quality of this phone is bad, it broke after a week    0
This book is very informative and worth reading    1
The clothes have incorrect sizes and rough fabric    0
The concert atmosphere is lively, and the singer performed well    1
The takeout delivery was slow, and the food was cold    0
The park has a beautiful environment, suitable for walking    1
The software interface is complicated and not user-friendly    0
The new smartphone has excellent battery life and fast performance    1
The customer service team resolved my issue quickly and politely    1
The flight was delayed without proper notification    0
The coffee tastes bitter and overpriced    0
The gym equipment is well-maintained and the staff is helpful    1"""
    
    # Convert to DataFrame
    df = pd.read_csv(StringIO(demo_data), sep='\t', header=None, names=['text', 'sentiment'])
    print(f"Created demo dataset with {len(df)} samples covering multiple domains")
    return df


def train_naive_bayes_model(stopwords, dataset_df, save_model=True):
    """Train Multinomial Naive Bayes model with cross-validation and hyperparameter tuning"""
    if dataset_df is None or len(dataset_df) < 10:
        print("Dataset is empty or has too few records to train model")
        return None, None
    
    print("\nTraining Multinomial Naive Bayes model...")
    start_time = time.time()
    
    # Preprocess: tokenize and convert to bag of words
    print("Preprocessing data...")
    dataset_df['tokenized_text'] = dataset_df['text'].apply(lambda x: ' '.join(seg_word(x, stopwords)))
    
    # Feature extraction: TF-IDF vectorization with improved parameters
    print("Extracting features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Increased from 3000 to capture more features
        min_df=2,           # Ignore terms that appear in less than 2 documents
        max_df=0.9,         # Ignore terms that appear in more than 90% of documents
        ngram_range=(1, 2), # Include unigrams and bigrams
        use_idf=True,
        smooth_idf=True
    )
    
    X = vectorizer.fit_transform(dataset_df['tokenized_text'])
    
    # Ensure labels are binary (0/1)
    y = dataset_df['sentiment'].astype(int)
    
    # Split into training and test sets (80% train, 20% test) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    

    # Train model with hyperparameter tuning
    print("Training model with hyperparameter tuning...")
    best_alpha = find_best_alpha(X_train, y_train)
    model = MultinomialNB(alpha=best_alpha)
    model.fit(X_train, y_train)
    
    # Model evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Positive class probability
    
    # Calculate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred_proba)
    mae = mean_absolute_error(y_test, y_pred_proba)
    r2 = r2_score(y_test, y_pred_proba)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred, 
                                       target_names=['Negative', 'Positive'], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n=== Naive Bayes Model Evaluation ===")
    print(f"Best alpha: {best_alpha}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Coefficient of Determination (R²): {r2:.4f}")
    print(f"5-fold Cross-validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nClassification Metrics:")
    print(f"  Positive F1-score: {class_report['Positive']['f1-score']:.4f}")
    print(f"  Negative F1-score: {class_report['Negative']['f1-score']:.4f}")
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save the model if requested
    if save_model:
        save_model_to_file(model, vectorizer)
    
    return model, vectorizer


def find_best_alpha(X_train, y_train):
    """Simple hyperparameter tuning for MultinomialNB alpha parameter"""
    print("Searching for best alpha parameter...")
    alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    best_alpha = 1.0
    best_score = 0.0
    
    for alpha in alphas:
        model = MultinomialNB(alpha=alpha)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        mean_score = scores.mean()
        
        print(f"Alpha: {alpha}, Cross-validation accuracy: {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha
    
    return best_alpha


def save_model_to_file(model, vectorizer):
    """Save trained model and vectorizer to files"""
    try:
        import joblib
        joblib.dump(model, 'naive_bayes_model.pkl')
        joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
        print("Model and vectorizer saved to files: naive_bayes_model.pkl, tfidf_vectorizer.pkl")
    except Exception as e:
        print(f"Failed to save model: {e}")


def load_model_from_file():
    """Load trained model and vectorizer from files"""
    try:
        if os.path.exists('naive_bayes_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
            import joblib
            model = joblib.load('naive_bayes_model.pkl')
            vectorizer = joblib.load('tfidf_vectorizer.pkl')
            print("Model and vectorizer loaded from files")
            return model, vectorizer
        else:
            print("Saved model files not found")
            return None, None
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None


def predict_sentiment_score(model, vectorizer, sentence, stopwords):
    """Predict sentiment score using Naive Bayes with error handling"""
    if not sentence or model is None or vectorizer is None:
        return 0.0
    
    try:
        tokenized_text = ' '.join(seg_word(sentence, stopwords))
        X = vectorizer.transform([tokenized_text])
        score = model.predict_proba(X)[0, 1]  # Positive class probability
        return 2 * score - 1  # Map to [-1, 1] range
    except Exception as e:
        print(f"Prediction error: {e}, Sentence: {sentence}")
        return 0.0


# ---------------------------
# 4. Visualization Functions
# ---------------------------
def label_to_score(label):
    """Convert binary label (0/1) to sentiment score (-1/1)"""
    return 2 * label - 1


def plot_comparison_charts(sentences, dict_scores, nb_scores, label_scores, title_suffix="", interactive=True):
    """Generate all comparison charts with improved readability"""
    try:
        plot_bar_chart(sentences, dict_scores, nb_scores, label_scores, title_suffix, interactive)
        plot_line_chart(sentences, dict_scores, nb_scores, label_scores, title_suffix, interactive)
        plot_scatter_chart(sentences, dict_scores, nb_scores, label_scores, title_suffix, interactive)
        plot_error_chart(sentences, dict_scores, nb_scores, label_scores, title_suffix, interactive)
    except Exception as e:
        print(f"Error generating charts: {e}")
        print("Charts may not have been displayed or saved due to an error.")


def plot_bar_chart(sentences, dict_scores, nb_scores, label_scores, title_suffix="", interactive=True):
    """Plot bar chart comparing sentiment scores"""
    valid_mask = [bool(s.strip()) for s in sentences]
    valid_sentences = [s for s, v in zip(sentences, valid_mask) if v]
    valid_dict = [s for s, v in zip(dict_scores, valid_mask) if v]
    valid_nb = [s for s, v in zip(nb_scores, valid_mask) if v]
    valid_labels = [s for s, v in zip(label_scores, valid_mask) if v]
    
    if not valid_sentences:
        print("No valid test sentences, skipping chart")
        return
    
    short_sentences = [s[:15] + '...' if len(s) > 15 else s for s in valid_sentences]
    fig, ax = plt.subplots(figsize=(14, 7))  # Larger chart size
    
    x = np.arange(len(valid_sentences))
    width = 0.25
    ax.bar(x - width, valid_dict, width, label='Lexicon-based', color='skyblue', alpha=0.8)
    ax.bar(x, valid_nb, width, label='Naive Bayes', color='salmon', alpha=0.8)
    ax.bar(x + width, valid_labels, width, label='Annotated Score', color='lightgreen', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(short_sentences, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Sentiment Score', fontsize=12)
    ax.set_title(f'Comparison of Sentiment Scores by Three Methods{title_suffix}', fontsize=14)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    chart_file = f'sentiment_bar{title_suffix}.png'
    plt.savefig(chart_file, dpi=300)
    print(f"Bar chart saved as {chart_file}")
    
    # Display the plot
    if IN_JUPYTER:
        plt.show()
    else:
        plt.draw()
        if interactive:
            plt.pause(0.1)
            input("Press Enter to continue after viewing the chart...")
            plt.close()
        else:
            plt.show(block=False)


def plot_line_chart(sentences, dict_scores, nb_scores, label_scores, title_suffix="", interactive=True):
    """Plot line chart showing score trends"""
    valid_mask = [bool(s.strip()) for s in sentences]
    valid_sentences = [s for s, v in zip(sentences, valid_mask) if v]
    valid_dict = [s for s, v in zip(dict_scores, valid_mask) if v]
    valid_nb = [s for s, v in zip(nb_scores, valid_mask) if v]
    valid_labels = [s for s, v in zip(label_scores, valid_mask) if v]
    
    if not valid_sentences:
        print("No valid test sentences, skipping chart")
        return
    
    short_sentences = [s[:15] + '...' if len(s) > 15 else s for s in valid_sentences]
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(short_sentences, valid_dict, marker='o', label='Lexicon-based', color='skyblue', alpha=0.8, linewidth=2)
    ax.plot(short_sentences, valid_nb, marker='s', label='Naive Bayes', color='salmon', alpha=0.8, linewidth=2)
    ax.plot(short_sentences, valid_labels, marker='^', label='Annotated Score', color='lightgreen', alpha=0.8, linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xticklabels(short_sentences, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Sentiment Score', fontsize=12)
    ax.set_title(f'Sentiment Score Trends by Three Methods{title_suffix}', fontsize=14)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    chart_file = f'sentiment_line{title_suffix}.png'
    plt.savefig(chart_file, dpi=300)
    print(f"Line chart saved as {chart_file}")
    
    # Display the plot
    if IN_JUPYTER:
        plt.show()
    else:
        plt.draw()
        if interactive:
            plt.pause(0.1)
            input("Press Enter to continue after viewing the chart...")
            plt.close()
        else:
            plt.show(block=False)


def plot_scatter_chart(sentences, dict_scores, nb_scores, label_scores, title_suffix="", interactive=True):
    """Plot scatter chart for prediction-annotation correlation"""
    valid_mask = [bool(s.strip()) for s in sentences]
    valid_sentences = [s for s, v in zip(sentences, valid_mask) if v]
    valid_dict = [s for s, v in zip(dict_scores, valid_mask) if v]
    valid_nb = [s for s, v in zip(nb_scores, valid_mask) if v]
    valid_labels = [s for s, v in zip(label_scores, valid_mask) if v]
    
    if not valid_sentences:
        print("No valid test sentences, skipping chart")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.scatter(valid_labels, valid_dict, color='skyblue', alpha=0.7, s=80, label='Lexicon vs Annotation')
    ax.scatter(valid_labels, valid_nb, color='salmon', alpha=0.7, s=80, label='Naive Bayes vs Annotation')
    
    min_val = min(min(valid_labels), min(valid_dict), min(valid_nb)) - 0.1
    max_val = max(max(valid_labels), max(valid_dict), max(valid_nb)) + 0.1
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Annotated Score', fontsize=12)
    ax.set_ylabel('Predicted Score', fontsize=12)
    ax.set_title(f'Correlation Between Predicted and Annotated Scores{title_suffix}', fontsize=14)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    chart_file = f'sentiment_scatter{title_suffix}.png'
    plt.savefig(chart_file, dpi=300)
    print(f"Scatter chart saved as {chart_file}")
    
    # Display the plot
    if IN_JUPYTER:
        plt.show()
    else:
        plt.draw()
        if interactive:
            plt.pause(0.1)
            input("Press Enter to continue after viewing the chart...")
            plt.close()
        else:
            plt.show(block=False)


def plot_error_chart(sentences, dict_scores, nb_scores, label_scores, title_suffix="", interactive=True):
    """Plot bar chart comparing prediction errors"""
    valid_mask = [bool(s.strip()) for s in sentences]
    valid_sentences = [s for s, v in zip(sentences, valid_mask) if v]
    valid_dict = [s for s, v in zip(dict_scores, valid_mask) if v]
    valid_nb = [s for s, v in zip(nb_scores, valid_mask) if v]
    valid_labels = [s for s, v in zip(label_scores, valid_mask) if v]
    
    if not valid_sentences:
        print("No valid test sentences, skipping chart")
        return
    
    short_sentences = [s[:15] + '...' if len(s) > 15 else s for s in valid_sentences]
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(valid_sentences))
    width = 0.35
    dict_errors = [abs(d - l) for d, l in zip(valid_dict, valid_labels)]
    nb_errors = [abs(n - l) for n, l in zip(valid_nb, valid_labels)]
    
    ax.bar(x - width/2, dict_errors, width, label='Lexicon Error', color='skyblue', alpha=0.8)
    ax.bar(x + width/2, nb_errors, width, label='Naive Bayes Error', color='salmon', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_sentences, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title(f'Comparison of Prediction Errors by Two Methods{title_suffix}', fontsize=14)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    chart_file = f'sentiment_error{title_suffix}.png'
    plt.savefig(chart_file, dpi=300)
    print(f"Error chart saved as {chart_file}")
    
    # Display the plot
    if IN_JUPYTER:
        plt.show()
    else:
        plt.draw()
        if interactive:
            plt.pause(0.1)
            input("Press Enter to continue after viewing the chart...")
            plt.close()
        else:
            plt.show(block=False)


def plot_feature_importance(model, vectorizer, top_n=20):
    """Plot top positive and negative feature weights"""
    if model is None or vectorizer is None:
        print("Model or vectorizer not available, cannot plot feature importance")
        return
    
    try:
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.feature_log_prob_[1] - model.feature_log_prob_[0]  # Positive - Negative
        
        # Sort features by coefficient
        top_positive_indices = np.argsort(coefs)[-top_n:]
        top_negative_indices = np.argsort(coefs)[:top_n]
        
        top_positive_features = [feature_names[i] for i in top_positive_indices]
        top_positive_coefs = [coefs[i] for i in top_positive_indices]
        
        top_negative_features = [feature_names[i] for i in top_negative_indices]
        top_negative_coefs = [coefs[i] for i in top_negative_indices]
        
        # Plot positive features
        plt.figure(figsize=(12, 8))
        plt.barh(top_positive_features, top_positive_coefs, color='skyblue')
        plt.title(f'Top {top_n} Positive Features')
        plt.xlabel('Feature Coefficient')
        plt.tight_layout()
        plt.savefig('positive_features.png', dpi=300)
        
        # Plot negative features
        plt.figure(figsize=(12, 8))
        plt.barh(top_negative_features, top_negative_coefs, color='salmon')
        plt.title(f'Top {top_n} Negative Features')
        plt.xlabel('Feature Coefficient')
        plt.tight_layout()
        plt.savefig('negative_features.png', dpi=300)
        
        print("Feature importance plots saved as positive_features.png and negative_features.png")
        
    except Exception as e:
        print(f"Error plotting feature importance: {e}")


# ---------------------------
# 5. Interactive Functions
# ---------------------------
def show_main_menu():
    """Display the main menu"""
    print("\n" + "=" * 60)
    print("Sentiment Analysis System (Multinomial Naive Bayes Version)")
    print("=" * 60)
    
    model_status = "Not Trained" if model is None else "Trained"
    print(f"Naive Bayes Model Status: {model_status}")
    
    print("1. Analyze Single Text with Lexicon-based Method")
    print("2. Analyze Single Text with Naive Bayes")
    print("3. Compare Both Methods on Multiple Texts")
    print("4. Train Naive Bayes Model")
    print("5. View Model Evaluation Results")
    print("6. Show Feature Importance")
    print("7. Save Model")
    print("8. Load Saved Model")
    print("9. Exit Program")
    print("=" * 60)


def get_user_choice():
    """Get user input with validation"""
    while True:
        try:
            choice = int(input("Enter your choice (1-9): "))
            if 1 <= choice <= 9:
                return choice
            else:
                print("Invalid choice, enter 1-9")
        except ValueError:
            print("Please enter a number")


def analyze_single_text(stopwords, negation_words, degree_dict, sentiment_dict, model, vectorizer):
    """Analyze sentiment of a single text with detailed breakdown"""
    print("\n--- Single Text Sentiment Analysis ---")
    text = input("Enter text to analyze (type 'q' to return to menu): ")
    if text.lower() == 'q':
        return
    
    # Lexicon-based score
    dict_score = sentiment_score_by_rule(text, stopwords, negation_words, degree_dict, sentiment_dict)
    
    # Tokenize and show breakdown
    tokens = seg_word(text, stopwords)
    print("\nTokenized text:", ' '.join(tokens))
    
    # Show lexicon breakdown
    print("\nLexicon-based analysis breakdown:")
    for i, token in enumerate(tokens):
        if token in sentiment_dict:
            print(f"  '{token}': Sentiment score = {sentiment_dict[token]:.2f}")
        elif token in negation_words:
            print(f"  '{token}': Negation word")
        elif token in degree_dict:
            print(f"  '{token}': Degree modifier (weight = {degree_dict[token]:.2f})")
    
    # Naive Bayes score (if model is trained)
    nb_score = predict_sentiment_score(model, vectorizer, text, stopwords) if model else 0.0
    
    print(f"\nText: {text}")
    print(f"Lexicon-based score: {dict_score:.4f}")
    if model:
        print(f"Naive Bayes score: {nb_score:.4f}")
    
    # Interpret the score
    interpretation = "Sentiment tendency: "
    if dict_score > 0.3 or (model and nb_score > 0.3):
        interpretation += "Strongly Positive"
    elif dict_score > 0 or (model and nb_score > 0):
        interpretation += "Mildly Positive"
    elif dict_score < -0.3 or (model and nb_score < -0.3):
        interpretation += "Strongly Negative"
    else:
        interpretation += "Neutral"
    print(interpretation)


def compare_methods_on_custom_texts(stopwords, negation_words, degree_dict, sentiment_dict, model, vectorizer, dataset_df):
    """Compare both methods on custom texts"""
    if model is None:
        print("Naive Bayes model not trained, cannot use this feature")
        input("Press Enter to continue...")
        return
    
    print("\n--- Multiple Texts Sentiment Analysis Comparison ---")
    print("Enter texts to analyze (one per line, type 'q' to finish):")
    
    texts = []
    while True:
        text = input()
        if text.lower() == 'q':
            break
        if text.strip():
            texts.append(text)
    
    if not texts:
        print("No texts entered, returning to menu")
        return
    
    # Prepare data
    dict_scores, nb_scores = [], []
    
    # Analyze each text
    print("\nAnalyzing texts...")
    for i, text in enumerate(texts):
        print(f"Analyzing text {i+1}/{len(texts)}: {text[:50]}...")
        
        dict_score = sentiment_score_by_rule(text, stopwords, negation_words, degree_dict, sentiment_dict)
        nb_score = predict_sentiment_score(model, vectorizer, text, stopwords)
        
        dict_scores.append(dict_score)
        nb_scores.append(nb_score)
        
        print(f"  Lexicon: {dict_score:.4f} | Naive Bayes: {nb_score:.4f}")
    
    # Generate charts
    plot_option = input("\nGenerate comparison charts? (y/n): ").lower()
    if plot_option == 'y':
        # Generate random labels for demonstration (in real use, use actual labels)
        print("\nSince no annotated labels are provided, generating random labels for demonstration...")
        label_scores = [label_to_score(np.random.randint(0, 2)) for _ in texts]
        plot_comparison_charts(texts, dict_scores, nb_scores, label_scores, " (Custom Texts)")
    
    # Show comparison summary
    avg_dict = sum(dict_scores) / len(dict_scores)
    avg_nb = sum(nb_scores) / len(nb_scores)
    
    print("\nComparison Summary:")
    print(f"Average lexicon-based score: {avg_dict:.4f}")
    print(f"Average Naive Bayes score: {avg_nb:.4f}")
    
    if avg_dict > 0 and avg_nb > 0:
        print("Overall sentiment: Positive")
    elif avg_dict < 0 and avg_nb < 0:
        print("Overall sentiment: Negative")
    else:
        print("Mixed sentiment detected")


def train_model(stopwords, dataset_df):
    """Train Naive Bayes model with confirmation and option to use saved model"""
    global model, vectorizer
    
    # Check if saved model exists
    if os.path.exists('naive_bayes_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
        use_saved = input("Saved model found. Use saved model instead of training? (y/n): ").lower()
        if use_saved == 'y':
            model, vectorizer = load_model_from_file()
            if model and vectorizer:
                print("Using previously saved model.")
            return
    
    print("\n=== Train Naive Bayes Model ===")
    
    if dataset_df is None:
        print("Error: Dataset not loaded. Ensure sentiment_dataset.tsv exists.")
        input("Press Enter to continue...")
        return
    
    confirm = input("About to train the model, this may take some time. Proceed? (y/n): ").lower()
    if confirm != 'y':
        print("Training canceled.")
        input("Press Enter to continue...")
        return
    
    model, vectorizer = train_naive_bayes_model(stopwords, dataset_df)
    
    if model:
        print("\nModel training completed!")
        print(f"Naive Bayes model is now ready for sentiment analysis.")
    else:
        print("Model training failed. Check dataset and try again.")
    
    input("Press Enter to continue...")


def show_model_evaluation(stopwords, dataset_df, model, vectorizer):
    """Show detailed model evaluation results with additional insights"""
    print("\n--- Model Evaluation Results ---")
    
    if model is None or vectorizer is None:
        print("Model not trained, cannot show evaluation results")
        input("Press Enter to continue...")
        return
    
    # Select a random subset of test samples for demonstration
    test_df = dataset_df.sample(min(20, len(dataset_df)), random_state=42)
    test_sentences = test_df['text'].tolist()
    true_labels = test_df['sentiment'].tolist()
    label_scores = [label_to_score(l) for l in true_labels]
    
    # Calculate prediction scores
    print("Calculating predictions...")
    dict_scores = [sentiment_score_by_rule(s, stopwords, negation_words, degree_dict, sentiment_dict) for s in test_sentences]
    nb_scores = [predict_sentiment_score(model, vectorizer, s, stopwords) for s in test_sentences]
    
    # Display sample predictions
    print("\nSample Predictions:")
    for i, (s, d, n, l) in enumerate(zip(test_sentences[:5], dict_scores[:5], nb_scores[:5], true_labels[:5])):
        label_text = "Positive" if l == 1 else "Negative"
        dict_interpret = "Positive" if d > 0 else "Negative"
        nb_interpret = "Positive" if n > 0 else "Negative"
        
        print(f"\n{i+1}. Text: {s[:60]}...")
        print(f"   Actual label: {label_text}")
        print(f"   Lexicon-based: {d:.4f} ({dict_interpret})")
        print(f"   Naive Bayes: {n:.4f} ({nb_interpret})")
    
    # Calculate average errors
    dict_errors = [abs(d - l) for d, l in zip(dict_scores, label_scores)]
    nb_errors = [abs(n - l) for n, l in zip(nb_scores, label_scores)]
    dict_avg_error = np.mean(dict_errors)
    nb_avg_error = np.mean(nb_errors)
    
    print(f"\nAverage prediction errors:")
    print(f"  Lexicon-based: {dict_avg_error:.4f}")
    print(f"  Naive Bayes: {nb_avg_error:.4f}")
    
    # Compare the two methods
    better_method = "Lexicon-based" if dict_avg_error < nb_avg_error else "Naive Bayes"
    print(f"\n{better_method} has lower average error ({'%.4f' % min(dict_avg_error, nb_avg_error)})")
    
    # Generate charts
    plot_option = input("\nGenerate evaluation charts? (y/n): ").lower()
    if plot_option == 'y':
        plot_comparison_charts(test_sentences, dict_scores, nb_scores, label_scores, " (Model Evaluation)")
    
    # Show feature importance
    feature_option = input("Show feature importance plots? (y/n): ").lower()
    if feature_option == 'y':
        plot_feature_importance(model, vectorizer)
    
    input("Press Enter to continue...")


# ---------------------------
# 6. Main Program
# ---------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Initializing Sentiment Analysis System with Naive Bayes...")
    print("=" * 60)
    
    # Global model and vectorizer
    model, vectorizer = None, None
    
    # Load sentiment analysis resources
    print("Loading sentiment analysis resources...")
    stopwords, negation_words, degree_dict, sentiment_dict = load_sentiment_resources()
    
    # Load dataset
    print("Loading dataset...")
    dataset_df = load_custom_dataset('sentiment_dataset.tsv')
    
    # Try to load saved model
    model, vectorizer = load_model_from_file()
    
    # Interactive main loop
    while True:
        show_main_menu()
        choice = get_user_choice()
        
        if choice == 1:
            # Lexicon-based single text analysis
            analyze_single_text(stopwords, negation_words, degree_dict, sentiment_dict, model, vectorizer)
        
        elif choice == 2:
            # Naive Bayes single text analysis
            if model is None:
                print("Naive Bayes model not trained, using lexicon-based method only")
            analyze_single_text(stopwords, negation_words, degree_dict, sentiment_dict, model, vectorizer)
        
        elif choice == 3:
            # Compare both methods
            compare_methods_on_custom_texts(stopwords, negation_words, degree_dict, sentiment_dict, model, vectorizer, dataset_df)
        
        elif choice == 4:
            # Train model
            train_model(stopwords, dataset_df)
        
        elif choice == 5:
            # Show model evaluation
            show_model_evaluation(stopwords, dataset_df, model, vectorizer)
        
        elif choice == 6:
            # Show feature importance
            if model and vectorizer:
                plot_feature_importance(model, vectorizer)
            else:
                print("Model not trained, cannot show feature importance")
            input("Press Enter to continue...")
        
        elif choice == 7:
            # Save model
            if model and vectorizer:
                save_model_to_file(model, vectorizer)
            else:
                print("No trained model to save")
            input("Press Enter to continue...")
        
        elif choice == 8:
            # Load model
            model, vectorizer = load_model_from_file()
            input("Press Enter to continue...")
        
        elif choice == 9:
            # Exit program
            print("\nThank you for using the Sentiment Analysis System. Goodbye!")
            break
    
    print("=" * 60)