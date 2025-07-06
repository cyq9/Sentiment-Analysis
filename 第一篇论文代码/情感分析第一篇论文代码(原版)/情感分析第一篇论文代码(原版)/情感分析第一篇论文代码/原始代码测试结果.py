import os
import re
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

# Ignore irrelevant warnings
warnings.filterwarnings('ignore')

# Set font for Chinese characters and style
sns.set(font="Arial", style="whitegrid")  
plt.rcParams["axes.unicode_minus"] = False  # Ensure negative signs display correctly

# Detect execution environment for plotting optimization
try:
    get_ipython
    IN_JUPYTER = True
except NameError:
    IN_JUPYTER = False
    plt.switch_backend('TkAgg')  # Use interactive backend for non-Jupyter environments


# ---------------------------
# 1. Utility Functions: File and Text Processing
# ---------------------------
def load_file(file_path, default_content=None, encoding='utf-8'):
    """Universal file loading function with error handling"""
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
    negation_words = load_file('negation_words.txt', default_content=["不", "没", "没有", "别", "莫", "勿"])  # Keep Chinese negation words as is
    degree_dict = defaultdict(float)
    for line in load_file('degree_adverbs.txt', default_content=[]):
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
    """Rule-based sentiment analysis scoring"""
    seg_list = seg_word(sentence, stopwords)
    if not seg_list:
        return 0.0

    sentiment_words, negation_marks, degree_marks = defaultdict(float), defaultdict(int), defaultdict(float)
    for i, word in enumerate(seg_list):
        if word in sentiment_dict:
            sentiment_words[i] = sentiment_dict[word]
        elif word in negation_words:
            negation_marks[i] = -1
        elif word in degree_dict:
            degree_marks[i] = degree_dict[word]

    score = 0.0
    weight = 1.0
    sentiment_indices = sorted(sentiment_words.keys())
    for i in range(len(seg_list)):
        if i in sentiment_words:
            score += weight * sentiment_words[i]
            next_sen_idx = None
            for idx in sentiment_indices:
                if idx > i:
                    next_sen_idx = idx
                    break
            next_sen_idx = next_sen_idx or len(seg_list)
            for j in range(i + 1, next_sen_idx):
                if j in negation_marks:
                    weight *= -1
                elif j in degree_marks:
                    weight *= degree_marks[j]
            weight = 1.0
    return score


# ---------------------------
# 3. Machine Learning Model Training
# ---------------------------
def load_50_dataset(dataset_file='sentiment_dataset.tsv'):
    """Load 50-sentence sentiment analysis dataset"""
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file {dataset_file} not found")
        print("Example format (text\tsentiment label):\nThis movie is amazing!\t1\nPoor service attitude\t0")
        return None
    try:
        df = pd.read_csv(dataset_file, sep='\t')
        print(f"Dataset loaded successfully: {len(df)} records")
        return df
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None


def train_models(stopwords, dataset_df):
    """Train all three models: Multinomial Naive Bayes, Logistic Regression, Linear Regression"""
    if dataset_df is None:
        return None, None, None, None
    
    # Data preprocessing
    dataset_df['tokenized_text'] = dataset_df['text'].apply(lambda x: ' '.join(seg_word(x, stopwords)))
    
    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(dataset_df['tokenized_text'])
    
    # Classification labels (0/1)
    y_class = dataset_df['sentiment'].astype(int)
    # Regression labels (-1/1)
    y_reg = dataset_df['sentiment'].astype(float).apply(lambda x: 2 * x - 1)
    
    # Split into training and test sets (80% train, 20% test)
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # 1. Multinomial Naive Bayes model (classification)
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_class_train)
    print("\n=== Multinomial Naive Bayes Model Evaluation ===")
    nb_pred = nb_model.predict(X_test)
    print(classification_report(y_class_test, nb_pred))
    print(f"Accuracy: {accuracy_score(y_class_test, nb_pred):.4f}")
    
    # 2. Logistic Regression model (classification)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_class_train)
    print("\n=== Logistic Regression Model Evaluation ===")
    lr_pred = lr_model.predict(X_test)
    print(classification_report(y_class_test, lr_pred))
    print(f"Accuracy: {accuracy_score(y_class_test, lr_pred):.4f}")
    
    # 3. Linear Regression model (regression)
    linreg_model = LinearRegression()
    linreg_model.fit(X_train, y_reg_train)
    print("\n=== Linear Regression Model Evaluation ===")
    linreg_pred = linreg_model.predict(X_test)
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_reg_test, linreg_pred):.4f}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_reg_test, linreg_pred):.4f}")
    print(f"Coefficient of Determination (R²): {r2_score(y_reg_test, linreg_pred):.4f}")
    
    return nb_model, lr_model, linreg_model, vectorizer


# ---------------------------
# 4. Model Prediction Functions
# ---------------------------
def predict_nb_score(model, vectorizer, sentence, stopwords):
    """Predict sentiment score using Multinomial Naive Bayes (mapped to [-1, 1])"""
    if not sentence:
        return 0.0
    tokenized_text = ' '.join(seg_word(sentence, stopwords))
    try:
        X = vectorizer.transform([tokenized_text])
        proba = model.predict_proba(X)[0][1]
        return 2 * proba - 1  # Map probability to [-1, 1]
    except Exception as e:
        print(f"Multinomial Naive Bayes prediction error: {e}")
        return 0.0


def predict_lr_score(model, vectorizer, sentence, stopwords):
    """Predict sentiment score using Logistic Regression (mapped to [-1, 1])"""
    if not sentence:
        return 0.0
    tokenized_text = ' '.join(seg_word(sentence, stopwords))
    try:
        X = vectorizer.transform([tokenized_text])
        proba = model.predict_proba(X)[0][1]
        return 2 * proba - 1  # Map probability to [-1, 1]
    except Exception as e:
        print(f"Logistic Regression prediction error: {e}")
        return 0.0


def predict_linreg_score(model, vectorizer, sentence, stopwords):
    """Predict sentiment score using Linear Regression (clamped to [-1, 1])"""
    if not sentence:
        return 0.0
    tokenized_text = ' '.join(seg_word(sentence, stopwords))
    try:
        X = vectorizer.transform([tokenized_text])
        score = model.predict(X)[0]
        return max(-1.0, min(1.0, score))  # Clamp to [-1, 1] range
    except Exception as e:
        print(f"Linear Regression prediction error: {e}")
        return 0.0


# ---------------------------
# 5. Ensemble Model
# ---------------------------
def ensemble_predict(sentence, stopwords, negation_words, degree_dict, sentiment_dict, 
                     nb_model, lr_model, linreg_model, vectorizer, 
                     weights=None):
    """Ensemble prediction from four methods: rule-based + three machine learning models"""
    if weights is None:
        # Default weights: rule-based, Multinomial NB, Logistic Regression, Linear Regression
        weights = [0, 0.1, 0.1, 0.8]
    
    # Calculate scores from each method
    dict_score = sentiment_score_by_rule(sentence, stopwords, negation_words, degree_dict, sentiment_dict)
    nb_score = predict_nb_score(nb_model, vectorizer, sentence, stopwords)
    lr_score = predict_lr_score(lr_model, vectorizer, sentence, stopwords)
    linreg_score = predict_linreg_score(linreg_model, vectorizer, sentence, stopwords)
    
    # Weighted average
    final_score = (
        weights[0] * dict_score + 
        weights[1] * nb_score + 
        weights[2] * lr_score + 
        weights[3] * linreg_score
    )
    
    # Clamp to [-1, 1] range
    final_score = max(-1.0, min(1.0, final_score))
    
    return {
        'dict_score': dict_score,
        'nb_score': nb_score,
        'lr_score': lr_score,
        'linreg_score': linreg_score,
        'final_score': final_score
    }


# ---------------------------
# 6. Visualization and Evaluation
# ---------------------------
def label_to_score(label):
    """Convert annotated label (0/1) to sentiment score (-1/1)"""
    return 2 * label - 1


def evaluate_ensemble(test_df, stopwords, negation_words, degree_dict, sentiment_dict, 
                      nb_model, lr_model, linreg_model, vectorizer, weights=None):
    """Evaluate ensemble model performance and generate visualization charts"""
    true_scores = test_df['sentiment'].apply(label_to_score).tolist()
    dict_scores, nb_scores, lr_scores, linreg_scores, ensemble_scores = [], [], [], [], []
    sentences = test_df['text'].tolist()
    
    # Calculate scores for each model
    for i, row in test_df.iterrows():
        sentence = row['text']
        scores = ensemble_predict(sentence, stopwords, negation_words, degree_dict, sentiment_dict,
                                 nb_model, lr_model, linreg_model, vectorizer, weights)
        
        dict_scores.append(scores['dict_score'])
        nb_scores.append(scores['nb_score'])
        lr_scores.append(scores['lr_score'])
        linreg_scores.append(scores['linreg_score'])
        ensemble_scores.append(scores['final_score'])
    
    # Calculate errors
    dict_error = mean_absolute_error(true_scores, dict_scores)
    nb_error = mean_absolute_error(true_scores, nb_scores)
    lr_error = mean_absolute_error(true_scores, lr_scores)
    linreg_error = mean_absolute_error(true_scores, linreg_scores)
    ensemble_error = mean_absolute_error(true_scores, ensemble_scores)
    
    # Identify the best model
    model_errors = {
        'Rule-based': dict_error,
        'Multinomial NB': nb_error,
        'Logistic Regression': lr_error,
        'Linear Regression': linreg_error,
        'Ensemble Model': ensemble_error
    }
    best_model = min(model_errors, key=model_errors.get)
    
    # Plot comparison charts
    plot_comparison_charts(sentences, dict_scores, nb_scores, lr_scores, linreg_scores, 
                          ensemble_scores, true_scores, model_errors, best_model)
    
    return {
        'dict_error': dict_error,
        'nb_error': nb_error,
        'lr_error': lr_error,
        'linreg_error': linreg_error,
        'ensemble_error': ensemble_error,
        'best_model': best_model
    }


def plot_comparison_charts(sentences, dict_scores, nb_scores, lr_scores, linreg_scores, 
                          ensemble_scores, true_scores, model_errors, best_model):
    """Plot various comparison charts with optimized display logic"""
    # Display only first 10 sentences to avoid overcrowding
    max_display = min(10, len(sentences))
    short_sentences = [s[:15] + '...' if len(s) > 15 else s for s in sentences[:max_display]]
    
    # Create figure
    fig = plt.figure(figsize=(18, 15))
    plt.suptitle('Sentiment Analysis Model Comparison and Evaluation', fontsize=16, y=0.99)
    
    # 1. Error comparison bar chart
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    models = list(model_errors.keys())
    errors = list(model_errors.values())
    colors = ['skyblue', 'lightgreen', 'salmon', 'lightyellow', 'plum']
    
    bars = ax1.bar(models, errors, color=colors)
    ax1.set_ylabel('Mean Absolute Error (MAE)')
    ax1.set_title('Error Comparison of Models')
    ax1.tick_params(axis='x', rotation=45)
    
    # Mark the best model
    for bar, model in zip(bars, models):
        if model == best_model:
            bar.set_color('red')
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'Best: {height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Score comparison bar chart
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    x = np.arange(max_display)
    width = 0.15
    
    ax2.bar(x - 2*width, dict_scores[:max_display], width, label='Rule-based')
    ax2.bar(x - width, nb_scores[:max_display], width, label='Multinomial NB')
    ax2.bar(x, lr_scores[:max_display], width, label='Logistic Regression')
    ax2.bar(x + width, linreg_scores[:max_display], width, label='Linear Regression')
    ax2.bar(x + 2*width, ensemble_scores[:max_display], width, label='Ensemble Model')
    ax2.plot(x, true_scores[:max_display], 'ko-', label='Annotated Score')
    
    ax2.set_ylabel('Sentiment Score')
    ax2.set_title('Sentiment Score Comparison of Models')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_sentences, rotation=45, ha='right')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # 3. Correlation heatmap
    ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    data = {
        'Annotated Score': true_scores,
        'Rule-based': dict_scores,
        'Multinomial NB': nb_scores,
        'Logistic Regression': lr_scores,
        'Linear Regression': linreg_scores,
        'Ensemble Model': ensemble_scores
    }
    corr_df = pd.DataFrame(data)
    corr_matrix = corr_df.corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, ax=ax3)
    ax3.set_title('Correlation Between Model Scores and Annotated Scores')
    
    # 4. Error distribution boxplot
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    dict_errors = [abs(d - t) for d, t in zip(dict_scores, true_scores)]
    nb_errors = [abs(n - t) for n, t in zip(nb_scores, true_scores)]
    lr_errors = [abs(l - t) for l, t in zip(lr_scores, true_scores)]
    linreg_errors = [abs(lr - t) for lr, t in zip(linreg_scores, true_scores)]
    ensemble_errors = [abs(e - t) for e, t in zip(ensemble_scores, true_scores)]
    
    boxplot_data = [dict_errors, nb_errors, lr_errors, linreg_errors, ensemble_errors]
    box = ax4.boxplot(boxplot_data, labels=models, patch_artist=True)
    
    # Set boxplot colors
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('Error Distribution of Models')
    
    # 5. Best model table
    ax5 = plt.subplot2grid((3, 2), (2, 1))
    ax5.axis('off')
    
    # Sort models by performance
    sorted_models = sorted(model_errors.items(), key=lambda x: x[1])
    
    # Create model performance table
    cell_text = [[f"{model}", f"{error:.4f}"] for model, error in sorted_models]
    table = ax5.table(cellText=cell_text,
                      colLabels=['Model', 'Mean Absolute Error'],
                      loc='center',
                      cellLoc='center')
    
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    ax5.set_title('Model Performance Ranking')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make space for the title
    plt.savefig('sentiment_model_comparison.png', dpi=300, bbox_inches='tight')
    print("Chart saved as sentiment_model_comparison.png")
    
    # Display chart based on environment
    if IN_JUPYTER:
        plt.show()
    else:
        plt.show()
        input("Press Enter to continue...")
        plt.close()


# ---------------------------
# 7. Interactive Functions
# ---------------------------
def show_main_menu():
    """Show main menu"""
    print("\n" + "=" * 60)
    print("Sentiment Analysis Ensemble Model System - Supporting Four Analysis Methods")
    print("=" * 60)
    print("1. Single Text Sentiment Analysis (Show all model results)")
    print("2. Multi-Text Model Comparison and Visualization")
    print("3. Custom Weight Ensemble Model Evaluation")
    print("4. View Model Performance Report")
    print("5. Exit Program")
    print("=" * 60)


def get_user_choice():
    """Get user choice"""
    while True:
        try:
            choice = int(input("Enter your choice (1-5): "))
            if 1 <= choice <= 5:
                return choice
            else:
                print("Invalid choice, please enter a number between 1 and 5")
        except ValueError:
            print("Please enter a valid number")


def analyze_single_text(stopwords, negation_words, degree_dict, sentiment_dict, 
                       nb_model, lr_model, linreg_model, vectorizer):
    """Single text sentiment analysis"""
    print("\n--- Single Text Sentiment Analysis ---")
    text = input("Enter text to analyze (type 'q' to return to menu): ")
    if text.lower() == 'q':
        return
    
    # Ensemble prediction
    scores = ensemble_predict(text, stopwords, negation_words, degree_dict, sentiment_dict,
                             nb_model, lr_model, linreg_model, vectorizer)
    
    # Display results
    print(f"\nAnalyzing text: {text}")
    print(f"[Rule-based] Sentiment score: {scores['dict_score']:.4f}")
    print(f"[Multinomial NB] Sentiment score: {scores['nb_score']:.4f}")
    print(f"[Logistic Regression] Sentiment score: {scores['lr_score']:.4f}")
    print(f"[Linear Regression] Sentiment score: {scores['linreg_score']:.4f}")
    print(f"[Ensemble Model] Final score: {scores['final_score']:.4f}")
    
    # Interpret sentiment
    final_score = scores['final_score']
    interpretation = "Sentiment tendency: "
    if final_score > 0.3:
        interpretation += "Strongly Positive"
    elif final_score > 0:
        interpretation += "Mildly Positive"
    elif final_score < -0.3:
        interpretation += "Strongly Negative"
    else:
        interpretation += "Neutral"
    print(interpretation)


def compare_models_on_texts(stopwords, negation_words, degree_dict, sentiment_dict, 
                          nb_model, lr_model, linreg_model, vectorizer, dataset_df):
    """Compare models on custom texts"""
    print("\n--- Multi-Text Model Comparison ---")
    print("Enter texts to analyze (one per line, press Enter to finish):")
    
    texts = []
    while True:
        text = input()
        if not text:
            break
        texts.append(text)
    
    if not texts:
        print("No texts entered, returning to menu")
        return
    
    # Analyze texts
    results = []
    print("\nAnalyzing texts...")
    for i, text in enumerate(texts):
        print(f"Analyzing text {i+1}/{len(texts)}: {text[:30]}...")
        scores = ensemble_predict(text, stopwords, negation_words, degree_dict, sentiment_dict,
                                 nb_model, lr_model, linreg_model, vectorizer)
        results.append(scores)
        print(f"  Ensemble score: {scores['final_score']:.4f}")
    
    # Generate charts
    plot_option = input("\nGenerate comparison charts? (y/n): ").lower()
    if plot_option == 'y':
        # Use test set annotations as reference
        _, test_df = train_test_split(dataset_df, test_size=0.2, random_state=42)
        evaluate_ensemble(test_df, stopwords, negation_words, degree_dict, sentiment_dict,
                         nb_model, lr_model, linreg_model, vectorizer)


def custom_ensemble_evaluation(stopwords, negation_words, degree_dict, sentiment_dict, 
                              nb_model, lr_model, linreg_model, vectorizer, dataset_df):
    """Evaluate ensemble model with custom weights"""
    print("\n--- Custom Weight Ensemble Model Evaluation ---")
    print("Enter weights for the four methods (space-separated, sum to 1)")
    print("Example: 0.1 0.1 0.1 0.7 (Rule-based MultinomialNB LogisticRegression LinearRegression)")
    
    while True:
        try:
            weights_input = input("Enter weights: ")
            weights = [float(w) for w in weights_input.split()]
            if len(weights) != 4:
                print("Please enter 4 weight values")
                continue
            if abs(sum(weights) - 1.0) > 0.01:
                print("Weights must sum to 1")
                continue
            break
        except ValueError:
            print("Please enter valid numbers")
    
    print(f"\nUsing weights: Rule={weights[0]}, NB={weights[1]}, LR={weights[2]}, LinReg={weights[3]}")
    
    # Evaluate ensemble model
    _, test_df = train_test_split(dataset_df, test_size=0.2, random_state=42)
    results = evaluate_ensemble(test_df, stopwords, negation_words, degree_dict, sentiment_dict,
                             nb_model, lr_model, linreg_model, vectorizer, weights)
    
    print("\n=== Evaluation Results ===")
    print(f"Best model: {results['best_model']}")
    print(f"Ensemble model MAE: {results['ensemble_error']:.4f}")


def show_model_performance(stopwords, negation_words, degree_dict, sentiment_dict, 
                         nb_model, lr_model, linreg_model, vectorizer, dataset_df):
    """Show model performance report"""
    print("\n--- Model Performance Report ---")
    _, test_df = train_test_split(dataset_df, test_size=0.2, random_state=42)
    results = evaluate_ensemble(test_df, stopwords, negation_words, degree_dict, sentiment_dict,
                             nb_model, lr_model, linreg_model, vectorizer)
    
    print("\n=== Model Mean Absolute Errors ===")
    print(f"Rule-based: {results['dict_error']:.4f}")
    print(f"Multinomial NB: {results['nb_error']:.4f}")
    print(f"Logistic Regression: {results['lr_error']:.4f}")
    print(f"Linear Regression: {results['linreg_error']:.4f}")
    print(f"Ensemble Model: {results['ensemble_error']:.4f}")
    print(f"Best model: {results['best_model']}")


# ---------------------------
# 8. Main Program
# ---------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Sentiment Analysis Ensemble Model System - Supporting Rule-based and Three Machine Learning Models")
    print("=" * 60)
    
    # 1. Load sentiment lexicons
    print("\n=== Loading Sentiment Lexicons ===")
    stopwords, negation_words, degree_dict, sentiment_dict = load_sentiment_resources()
    
    # 2. Load dataset
    print("\n=== Loading Dataset ===")
    dataset_df = load_50_dataset('sentiment_dataset.tsv')
    if dataset_df is None:
        print("Dataset loading failed, program exiting")
        exit(1)
    
    # 3. Train all models
    print("\n=== Training Machine Learning Models ===")
    nb_model, lr_model, linreg_model, vectorizer = train_models(stopwords, dataset_df)
    if None in [nb_model, lr_model, linreg_model, vectorizer]:
        print("Model training failed, program exiting")
        exit(1)
    
    print("\n=== System Initialization Complete ===")
    print("Supports four sentiment analysis methods: Rule-based, Multinomial Naive Bayes, Logistic Regression, Linear Regression")
    
    # 4. Interactive loop
    while True:
        show_main_menu()
        choice = get_user_choice()
        
        if choice == 1:
            analyze_single_text(stopwords, negation_words, degree_dict, sentiment_dict,
                              nb_model, lr_model, linreg_model, vectorizer)
        
        elif choice == 2:
            compare_models_on_texts(stopwords, negation_words, degree_dict, sentiment_dict,
                                 nb_model, lr_model, linreg_model, vectorizer, dataset_df)
        
        elif choice == 3:
            custom_ensemble_evaluation(stopwords, negation_words, degree_dict, sentiment_dict,
                                    nb_model, lr_model, linreg_model, vectorizer, dataset_df)
        
        elif choice == 4:
            show_model_performance(stopwords, negation_words, degree_dict, sentiment_dict,
                                 nb_model, lr_model, linreg_model, vectorizer, dataset_df)
        
        elif choice == 5:
            print("\nThank you for using the Sentiment Analysis Ensemble Model System. Goodbye!")
            break
    
    print("=" * 60)