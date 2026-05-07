import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def task1_text_classification():
    print("\n" + "=" * 60)
    print("TASK 1: TEXT CLASSIFICATION - Sentiment Analysis")
    print("=" * 60)
    
    training_texts = [
        "I love this movie, it's amazing!",
        "Great film, wonderful acting",
        "Hated every minute of it",
        "Terrible, waste of time and money",
        "Absolutely fantastic experience",
        "Not good at all, very disappointing"
    ]
    training_labels = ["positive", "positive", "negative", "negative", "positive", "negative"]
    
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(training_texts, training_labels)
    
    test_texts = [
        "The plot was boring and the acting was poor",
        "Fantastic movie, I really enjoyed it",
        "It was okay, nothing special"
    ]
    
    print("\nTraining Data (6 reviews):")
    for text, label in zip(training_texts, training_labels):
        print(f"  {label:8} | {text}")
    
    print("\nPredictions on New Reviews:")
    for text in test_texts:
        prediction = model.predict([text])[0]
        print(f"  {prediction:8} | {text}")
    
    print("\n✅ Text Classification: This demonstrates how computers automatically categorize")
    print("   text into groups (positive/negative) - used in spam detection, sentiment analysis")

def task2_token_classification():
    print("\n" + "=" * 60)
    print("TASK 2: TOKEN CLASSIFICATION - Named Entity Recognition (NER)")
    print("=" * 60)
    
    sentences = [
        "Apple is based in California.",
        "Elon Musk founded Tesla in Texas.",
        "Microsoft CEO Satya Nadella visited India last month."
    ]
    
    print("\nToken Classification (each word gets a label):")
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)
        print(f"\n  Sentence: {sentence}")
        print(f"  Word -> Part-of-Speech Tags:")
        for word, tag in pos_tags[:5]:
            print(f"    '{word}' -> {tag}")
        entities = []
        for i, (word, tag) in enumerate(pos_tags):
            if tag == 'NNP':
                if i + 1 < len(pos_tags) and pos_tags[i+1][1] == 'NNP':
                    entities.append(word + " " + pos_tags[i+1][0])
                else:
                    entities.append(word)
        if entities:
            print(f"  Detected Entities: {', '.join(set(entities))}")
    
    print("\n✅ Token Classification: This labels each word/token - used in Named Entity")
    print("   Recognition to find names, places, companies in text")

def task3_question_answering():
    print("\n" + "=" * 60)
    print("TASK 3: QUESTION ANSWERING")
    print("=" * 60)
    
    context = """
    The Eiffel Tower is located in Paris, France. It was completed in 1889 
    and stands 330 meters tall. Alexandre Gustave Eiffel was the engineer 
    who designed this famous landmark. The tower was built for the 1889 
    World's Fair and has become a global cultural icon.
    """
    
    questions = [
        "Where is the Eiffel Tower located?",
        "Who designed the Eiffel Tower?",
        "When was it completed?",
        "How tall is the Eiffel Tower?"
    ]
    
    def simple_answer(question, text):
        text_lower = text.lower()
        question_lower = question.lower()
        
        if "located" in question_lower:
            match = re.search(r'located in ([\w\s,]+)\.', text)
            if match:
                return match.group(1).strip()
        elif "who" in question_lower:
            if "design" in question_lower:
                match = re.search(r'([\w\s]+) was the engineer', text)
                if match:
                    return match.group(1).strip()
        elif "when" in question_lower or "completed" in question_lower:
            match = re.search(r'completed in (\d{4})', text)
            if match:
                return match.group(1)
        elif "tall" in question_lower:
            match = re.search(r'stands (\d+) meters', text)
            if match:
                return match.group(1) + " meters"
        return "Answer not found in context"
    
    print(f"\nContext Document:\n{context.strip()}")
    print("\nQuestion Answering Results:")
    for q in questions:
        answer = simple_answer(q, context)
        print(f"  Q: {q}")
        print(f"  A: {answer}\n")
    
    print("✅ Question Answering: Computers find specific answers in documents")
    print("   - used in search engines, virtual assistants like Siri/Alexa")

def task4_summarization():
    print("\n" + "=" * 60)
    print("TASK 4: TEXT SUMMARIZATION")
    print("=" * 60)
    
    long_text = """
    Natural Language Processing (NLP) is a field of artificial intelligence that focuses 
    on the interaction between computers and human languages. NLP enables machines to 
    understand, interpret, and generate human language in a valuable way. There are many 
    practical applications of NLP in our daily lives. For example, chatbots use NLP to 
    communicate with customers. Search engines like Google use NLP to understand user 
    queries and provide relevant results. Email providers use NLP for spam detection, 
    automatically filtering unwanted messages. Translation services like Google Translate 
    rely heavily on NLP to convert text between different languages. Sentiment analysis 
    applications use NLP to determine whether customer reviews are positive or negative. 
    Virtual assistants such as Siri, Alexa, and Google Assistant utilize NLP to process 
    voice commands. Text summarization tools can condense long articles into short 
    paragraphs. Named Entity Recognition helps extract important information like names, 
    dates, and locations from documents. The field continues to evolve with deep learning 
    and transformers like BERT and GPT achieving remarkable results. As technology 
    advances, NLP will become even more integrated into our daily digital experiences.
    """
    
    sentences = re.split(r'[.!?]\s+', long_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in re.findall(r'\b\w+\b', long_text) 
             if word.lower() not in stop_words]
    
    from collections import Counter
    word_freq = Counter(words)
    
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        score = 0
        for word in re.findall(r'\b\w+\b', sentence.lower()):
            if word in word_freq:
                score += word_freq[word]
        score = score / max(len(sentence.split()), 1)
        sentence_scores[i] = score
    
    num_sentences = max(2, len(sentences) // 3)
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    sorted_sentences = sorted(top_sentences, key=lambda x: x[0])
    
    original_words = len(long_text.split())
    original_sentences = len(sentences)
    summary = ' '.join([sentences[idx] for idx, _ in sorted_sentences])
    summary_words = len(summary.split())
    
    print(f"\nOriginal Text: {original_words} words, {original_sentences} sentences")
    print(f"Original (first 300 chars):\n{long_text[:300]}...\n")
    
    print(f"Extractive Summary: {summary_words} words, {len(sorted_sentences)} sentences")
    print(f"{summary}\n")
    
    print("✅ Text Summarization: Condenses long documents while preserving key information")
    print("   - used in news aggregation, research paper abstracts, document management")

def task5_preprocessing():
    print("\n" + "=" * 60)
    print("TASK 5: TEXT PREPROCESSING (Tokenization, Stopwords, Stemming, Lemmatization)")
    print("=" * 60)
    
    sample_text = "The quick brown foxes are jumping over the lazy dogs in the beautiful forest."
    
    tokens = nltk.word_tokenize(sample_text.lower())
    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    print(f"\nOriginal Text: {sample_text}")
    print(f"\nStep 1 - Tokenization ({len(tokens)} tokens):")
    print(f"  {tokens}")
    
    print(f"\nStep 2 - Stopword Removal ({len(filtered_tokens)} tokens kept):")
    print(f"  Removed {len(tokens) - len(filtered_tokens)} stopwords")
    print(f"  Kept: {filtered_tokens}")
    
    print(f"\nStep 3 - Stemming:")
    print(f"  {' -> '.join(filtered_tokens[:3])} -> {stemmed_tokens[:3]}")
    print(f"  Stemmed result: {stemmed_tokens}")
    
    print(f"\nStep 4 - Lemmatization:")
    print(f"  {' -> '.join(filtered_tokens[:3])} -> {lemmatized_tokens[:3]}")
    print(f"  Lemmatized result: {lemmatized_tokens}")
    
    print("\n✅ Text Preprocessing: Essential first step for any NLP application")
    print("   - Tokenization: breaks text into words")
    print("   - Stopwords removal: filters out common words (the, is, at)")
    print("   - Stemming/Lemmatization: reduces words to base form")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LAB 9: NATURAL LANGUAGE PROCESSING (NLP) TASKS")
    print("Based on GeeksforGeeks NLP Tasks Tutorial")
    print("=" * 60)
    
    task1_text_classification()
    task2_token_classification()
    task3_question_answering()
    task4_summarization()
    task5_preprocessing()
    
    print("\n" + "=" * 60)
    print("LAB 9 COMPLETED ✓")
    print("All NLP tasks demonstrated successfully!")
    print("=" * 60 + "\n")