#!/usr/bin/env python3
import argparse
import os
import re
from collections import Counter
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Download required NLTK data
try:
    nltk.data.find('punkt')
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ChatSummarizer:
    def __init__(self, stop_words_file='stopwords.txt'):
        """Initialize summarizer with custom stop words."""
        self.user_messages = []
        self.ai_messages = []
        self.stop_words = set(stopwords.words('english'))
        
        # Load custom stop words if provided
        if stop_words_file and os.path.exists(stop_words_file):
            with open(stop_words_file, 'r') as f:
                self.stop_words.update(word.strip().lower() for word in f)

    def parse_chat_log(self, file_path):
        """Parse a chat log file into User and AI messages."""
        self.user_messages = []
        self.ai_messages = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('User:'):
                        self.user_messages.append(line[5:].strip())
                    elif line.startswith('AI:'):
                        self.ai_messages.append(line[3:].strip())
            logger.info(f"Parsed {file_path}: {len(self.user_messages)} user messages, {len(self.ai_messages)} AI messages")
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            raise

    def get_message_stats(self):
        """Return statistics about the chat log."""
        total_messages = len(self.user_messages) + len(self.ai_messages)
        return {
            'total_messages': total_messages,
            'user_messages': len(self.user_messages),
            'ai_messages': len(self.ai_messages)
        }

    def extract_keywords(self, top_n=5, use_tfidf=True):
        """Extract top keywords using TF-IDF or simple frequency count."""
        all_messages = self.user_messages + self.ai_messages
        
        if not all_messages:
            return []

        if use_tfidf:
            # Use TF-IDF for keyword extraction
            vectorizer = TfidfVectorizer(
                stop_words=list(self.stop_words),
                token_pattern=r'(?u)\b\w\w+\b'
            )
            try:
                tfidf_matrix = vectorizer.fit_transform(all_messages)
                feature_names = vectorizer.get_feature_names_out()
                # Get average TF-IDF score for each word
                scores = tfidf_matrix.mean(axis=0).A1
                keywords = sorted(
                    [(word, score) for word, score in zip(feature_names, scores)],
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n]
                return [word for word, _ in keywords]
            except ValueError:
                logger.warning("TF-IDF failed, falling back to frequency count")
        
        # Fallback to simple frequency count
        words = []
        for message in all_messages:
            tokens = word_tokenize(message.lower())
            words.extend(
                word for word in tokens
                if word.isalnum() and word not in self.stop_words
            )
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(top_n)]

    def generate_summary(self, file_path):
        """Generate and return a summary of the chat log."""
        stats = self.get_message_stats()
        keywords = self.extract_keywords()
        
        # Simple topic inference based on keywords
        topic = "general conversation"
        if keywords:
            topic = f"discussion about {', '.join(keywords[:2])}"

        summary = (
            f"Summary for {os.path.basename(file_path)}:\n"
            f"- Total exchanges: {stats['total_messages']}\n"
            f"- User messages: {stats['user_messages']}\n"
            f"- AI messages: {stats['ai_messages']}\n"
            f"- Main topic: {topic}\n"
            f"- Top keywords: {', '.join(keywords) if keywords else 'None'}\n"
        )
        return summary

def process_files(input_path, stop_words_file='stopwords.txt'):
    """Process single file or directory of chat logs and save summaries."""
    summarizer = ChatSummarizer(stop_words_file)
    summaries = []
    
    # Create summary directory if it doesn't exist
    summary_dir = Path('summary')
    summary_dir.mkdir(exist_ok=True)

    input_path = Path(input_path)
    
    if input_path.is_file():
        summarizer.parse_chat_log(input_path)
        summary = summarizer.generate_summary(input_path)
        summaries.append(summary)
        # Save summary to file
        output_file = summary_dir / f"{input_path.name}"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"Saved summary to {output_file}")
    elif input_path.is_dir():
        for file_path in input_path.glob('*.txt'):
            try:
                summarizer.parse_chat_log(file_path)
                summary = summarizer.generate_summary(file_path)
                summaries.append(summary)
                # Save summary to file
                output_file = summary_dir / f"{file_path.name}"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                logger.info(f"Saved summary to {output_file}")
            except Exception as e:
                logger.error(f"Skipping {file_path}: {str(e)}")
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    return summaries

def main():
    """Main function to handle command-line arguments and run summarizer."""
    parser = argparse.ArgumentParser(description="AI Chat Log Summarizer")
    parser.add_argument(
        'input_path',
        help="Path to a single chat log file or directory of chat logs"
    )
    parser.add_argument(
        '--stop-words',
        default='stopwords.txt',
        help="Path to custom stop words file"
    )
    args = parser.parse_args()

    try:
        summaries = process_files(args.input_path, args.stop_words)
        for summary in summaries:
            print(summary)
            print("-" * 50)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()