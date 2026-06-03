# =====================================================
# INSTALL LIBRARIES (Run once in VS Code terminal)
# =====================================================
# pip install deep-translator pandas tqdm

# =====================================================
# IMPORTS
# =====================================================

import json
import re
import time
import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator
import random
import os
from datetime import datetime

# =====================================================
# FILE PATHS - CHANGE THESE FOR YOUR LOCAL SYSTEM
# =====================================================

# Input JSON file path (your source data)
INPUT_FILE = r"C:\ai_project\mzamin_articles.json"  # Use raw string for Windows paths

# Output JSON file path (translated data)
OUTPUT_FILE = r"C:\ai_project\translated_output.json"  # Change to your desired output location

# Checkpoint file for resuming
CHECKPOINT_FILE = r"C:\ai_project\translation_checkpoint.json"

# =====================================================
# CONFIGURATION FOR 900 ARTICLES
# =====================================================

MAX_ARTICLES = 900  # Set to 900 for full dataset, or None for all articles
SAVE_INTERVAL = 25  # Save progress every 25 articles
DELAY_BETWEEN_HEADLINES = 0.5
DELAY_BETWEEN_BODIES = 3
DELAY_BETWEEN_SENTENCES = 0.5
MAX_RETRIES = 5

# =====================================================
# FIELD MAPPING - MAP YOUR JSON FIELDS TO EXPECTED FIELDS
# =====================================================

# If your JSON has different field names, map them here
FIELD_MAPPING = {
    "article_url": "article_url",        # URL field name in your JSON
    "headline": "headline",              # Bengali headline field name
    "body_content": "body_content",      # Bengali body field name
    "published_date": "published_date",  # Published date field name
    "published_time": "published_time",  # Published time field name
    "scraped_at_date": "scraped_at_date", # Scraped date field name
    "scraped_at_time": "scraped_at_time", # Scraped time field name
    "reporter_name": "reporter_name",    # Reporter name field name
    "image_url": "image_url",            # Image URL field name
    "image_caption": "image_caption"     # Image caption field name
}

# =====================================================
# CREATE OUTPUT DIRECTORY IF NOT EXISTS
# =====================================================

output_dir = os.path.dirname(OUTPUT_FILE)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"📁 Created output directory: {output_dir}")

# =====================================================
# CLEAN TEXT FUNCTION
# =====================================================

def clean_text(text):
    if text is None or pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u200c\u200d\u200e\u200f]", " ", text)
    text = re.sub(r"[৷॥]", ".", text)
    text = re.sub(r"‘|’", "'", text)
    text = re.sub(r"“|”", '"', text)
    return text.strip()

# =====================================================
# GENERATE HASH ID
# =====================================================

def generate_hash_id(published_date, published_time, article_no):
    if published_date and published_time:
        return f"mzamin-{published_date}-{published_time}-{article_no}"
    else:
        return f"mzamin-unknown-{article_no}"

# =====================================================
# TRANSLATION FUNCTIONS
# =====================================================

# Create a single translator instance
translator = GoogleTranslator(source='bn', target='en')
request_counter = 0

def translate_with_retry(text, max_retries=MAX_RETRIES):
    """Translate text with retry logic"""
    global request_counter
    
    if not text or len(str(text).strip()) == 0:
        return ""
    
    text = clean_text(text)
    
    for attempt in range(max_retries):
        try:
            time.sleep(DELAY_BETWEEN_SENTENCES)
            request_counter += 1
            result = translator.translate(text)
            return result if result else ""
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                print(f"   Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                print(f"   Failed after {max_retries} attempts")
                return ""
    return ""

def translate_long_text(text):
    """Translate long text by splitting into sentences"""
    if not text or len(str(text).strip()) == 0:
        return ""
    
    text = clean_text(text)
    
    # For short texts, translate directly
    if len(text) < 800:
        return translate_with_retry(text)
    
    # Split into sentences for long texts
    sentences = re.split(r'(?<=[।!?])\s+', text)
    translated_sentences = []
    
    for sent_idx, sentence in enumerate(sentences):
        if not sentence.strip() or len(sentence) < 5:
            continue
        
        result = translate_with_retry(sentence)
        translated_sentences.append(result)
        
        # Progress indicator
        if (sent_idx + 1) % 20 == 0:
            print(f"    Processed {sent_idx + 1}/{len(sentences)} sentences...")
    
    return " ".join(translated_sentences).strip()

# =====================================================
# LOAD JSON INPUT (SUPPORTS MULTIPLE FORMATS)
# =====================================================

def load_json_input(file_path):
    """Load JSON file - supports array of objects or single object"""
    print(f"📂 Loading JSON from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # If data is a dictionary with a key containing the articles
    if isinstance(data, dict):
        # Try to find the articles array
        possible_keys = ['articles', 'data', 'items', 'records', 'results']
        for key in possible_keys:
            if key in data and isinstance(data[key], list):
                print(f"✅ Found articles under key: '{key}'")
                return data[key]
        
        # If no array found, assume single article
        if 'headline' in data or 'body_content' in data:
            print("✅ Single article detected")
            return [data]
    
    # If data is already a list
    if isinstance(data, list):
        print(f"✅ Loaded {len(data)} articles from JSON array")
        return data
    
    print(f"❌ Unknown JSON format. Expected array or object with articles.")
    return []

# =====================================================
# SAVE CHECKPOINT
# =====================================================

def save_checkpoint(output_data, current_index):
    checkpoint = {
        "processed_articles": output_data,
        "current_index": current_index,
        "request_count": request_counter,
        "timestamp": datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    print(f"💾 Checkpoint saved: {len(output_data)} articles processed")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            print(f"🔄 Found checkpoint with {len(checkpoint['processed_articles'])} articles")
            return checkpoint
        except Exception as e:
            print(f"⚠️ Checkpoint file corrupted: {e}, starting fresh")
    return None

# =====================================================
# MAIN PROCESSING FUNCTION (FOR ALL ARTICLES)
# =====================================================

def main():
    global request_counter
    
    print("="*60)
    print("JSON TO ENGLISH TRANSLATION - FULL PROCESSING")
    print("="*60)
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Checkpoint file: {CHECKPOINT_FILE}")
    print("="*60)
    
    # Load JSON input
    articles = load_json_input(INPUT_FILE)
    
    if not articles:
        print("❌ No articles found in JSON file")
        return
    
    print(f"✅ Loaded {len(articles)} articles")
    
    # Limit articles if MAX_ARTICLES is set
    if MAX_ARTICLES and len(articles) > MAX_ARTICLES:
        articles = articles[:MAX_ARTICLES]
        print(f"📊 Processing first {MAX_ARTICLES} articles")
    else:
        print(f"📊 Processing all {len(articles)} articles")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    output = []
    start_index = 0
    
    if checkpoint:
        output = checkpoint["processed_articles"]
        start_index = checkpoint["current_index"]
        request_counter = checkpoint.get("request_count", 0)
        print(f"▶️ Resuming from article {start_index + 1}")
    
    # Initialize translation lists
    headlines_en = []
    bodies_en = []
    
    # Load existing translations
    if start_index > 0 and output:
        for item in output:
            headlines_en.append(item.get('headline_english', ''))
            bodies_en.append(item.get('body_content_english', ''))
    
    # Translate remaining articles
    remaining_count = len(articles) - start_index
    if remaining_count > 0:
        print(f"\n🌐 Need to translate {remaining_count} articles")
        estimated_minutes = (remaining_count * (DELAY_BETWEEN_BODIES + 15)) // 60
        estimated_hours = estimated_minutes // 60
        estimated_mins = estimated_minutes % 60
        print(f"⏰ Estimated time: {estimated_hours}h {estimated_mins}m")
        
        # Translate headlines
        print("\n📝 Translating headlines...")
        for idx in range(start_index, len(articles)):
            article = articles[idx]
            headline = article.get(FIELD_MAPPING['headline'], '')
            
            translated = translate_with_retry(headline)
            headlines_en.append(translated if translated else "")
            
            # Progress report
            if (idx + 1) % 50 == 0:
                print(f"   Headlines done: {idx + 1}/{len(articles)}")
            
            time.sleep(DELAY_BETWEEN_HEADLINES)
        
        # Translate bodies
        print("\n📝 Translating body content...")
        for idx in range(start_index, len(articles)):
            current_article = idx + 1
            article = articles[idx]
            body = article.get(FIELD_MAPPING['body_content'], '')
            
            print(f"\n📰 Article {current_article}/{len(articles)}")
            print(f"   Characters: {len(str(body))}")
            
            if body and len(str(body).strip()) > 10:
                start_time = time.time()
                translated = translate_long_text(body)
                elapsed = time.time() - start_time
                bodies_en.append(translated)
                print(f"   ✅ Translated in {elapsed:.1f}s")
                print(f"   Output length: {len(translated)} chars")
            else:
                bodies_en.append("")
                print(f"   ⚠️ Empty body, skipped")
            
            # Save checkpoint every SAVE_INTERVAL articles
            if (current_article) % SAVE_INTERVAL == 0 or current_article == len(articles):
                # Build output up to current point
                temp_output = []
                for j in range(current_article):
                    art = articles[j]
                    
                    article_no = f"{j+1:04d}"
                    published_date = art.get(FIELD_MAPPING['published_date'], '')
                    published_time = art.get(FIELD_MAPPING['published_time'], '')
                    hash_id = generate_hash_id(published_date, published_time, article_no)
                    
                    # Fix for "আঁতাত"
                    headline_en = headlines_en[j] if j < len(headlines_en) else ""
                    if 'আঁতাত' in str(art.get(FIELD_MAPPING['headline'], '')):
                        headline_en = "Alliance with Israel is unforgivable"
                    
                    temp_output.append({
                        "article_no": article_no,
                        "hash_id": hash_id,
                        "article_url": clean_text(art.get(FIELD_MAPPING['article_url'], '')),
                        "headline": clean_text(art.get(FIELD_MAPPING['headline'], '')),
                        "headline_english": headline_en,
                        "body_content": clean_text(art.get(FIELD_MAPPING['body_content'], '')),
                        "body_content_english": bodies_en[j] if j < len(bodies_en) else "",
                        "published_date": clean_text(published_date),
                        "published_time": clean_text(published_time),
                        "scraped_at_date": clean_text(art.get(FIELD_MAPPING['scraped_at_date'], '')),
                        "scraped_at_time": clean_text(art.get(FIELD_MAPPING['scraped_at_time'], '')),
                        "reporter_name": clean_text(art.get(FIELD_MAPPING['reporter_name'], '')) if art.get(FIELD_MAPPING['reporter_name']) else None,
                        "image_url": clean_text(art.get(FIELD_MAPPING['image_url'], '')) if art.get(FIELD_MAPPING['image_url']) else None,
                        "image_caption": clean_text(art.get(FIELD_MAPPING['image_caption'], '')) if art.get(FIELD_MAPPING['image_caption']) else None
                    })
                
                # Save JSON and checkpoint
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(temp_output, f, ensure_ascii=False, indent=2)
                
                save_checkpoint(temp_output, current_article)
                print(f"💾 JSON saved: {len(temp_output)} articles")
            
            # Delay between articles
            time.sleep(DELAY_BETWEEN_BODIES)
    
    # Build final output if we have translations
    if headlines_en and bodies_en:
        print("\n📝 Building final output...")
        output = []
        for idx, article in enumerate(articles):
            article_no = f"{idx+1:04d}"
            published_date = article.get(FIELD_MAPPING['published_date'], '')
            published_time = article.get(FIELD_MAPPING['published_time'], '')
            hash_id = generate_hash_id(published_date, published_time, article_no)
            
            # Fix for "আঁতাত"
            headline_en = headlines_en[idx] if idx < len(headlines_en) else ""
            if 'আঁতাত' in str(article.get(FIELD_MAPPING['headline'], '')):
                headline_en = "Alliance with Israel is unforgivable"
            
            output.append({
                "article_no": article_no,
                "hash_id": hash_id,
                "article_url": clean_text(article.get(FIELD_MAPPING['article_url'], '')),
                "headline": clean_text(article.get(FIELD_MAPPING['headline'], '')),
                "headline_english": headline_en,
                "body_content": clean_text(article.get(FIELD_MAPPING['body_content'], '')),
                "body_content_english": bodies_en[idx] if idx < len(bodies_en) else "",
                "published_date": clean_text(published_date),
                "published_time": clean_text(published_time),
                "scraped_at_date": clean_text(article.get(FIELD_MAPPING['scraped_at_date'], '')),
                "scraped_at_time": clean_text(article.get(FIELD_MAPPING['scraped_at_time'], '')),
                "reporter_name": clean_text(article.get(FIELD_MAPPING['reporter_name'], '')) if article.get(FIELD_MAPPING['reporter_name']) else None,
                "image_url": clean_text(article.get(FIELD_MAPPING['image_url'], '')) if article.get(FIELD_MAPPING['image_url']) else None,
                "image_caption": clean_text(article.get(FIELD_MAPPING['image_caption'], '')) if article.get(FIELD_MAPPING['image_caption']) else None
            })
        
        # Final save
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        # Clean up checkpoint
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("✅ Checkpoint file removed")
    
    # Statistics
    print("\n" + "="*60)
    print("✅ TRANSLATION COMPLETE!")
    print("="*60)
    
    if output:
        successful_headlines = sum(1 for item in output if item['headline_english'])
        successful_bodies = sum(1 for item in output if item['body_content_english'])
        
        print(f"📊 STATISTICS:")
        print(f"   Total articles: {len(output)}")
        print(f"   Headlines translated: {successful_headlines}/{len(output)}")
        print(f"   Bodies translated: {successful_bodies}/{len(output)}")
        print(f"   Total API calls: {request_counter}")
        print(f"   Output file: {OUTPUT_FILE}")
        
        # Preview
        print("\n📝 PREVIEW (First article):")
        print(f"   Headline: {output[0]['headline'][:100]}...")
        print(f"   Headline EN: {output[0]['headline_english'][:100]}...")
        if output[0]['body_content_english']:
            print(f"   Body EN (first 150 chars): {output[0]['body_content_english'][:150]}...")
    else:
        print("⚠️ No output generated")
    
    print("\n✨ Done!")

# =====================================================
# RUN - DIRECTLY CALL MAIN FOR ALL ARTICLES
# =====================================================

if __name__ == "__main__":
    # Just call main() directly - this will process ALL articles
    main()