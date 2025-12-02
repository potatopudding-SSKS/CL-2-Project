"""
Create a sample SCWS-style dataset from SimLex-999 by adding synthetic contexts.
This is a workaround since the original SCWS dataset is not available.
"""

import pandas as pd
import random

def create_sample_contexts():
    """Create sample contexts for word pairs from SimLex-999."""
    
    # Load SimLex-999 (comma-separated, not tab-separated)
    simlex = pd.read_csv('/home/yash-more/Downloads/21/cl2/project/CL-2-Project/datasets/SimLex-999.csv')
    
    # Simple context templates
    noun_contexts = [
        "The {word} was very interesting to observe.",
        "I saw a {word} yesterday at the park.",
        "The {word} is commonly found in nature.",
        "People often talk about the {word}.",
        "The {word} appeared suddenly."
    ]
    
    verb_contexts = [
        "They decided to {word} early in the morning.",
        "It's important to {word} carefully.",
        "We should {word} before it's too late.",
        "He tried to {word} but failed.",
        "She will {word} tomorrow."
    ]
    
    adj_contexts = [
        "The situation was very {word}.",
        "It seemed quite {word} to me.",
        "Everything felt {word} today.",
        "The view was {word}.",
        "The experience was {word}."
    ]
    
    # Create sample SCWS dataset
    scws_data = []
    
    for idx, row in simlex.iterrows():
        word1 = row['word1']
        word2 = row['word2']
        pos = row['POS']
        sim_score = row['SimLex999']
        
        # Select appropriate context templates
        if pos == 'N':
            contexts = noun_contexts
        elif pos == 'V':
            contexts = verb_contexts
        else:
            contexts = adj_contexts
        
        # Create contexts for both words
        context1 = random.choice(contexts).format(word=word1)
        context2 = random.choice(contexts).format(word=word2)
        
        scws_data.append({
            'word1': word1,
            'word2': word2,
            'pos': pos,
            'context1': context1,
            'context2': context2,
            'similarity': sim_score / 10.0  # Normalize to 0-1 scale like SCWS
        })
    
    # Save to file
    scws_df = pd.DataFrame(scws_data)
    output_path = '/home/yash-more/Downloads/21/cl2/project/CL-2-Project/datasets/scws_sample.txt'
    scws_df.to_csv(output_path, sep='\t', index=False)
    
    print(f"Created sample SCWS dataset with {len(scws_data)} pairs")
    print(f"Saved to: {output_path}")
    print(f"\nFirst few rows:")
    print(scws_df.head())

if __name__ == "__main__":
    create_sample_contexts()
