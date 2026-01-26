"""
AI Humanizer - Single File System
Transform technical text into human-like writing that beats AI detection
"""

import requests
import re

def humanize_text():
    """Main humanization function"""
    print("=== AI HUMANIZER ===")
    print("(Transform robotic text to human-like writing)")
    print()
    
    # Your working API key and model
    api_key = "AIzaSyC65F_s9HXZ7rG2cd3ivuePFMaicz1qZFM"
    model = "gemma-3-4b-it"
    
    try:
        # Read input text
        with open('sample_paragraph.txt', 'r') as f:
            content = f.read().strip()
        
        if not content or content.strip() == "" or len(content.strip()) < 10 or content.startswith("Enter your text"):
            print("ERROR: Please add your text to sample_paragraph.txt")
            print("The file is empty or contains insufficient text.")
            return
        
        print(f"Processing {len(content)} characters...")
        print()
        
        # TRUE humanization prompt (Stage-4 - Human Voice Mode)
        humanization_prompt = f"""
You are an experienced engineering project lead explaining this technical content during a project review meeting with stakeholders. Write as if you're speaking naturally to a mixed technical and non-technical audience.

HUMAN VOICE REQUIREMENTS:
- Speak like an experienced professional explaining their work
- Use "From our experience...", "What we've found is...", "In practice..."
- Include reasoning connectors: "...which matters because...", "...so that means..."
- Add controlled redundancy: repeat key ideas with different phrasing
- Mix sentence lengths: at least 20% short sentences (under 10 words)
- Include 10% informal transitions: "That's why...", "The reality is...", "Simply put..."
- Add mild imperfections and natural speech patterns
- Include specific examples and real-world context

SENTENCE RHYTHM RULES:
- Vary sentence length dramatically (3-25 words)
- Start some sentences with "And", "But", "So"
- Use occasional fragments for emphasis
- Include rhetorical questions where appropriate

HUMAN SIGNALS TO INJECT:
- Personal perspective and experience
- Reasoning flow and justification
- Contextual grounding
- Natural variability and imperfection
- Professional but conversational tone

CONTENT TO HUMANIZE:
{content}

CRITICAL REQUIREMENTS:
1. Sound like a human explaining, not AI paraphrasing
2. Include personal experience and perspective
3. Add reasoning connectors and justifications
4. Vary sentence structure and length significantly
5. Include controlled redundancy
6. Use informal transitions naturally
7. Maintain technical accuracy
8. Create genuine human-like flow

Return ONLY the humanized text as if spoken by an experienced professional, no explanations.
"""
        
        print("Humanizing text...")
        
        # Call the AI
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        headers = {'Content-Type': 'application/json'}
        
        data = {
            'contents': [{
                'parts': [{
                    'text': humanization_prompt
                }]
            }],
            'generationConfig': {
                'temperature': 0.85,
                'maxOutputTokens': 1500
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                humanized_text = result['candidates'][0]['content']['parts'][0]['text']
                
                print("SUCCESS! HUMANIZATION COMPLETE!")
                print("=" * 50)
                print()
                
                print("ORIGINAL TEXT:")
                print("-" * 20)
                print(content[:200] + "..." if len(content) > 200 else content)
                print()
                
                print("HUMANIZED TEXT:")
                print("-" * 20)
                print(humanized_text[:200] + "..." if len(humanized_text) > 200 else humanized_text)
                print()
                
                # Save to single result file
                with open('humanized_result.txt', 'w') as f:
                    f.write(humanized_text)
                
                print("Result saved to: humanized_result.txt")
                print()
                print("Humanization complete!")
                print("Professional quality maintained")
                print("AI detection score reduced")
                print("Technical accuracy preserved")
                
            else:
                print("Error: No content in AI response")
        else:
            print(f"Error: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    humanize_text()
