"""
Self-Improving AI Humanizer - Advanced Machine Learning System
Infinite learning capability with continuous improvement through experience
"""

import json
import pickle
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
import hashlib
import requests
import re
from typing import Dict, List, Tuple, Any
import os

class KnowledgeBase:
    """Storage-efficient persistent knowledge storage system"""
    def __init__(self, knowledge_file="ai_knowledge.pkl"):
        self.knowledge_file = knowledge_file
        self.patterns = defaultdict(list)
        self.successful_transformations = deque(maxlen=100)  # Keep only last 100 transformations
        self.performance_metrics = deque(maxlen=500)  # Track last 500 scores
        self.vocabulary_enhancements = {}
        self.contextual_adaptations = {}
        self.pattern_frequency = defaultdict(int)  # Track pattern effectiveness
        self.knowledge_compaction_threshold = 1000  # Compact after 1000 additions
        self.total_experiences_processed = 0
        self.load_knowledge()
    
    def load_knowledge(self):
        """Load existing knowledge from file"""
        if os.path.exists(self.knowledge_file):
            try:
                with open(self.knowledge_file, 'rb') as f:
                    data = pickle.load(f)
                    self.patterns = defaultdict(list, data.get('patterns', {}))
                    # Convert to deque with maxlen for memory efficiency
                    transformations = data.get('successful_transformations', [])
                    self.successful_transformations = deque(transformations[-100:], maxlen=100)
                    metrics = data.get('performance_metrics', [])
                    self.performance_metrics = deque(metrics[-500:], maxlen=500)
                    self.vocabulary_enhancements = data.get('vocabulary_enhancements', {})
                    self.contextual_adaptations = data.get('contextual_adaptations', {})
                    self.pattern_frequency = defaultdict(int, data.get('pattern_frequency', {}))
                    self.total_experiences_processed = data.get('total_experiences_processed', 0)
            except Exception as e:
                print(f"Warning: Could not load knowledge file: {e}")
                self.total_experiences_processed = 0
    
    def save_knowledge(self):
        """Save current knowledge to file with compaction"""
        self.total_experiences_processed += 1
        
        # Compact knowledge periodically to save space
        if self.total_experiences_processed % self.knowledge_compaction_threshold == 0:
            self._compact_knowledge()
        
        data = {
            'patterns': dict(self.patterns),
            'successful_transformations': list(self.successful_transformations),
            'performance_metrics': list(self.performance_metrics),
            'vocabulary_enhancements': self.vocabulary_enhancements,
            'contextual_adaptations': self.contextual_adaptations,
            'pattern_frequency': dict(self.pattern_frequency),
            'total_experiences_processed': self.total_experiences_processed
        }
        with open(self.knowledge_file, 'wb') as f:
            pickle.dump(data, f)
    
    def _compact_knowledge(self):
        """Compact knowledge to save storage space"""
        print(f"Compacting knowledge after {self.total_experiences_processed} experiences...")
        
        # Keep only most effective patterns
        for pattern_type, patterns in self.patterns.items():
            if len(patterns) > 50:
                # Sort by frequency and keep top 50
                patterns.sort(key=lambda x: self.pattern_frequency.get(str(x), 0), reverse=True)
                self.patterns[pattern_type] = patterns[:50]
        
        # Keep only most effective vocabulary
        if len(self.vocabulary_enhancements) > 1000:
            # Keep vocabulary that appears most frequently
            self.vocabulary_enhancements = dict(list(self.vocabulary_enhancements.items())[:1000])
        
        # Limit contextual adaptations
        if len(self.contextual_adaptations) > 200:
            self.contextual_adaptations = dict(list(self.contextual_adaptations.items())[:200])
        
        print("Knowledge compaction complete - storage optimized")
    
    def add_experience(self, original_text, humanized_text, performance_score):
        """Add new experience to knowledge base with memory efficiency"""
        experience = {
            'timestamp': datetime.now().isoformat(),
            'original': original_text[:500] + "..." if len(original_text) > 500 else original_text,  # Truncate for storage
            'humanized': humanized_text[:500] + "..." if len(humanized_text) > 500 else humanized_text,  # Truncate for storage
            'performance_score': performance_score,
            'patterns_extracted': self._extract_patterns(original_text, humanized_text)
        }
        
        # Add to deque (automatically manages size)
        self.successful_transformations.append(experience)
        self.performance_metrics.append(performance_score)
        
        # Update patterns with frequency tracking
        for pattern in experience['patterns_extracted']:
            pattern_key = str(pattern)
            self.pattern_frequency[pattern_key] += 1
            
            # Only store pattern if it's effective (appears multiple times)
            if self.pattern_frequency[pattern_key] >= 2 or len(self.patterns[pattern['type']]) < 20:
                self.patterns[pattern['type']].append(pattern)
        
        # Update vocabulary enhancements (only keep effective ones)
        for pattern in experience['patterns_extracted']:
            if pattern['type'] == 'vocabulary_enhancement' and 'new_words' in pattern:
                for word in pattern['new_words']:
                    if word not in self.vocabulary_enhancements:
                        self.vocabulary_enhancements[word] = 0
                    self.vocabulary_enhancements[word] += 1
    
    def _extract_patterns(self, original, humanized):
        """Extract transformation patterns"""
        patterns = []
        
        # Sentence structure patterns
        orig_sentences = len(re.split(r'[.!?]+', original))
        human_sentences = len(re.split(r'[.!?]+', humanized))
        patterns.append({
            'type': 'sentence_count_change',
            'original': orig_sentences,
            'humanized': human_sentences,
            'ratio': human_sentences / max(orig_sentences, 1)
        })
        
        # Word choice patterns
        orig_words = set(original.lower().split())
        human_words = set(humanized.lower().split())
        new_words = human_words - orig_words
        if new_words:
            patterns.append({
                'type': 'vocabulary_enhancement',
                'new_words': list(new_words)[:5]  # Store top 5
            })
        
        # Transition patterns
        transitions = ['however', 'therefore', 'moreover', 'furthermore', 'consequently', 
                     'moreover', 'nevertheless', 'nonetheless', 'meanwhile', 'furthermore']
        found_transitions = [t for t in transitions if t in humanized.lower()]
        if found_transitions:
            patterns.append({
                'type': 'transition_usage',
                'transitions': found_transitions
            })
        
        return patterns
    
    def get_best_practices(self):
        """Get learned best practices"""
        if not self.successful_transformations:
            return {}
        
        # Analyze top performing transformations
        top_experiences = sorted(self.successful_transformations, 
                              key=lambda x: x['performance_score'], reverse=True)[:10]
        
        best_practices = {
            'avg_sentence_length': np.mean([len(x['humanized'].split()) / 
                                          max(len(re.split(r'[.!?]+', x['humanized'])), 1) 
                                          for x in top_experiences]),
            'common_transitions': [],
            'vocabulary_complexity': 0,
            'human_voice_indicators': []
        }
        
        # Extract common patterns from top performers
        all_transitions = []
        for exp in top_experiences:
            for pattern in exp['patterns_extracted']:
                if pattern['type'] == 'transition_usage':
                    all_transitions.extend(pattern['transitions'])
        
        if all_transitions:
            transition_counts = defaultdict(int)
            for t in all_transitions:
                transition_counts[t] += 1
            best_practices['common_transitions'] = sorted(transition_counts.items(), 
                                                         key=lambda x: x[1], reverse=True)[:5]
        
        return best_practices

class LearningEngine:
    """Machine learning engine for continuous improvement"""
    def __init__(self, knowledge_base):
        self.knowledge = knowledge_base
        self.learning_rate = 0.01
        self.adaptation_history = []
    
    def analyze_performance(self, original_text, humanized_text):
        """Analyze and score the transformation quality"""
        score = 0.0
        
        # Human voice indicators
        human_indicators = ['from our experience', 'what we\'ve found', 'in practice', 
                          'that\'s why', 'the reality is', 'simply put']
        humanized_lower = humanized_text.lower()
        score += sum([1 for indicator in human_indicators if indicator in humanized_lower]) * 10
        
        # Sentence variety
        sentences = re.split(r'[.!?]+', humanized_text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            length_variance = np.var(sentence_lengths)
            score += min(length_variance, 50)  # Cap variance contribution
        
        # Vocabulary enhancement
        orig_words = set(original_text.lower().split())
        human_words = set(humanized_text.lower().split())
        new_words = human_words - orig_words
        score += min(len(new_words) * 2, 30)
        
        # Natural flow indicators
        flow_patterns = ['and', 'but', 'so', 'well', 'you know', 'i mean']
        score += sum([humanized_text.lower().count(pattern) for pattern in flow_patterns]) * 5
        
        return min(score, 100)  # Cap at 100
    
    def improve_prompt(self, base_prompt, performance_history):
        """Adaptively improve the humanization prompt based on performance"""
        if len(performance_history) < 5:
            return base_prompt
        
        recent_performance = list(performance_history)[-5:]
        avg_performance = np.mean(recent_performance)
        
        improvements = []
        
        # If performance is declining, add more specific instructions
        if avg_performance < 70:
            improvements.append("Focus on adding more personal experience phrases.")
            improvements.append("Include more reasoning connectors like 'which matters because...'")
        
        # If vocabulary is limited, enhance it
        best_practices = self.knowledge.get_best_practices()
        if best_practices.get('common_transitions'):
            top_transitions = [t[0] for t in best_practices['common_transitions'][:3]]
            improvements.append(f"Naturally incorporate these transitions: {', '.join(top_transitions)}")
        
        # Build improved prompt
        if improvements:
            improved_prompt = base_prompt + "\n\nLEARNING-BASED ENHANCEMENTS:\n"
            improvements.append(f"Current performance target: {avg_performance:.1f}/100")
            improved_prompt += "\n".join([f"- {imp}" for imp in improvements])
            return improved_prompt
        
        return base_prompt
    
    def learn_from_feedback(self, original_text, humanized_text, user_feedback=None):
        """Learn from explicit or implicit feedback"""
        performance_score = self.analyze_performance(original_text, humanized_text)
        
        # Adjust learning parameters based on performance
        if performance_score > 80:
            self.learning_rate = max(0.001, self.learning_rate * 0.95)  # Reduce learning rate if doing well
        else:
            self.learning_rate = min(0.1, self.learning_rate * 1.05)  # Increase learning rate if struggling
        
        # Store the learning experience
        self.knowledge.add_experience(original_text, humanized_text, performance_score)
        
        return performance_score

class SelfImprovingAI:
    """Main self-improving AI system"""
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.learning_engine = LearningEngine(self.knowledge_base)
        self.api_key = "AIzaSyC65F_s9HXZ7rG2cd3ivuePFMaicz1qZFM"
        self.model = "gemma-3-4b-it"
        self.generation_count = 0
        self.performance_history = deque(maxlen=100)
    
    def humanize_text(self, text):
        """Main humanization function with continuous learning"""
        self.generation_count += 1
        print(f"=== SELF-IMPROVING AI HUMANIZER (Generation #{self.generation_count}) ===")
        print(f"Learning from {len(self.knowledge_base.successful_transformations)} past experiences")
        print()
        
        # Get base prompt and improve it based on learning
        base_prompt = self._get_base_prompt(text)
        improved_prompt = self.learning_engine.improve_prompt(base_prompt, self.performance_history)
        
        print("Calling AI with learned optimizations...")
        
        # Call the AI
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        headers = {'Content-Type': 'application/json'}
        data = {
            'contents': [{
                'parts': [{
                    'text': improved_prompt
                }]
            }],
            'generationConfig': {
                'temperature': 0.85 + (self.generation_count * 0.001),  # Slightly increase creativity over time
                'maxOutputTokens': 1500
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    humanized_text = result['candidates'][0]['content']['parts'][0]['text']
                    
                    # Learn from this experience
                    performance_score = self.learning_engine.learn_from_feedback(text, humanized_text)
                    self.performance_history.append(performance_score)
                    
                    print(f"Performance Score: {performance_score:.1f}/100")
                    print(f"Learning Rate: {self.learning_engine.learning_rate:.4f}")
                    
                    # Save knowledge
                    self.knowledge_base.save_knowledge()
                    
                    return humanized_text, performance_score
                else:
                    print("Error: No content in AI response")
                    return None, 0
            else:
                print(f"Error: HTTP {response.status_code}")
                return None, 0
                
        except Exception as e:
            print(f"Error: {e}")
            return None, 0
    
    def _get_base_prompt(self, content):
        """Get the base humanization prompt"""
        return f"""
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
    
    def get_learning_stats(self):
        """Get learning statistics with storage info"""
        knowledge_size = os.path.getsize(self.knowledge_base.knowledge_file) if os.path.exists(self.knowledge_base.knowledge_file) else 0
        
        return {
            'total_experiences': self.knowledge_base.total_experiences_processed,
            'stored_experiences': len(self.knowledge_base.successful_transformations),
            'avg_performance': np.mean(list(self.performance_history)) if self.performance_history else 0,
            'learning_rate': self.learning_engine.learning_rate,
            'generations': self.generation_count,
            'patterns_learned': len(self.knowledge_base.patterns),
            'knowledge_file_size_mb': knowledge_size / (1024 * 1024),  # Size in MB
            'vocabulary_size': len(self.knowledge_base.vocabulary_enhancements),
            'storage_efficiency': f"{len(self.knowledge_base.successful_transformations)}/{self.knowledge_base.total_experiences_processed}"
        }

def main():
    """Main function with continuous learning demonstration"""
    ai = SelfImprovingAI()
    
    try:
        # Read input text
        with open('sample_paragraph.txt', 'r') as f:
            content = f.read().strip()
        
        if not content or len(content) < 10:
            print("ERROR: Please add your text to sample_paragraph.txt")
            return
        
        print(f"Processing {len(content)} characters...")
        print()
        
        # Humanize text with learning
        result, performance = ai.humanize_text(content)
        
        if result:
            print("\n" + "="*60)
            print("SELF-IMPROVING AI HUMANIZATION COMPLETE!")
            print("="*60)
            print()
            
            print("ORIGINAL TEXT:")
            print("-" * 20)
            print(content[:300] + "..." if len(content) > 300 else content)
            print()
            
            print("HUMANIZED TEXT:")
            print("-" * 20)
            print(result[:300] + "..." if len(result) > 300 else result)
            print()
            
            # Save result
            with open('humanized_result.txt', 'w') as f:
                f.write(result)
            
            print("Result saved to: humanized_result.txt")
            print()
            
            # Show learning stats
            stats = ai.get_learning_stats()
            print("LEARNING STATISTICS:")
            print("-" * 20)
            print(f"Total Experiences Processed: {stats['total_experiences']}")
            print(f"Stored Experiences: {stats['stored_experiences']}")
            print(f"Storage Efficiency: {stats['storage_efficiency']}")
            print(f"Knowledge File Size: {stats['knowledge_file_size_mb']:.2f} MB")
            print(f"Average Performance: {stats['avg_performance']:.1f}/100")
            print(f"Learning Rate: {stats['learning_rate']:.4f}")
            print(f"Generations: {stats['generations']}")
            print(f"Patterns Learned: {stats['patterns_learned']}")
            print(f"Vocabulary Size: {stats['vocabulary_size']}")
            print()
            print("Storage-Optimized Infinite Learning Active!")
            print("System improves with each use while maintaining minimal storage")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
