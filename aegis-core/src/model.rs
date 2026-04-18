use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A simple N-gram language model for Federated Learning.
/// It maps a context (the previous word) to a frequency map of subsequent words.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LanguageModel {
    /// Outer key: previous word
    /// Inner key: next word
    /// Inner value: frequency count
    pub bigrams: HashMap<String, HashMap<String, u64>>,

    /// Total number of words trained on (used to weight during aggregation)
    pub data_points: u64,
}

impl LanguageModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Train the model on a single string of text.
    pub fn train(&mut self, text: &str) {
        let words: Vec<String> = text
            .split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return;
        }

        self.data_points += words.len() as u64;

        for window in words.windows(2) {
            let context = &window[0];
            let next_word = &window[1];

            let counts = self.bigrams.entry(context.clone()).or_insert_with(HashMap::new);
            *counts.entry(next_word.clone()).or_insert(0) += 1;
        }
    }

    /// Predict the most likely next word given a context string.
    pub fn predict_next_word(&self, context: &str) -> Option<String> {
        let context = context.trim().to_lowercase();

        if let Some(counts) = self.bigrams.get(&context) {
            // Find the word with the highest frequency
            counts
                .iter()
                .max_by_key(|&(_, count)| count)
                .map(|(word, _)| word.clone())
        } else {
            None
        }
    }

    /// Merge another LanguageModel into this one, used by the Aggregator.
    pub fn merge(&mut self, other: &LanguageModel) {
        self.data_points += other.data_points;

        for (context, other_counts) in &other.bigrams {
            let my_counts = self.bigrams.entry(context.clone()).or_insert_with(HashMap::new);
            for (word, count) in other_counts {
                *my_counts.entry(word.clone()).or_insert(0) += count;
            }
        }
    }
}
