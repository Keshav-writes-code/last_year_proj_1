use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use candle_core::{Tensor, Device, DType};

/// A serializable representation of a neural network's weights.
/// Maps variable names (e.g., "linear1.weight", "linear1.bias") to 1D vectors of raw float32 data.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelWeights {
    pub layers: HashMap<String, LayerData>,
    pub data_points: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerData {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl ModelWeights {
    pub fn new() -> Self {
        Self::default()
    }

    /// Helper to convert a dictionary of candle Tensors into our serializable ModelWeights.
    pub fn from_tensors(tensors: &HashMap<String, Tensor>, data_points: u64) -> candle_core::Result<Self> {
        let mut layers = HashMap::new();

        for (name, tensor) in tensors {
            let shape = tensor.shape().dims().to_vec();
            // Flatten the tensor to a 1D vector of f32 for serialization
            let flattened = tensor.flatten_all()?;
            let data: Vec<f32> = flattened.to_vec1()?;
            layers.insert(name.clone(), LayerData { shape, data });
        }

        Ok(Self { layers, data_points })
    }

    /// Helper to reconstruct candle Tensors from the serialized ModelWeights.
    pub fn to_tensors(&self, device: &Device) -> candle_core::Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();

        for (name, layer_data) in &self.layers {
            let tensor = Tensor::from_vec(layer_data.data.clone(), layer_data.shape.as_slice(), device)?;
            tensors.insert(name.clone(), tensor);
        }

        Ok(tensors)
    }

    /// Averages another `ModelWeights` into this one, weighted by the number of data points.
    /// This is the core Federated Learning aggregation step.
    pub fn merge(&mut self, other: &ModelWeights) {
        if self.data_points == 0 {
            self.layers = other.layers.clone();
            self.data_points = other.data_points;
            return;
        }

        let total_data = self.data_points + other.data_points;
        let self_weight = self.data_points as f32 / total_data as f32;
        let other_weight = other.data_points as f32 / total_data as f32;

        for (name, my_layer) in self.layers.iter_mut() {
            if let Some(other_layer) = other.layers.get(name) {
                // Ensure shapes match
                if my_layer.shape == other_layer.shape {
                    for i in 0..my_layer.data.len() {
                        my_layer.data[i] = (my_layer.data[i] * self_weight) + (other_layer.data[i] * other_weight);
                    }
                }
            }
        }

        self.data_points = total_data;
    }
}
