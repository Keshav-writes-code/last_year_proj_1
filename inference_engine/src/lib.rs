use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

/// Represents a serialized chunk of tensor state to be sent across the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorState {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub step: usize,
}

impl TensorState {
    /// Create a simulated initial state from a user prompt.
    pub fn new_simulated(prompt: &str) -> Self {
        let length = prompt.len();
        // A dummy tensor representing the "embedding" of the prompt
        let data: Vec<f32> = prompt.chars().map(|c| c as u8 as f32 / 255.0).collect();
        Self {
            shape: vec![1, length],
            data,
            step: 0,
        }
    }

    /// Converts back to a candle Tensor for processing.
    pub fn to_tensor(&self, device: &Device) -> candle_core::Result<Tensor> {
        Tensor::from_vec(self.data.clone(), self.shape.clone(), device)
    }

    /// Converts from a candle Tensor.
    pub fn from_tensor(tensor: &Tensor, step: usize) -> candle_core::Result<Self> {
        let flattened = tensor.flatten_all()?;
        let data = flattened.to_vec1::<f32>()?;
        let shape = tensor.dims().to_vec();
        Ok(Self { shape, data, step })
    }
}

pub struct InferenceEngine {
    device: Device,
}

impl Default for InferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceEngine {
    pub fn new() -> Self {
        // Fallback to CPU for simplicity, though real app might use Metal/Cuda
        let device = Device::Cpu;
        Self { device }
    }

    /// Simulates processing a layer. We just perform a simple transformation.
    pub fn process_layer(&self, state: &TensorState) -> candle_core::Result<TensorState> {
        let mut tensor = state.to_tensor(&self.device)?;

        // Dummy operation: add 0.1 to every element to simulate compute.
        // For a real model, this would be a linear layer + activation.
        let ones = Tensor::ones_like(&tensor)?;
        tensor =
            tensor.broadcast_add(&(ones.broadcast_mul(&Tensor::new(0.1f32, &self.device)?)?))?;

        TensorState::from_tensor(&tensor, state.step + 1)
    }

    /// Dummy finalization to simulate generating text from the final logit tensor.
    pub fn finalize(&self, state: &TensorState) -> String {
        // Just take the float values and map them to basic characters as a demo
        let chars: String = state
            .data
            .iter()
            .map(|&f| {
                let val = (f * 255.0) as u8;
                // map it to readable ascii for fun, or just return basic chars
                let c = (val % 26) + 97;
                c as char
            })
            .collect();

        format!("Generated output (len {}): {}", state.data.len(), chars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_state_roundtrip() {
        let state = TensorState::new_simulated("Hello");
        let engine = InferenceEngine::new();
        let tensor = state.to_tensor(&engine.device).unwrap();
        let new_state = TensorState::from_tensor(&tensor, 1).unwrap();
        assert_eq!(state.shape, new_state.shape);
        assert_eq!(state.data, new_state.data);
    }

    #[test]
    fn test_process_layer() {
        let state = TensorState::new_simulated("Hi");
        let engine = InferenceEngine::new();
        let next_state = engine.process_layer(&state).unwrap();
        assert_eq!(next_state.step, 1);
        // values should be larger by ~0.1
        assert!(next_state.data[0] > state.data[0]);
    }
}
