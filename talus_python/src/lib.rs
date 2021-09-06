use talus;
pub mod python;

impl From<python::MorseNode> for talus::LabeledPoint<Vec<f64>> {
    fn from(item: python::MorseNode) -> Self {
        talus::LabeledPoint{value: item.value, point: item.vector, id: item.identifier}
    }
}
