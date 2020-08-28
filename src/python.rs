use std::collections::HashMap;
use petgraph::graph::{UnGraph, NodeIndex};
use super::{graph, morse, LabeledPoint};
use pyo3::prelude::*;
use pyo3::exceptions::OSError;


#[pyclass]
pub struct MorseNode {
    #[pyo3(get, set)]
    pub identifier: i64,
    #[pyo3(get, set)]
    pub value: f64,
    #[pyo3(get, set)]
    pub vector: Vec<f64>
}

#[pymethods]
impl MorseNode {
    #[new]
    fn new(identifier: i64, value: f64, vector: Vec<f64>) -> Self {
        MorseNode { identifier, value, vector }
    }
}

// TODO: better unify LabeledPoint and MorseNode
impl From<LabeledPoint<Vec<f64>>> for MorseNode {
    fn from(item: LabeledPoint<Vec<f64>>) -> Self {
        MorseNode{ identifier: item.id, vector: item.point, value: item.value }
    }
}


impl Clone for MorseNode {
    fn clone(&self) -> Self {
        MorseNode{value: self.value, vector: self.vector.clone(), identifier: self.identifier}
    }
}


#[pyclass]
#[derive(Clone)]
pub struct MorseFiltrationStepPy {
    #[pyo3(get)]
    lifetime: f64,
    #[pyo3(get)]
    destroyed_id: i64,
    #[pyo3(get)]
    owning_id: i64
}


impl std::convert::From<morse::MorseError> for PyErr {
    fn from(err: morse::MorseError) -> PyErr {
        OSError::py_err(err.to_string())
    }
}

impl std::convert::From<graph::GraphError> for PyErr {
    fn from(err: graph::GraphError) -> PyErr {
        OSError::py_err(err.to_string())
    }
}


#[pymodule]
fn talus(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MorseNode>()?;
    m.add_class::<MorseFiltrationStepPy>()?;
    m.add_class::<MorseComplexData>()?;

    #[pyfn(m, "_persistence_by_approximate_knn")]
    fn approximate_knn_persistence_py(points: Vec<MorseNode>, k: usize, sample_rate: f64,
                                      precision: f64) -> PyResult<(MorseComplexData, MorseComplexData)> {
        let labeled_points: Vec<LabeledPoint<Vec<f64>>> = points.into_iter().map(|p| p.clone().into()).collect();
        let g = graph::build_knn_approximate(&labeled_points, k, sample_rate, precision)?; 
        let complex = morse::MorseSmaleComplex::from_graph(&g)?;
        let data = complex.to_data(&g);
        Ok(data)
    }

    #[pyfn(m, "_persistence_by_knn")]
    fn knn_persistence_py(points: Vec<MorseNode>, k: usize) -> PyResult<(MorseComplexData, MorseComplexData)> {
        let labeled_points: Vec<LabeledPoint<Vec<f64>>> = points.into_iter().map(|p| p.clone().into()).collect();
        let g = graph::build_knn(&labeled_points, k)?; 
        let complex = morse::MorseSmaleComplex::from_graph(&g)?;
        let data = complex.to_data(&g);
        Ok(data)
    }

    #[pyfn(m, "_persistence")]
    fn persistence_py(nodes: Vec<MorseNode>, edges: Vec<(i64, i64)>) -> PyResult<(MorseComplexData, MorseComplexData)> {
        let mut labeled_nodes: Vec<NodeIndex> = Vec::with_capacity(nodes.len());
        let mut id_lookup: HashMap<i64, (usize, NodeIndex)> = HashMap::with_capacity(nodes.len());
        let mut g = UnGraph::new_undirected();
        for (i, node) in nodes.into_iter().enumerate() {
            let point: LabeledPoint<Vec<f64>> = node.clone().into();
            let node = g.add_node(point.clone());
            labeled_nodes.push(node);
            id_lookup.insert(point.id, (i, node));
        }
        for (left, right) in edges.iter() {
            g.add_edge((id_lookup.get(&left).unwrap()).1, id_lookup.get(&right).unwrap().1, 1.);
        }
        let complex = morse::MorseSmaleComplex::from_graph(&g)?;
        let data = complex.to_data(&g);
        Ok(data)
    }
    Ok(())
}



/// A struct that captures the important data about a MorseSmaleComplex
#[pyclass]
pub struct MorseComplexData {
    #[pyo3(get)]
    pub lifetimes: HashMap<i64, f64>,
    #[pyo3(get)]
    pub filtration: Vec<MorseFiltrationStepPy>,
    #[pyo3(get)]
    pub complex: HashMap<i64, i64>
}


impl morse::MorseSmaleComplex {
    fn to_data<T>(&self, graph: &UnGraph<LabeledPoint<T>, f64>) -> (MorseComplexData, MorseComplexData) {
        (self.descending_complex.to_data(graph), self.ascending_complex.to_data(graph))
    }
}

impl morse::MorseComplex {
    fn to_data<T>(&self, graph: &UnGraph<LabeledPoint<T>, f64>) -> MorseComplexData {
        let lifetimes = self.get_persistence();
        let filtration = &self.filtration;
        let lifetimes: HashMap<i64, f64> = lifetimes.iter()
            .map(|(k,v)| {
                let id = graph.node_weight(*k).unwrap().id;
                (id, *v)
            })
            .collect();
        let filtration: Vec<MorseFiltrationStepPy> = filtration.iter()
            .map(|filtration| {
                MorseFiltrationStepPy {
                    lifetime: filtration.time,
                    destroyed_id: graph.node_weight(filtration.destroyed_cell).unwrap().id,
                    owning_id: graph.node_weight(filtration.owning_cell).unwrap().id
                }
            })
            .collect();
        let complex: HashMap<i64, i64> = self.get_complex().iter()
            .map(|(node, ancestor)| {
                (graph.node_weight(*node).unwrap().id, graph.node_weight(*ancestor).unwrap().id)
            })
            .collect();
        MorseComplexData{lifetimes, filtration, complex}
    }
}
