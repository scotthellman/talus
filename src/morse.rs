//! Algorithms for analyzing the behavior of a scalar function over a graph.
use petgraph::graph::{UnGraph, NodeIndex};
use petgraph::unionfind::UnionFind;

use std::collections::{HashSet, HashMap};
use std::hash::{Hash, Hasher};
use std::f64;

use super::LabeledPoint;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MorseError {
    //#[error("descriptive fmt string here")]
    //GraphConstructionFailure (#[from] kdtree::ErrorKind),

    #[error("descriptive fmt string here")]
    NanInValues { },

    //FIXME: these errors should include info about the node

    #[error("descriptive fmt string here")]
    MissingNode {},

    #[error("descriptive fmt string here")]
    BadNeighbor {},

    #[error("descriptive fmt string here")]
    NoMaximum {},

    #[error("descriptive fmt string here")]
    MissingData {}
}

#[derive(Debug)]
struct MorseData {
    lifetime: f64,
    merge_parent: Option<NodeIndex>,
    ancestor: NodeIndex  // TODO: I dunno what the "proper" name for this is
}

#[derive(Debug)]
struct MorseNode {
    node: NodeIndex,
    data: Option<MorseData>
}

impl MorseNode {
    fn new(node: NodeIndex) -> MorseNode {
        MorseNode{node, data: None}
    }
}

impl Hash for MorseNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.hash(state);
    }
}

impl PartialEq for MorseNode {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl Eq for MorseNode {}

#[derive(Debug)]
struct PointedUnionFind {
    unionfind: UnionFind<usize>,
    reprs: Vec<usize>
}

impl PointedUnionFind {
    // this is insanely specific and will break if you use it outside of exactly
    // how it works in the morse complex code (and maybe even if you use it
    // exactly that way!)
    // This turns UnionFind into a structure that always keeps the representative
    // for the left hand size of a union constant. But to do this O(1)
    // i can't do things like "ensure consistency" outside of the access patterns
    // i know the morse complex code will follow
    // (specifically, this data structure offers no guarantees that
    // `find(find(x)) will be reasonable)
    fn new(n: usize) -> Self {
        let unionfind = UnionFind::new(n);
        let reprs = (0..n).collect();
        PointedUnionFind{unionfind, reprs}
    }

    fn find(&self, x: usize) -> usize {
        let inner_repr = self.unionfind.find(x);
        self.reprs[inner_repr]
    }

    fn union(&mut self, x: usize, y: usize) {
        // x is privileged!
        let old_outer = self.find(x);
        self.unionfind.union(x, y);
        let new_inner = self.unionfind.find(x);
        self.reprs[new_inner] = old_outer;
    }
}

/// Contains all of the filtration information for a MorseComplex
///
/// A Morse complex, especially one generated from discrete points of empirical data,
/// may contain extrema that are considered spurious. The filtration of a MorseComplex
/// provides a series of simplifications of that complex, created by merging less 
/// persistent extrema with more persistence extrema. Taken to its conclusion, all
/// extrema will have been merged with the global extreme.
///
/// The MorseFiltrationStep struct contains the information corresponding to one
/// step of this simplification process.
#[derive(Debug, Clone, Copy)]
pub struct MorseFiltrationStep {
    pub time: f64,
    pub destroyed_cell: NodeIndex,
    pub owning_cell: NodeIndex
}

/// Indicates whether a MorseComplex is Ascending or Descending.
///
/// See [MorseComplex](struct.MorseComplex.html) for a detailed explanation.
#[derive(Debug, Clone, Copy)]
pub enum MorseKind {
    Ascending,
    Descending
}

/// Contains both the ascending and descending morse complexes constructed
/// from a graph.
///
/// See [MorseComplex](struct.MorseComplex.html) for a detailed explanation.
#[derive(Debug)]
pub struct MorseSmaleComplex {
    pub ascending_complex: MorseComplex,
    pub descending_complex: MorseComplex
}

impl MorseSmaleComplex {

    /// Constructs a MorseSmaleComplex from the given graph.
    pub fn from_graph<T>(graph: &UnGraph<LabeledPoint<T>, f64>) -> Result<MorseSmaleComplex, MorseError> {
        let ascending_complex = MorseComplex::from_graph(MorseKind::Ascending, &graph)?;
        let descending_complex = MorseComplex::from_graph(MorseKind::Descending, &graph)?;

        Ok(MorseSmaleComplex{ascending_complex, descending_complex})
    }

    pub fn to_data<T>(&self, graph: &UnGraph<LabeledPoint<T>, f64>) -> (MorseComplexData, MorseComplexData) {
        (self.descending_complex.to_data(graph), self.ascending_complex.to_data(graph))
    }
}

/// The Morse complex constructed from a graph.
///
/// A Morse complex is, functionally, a partition of a graph into regions
/// belongs to the various extrema of the graph. For a _descending_ Morse complex,
/// the partitions correspond to maxima, while for an _ascending_ Morse complex,
/// the partitions correspond to minima.
///
/// Computing the Morse complex of a graph necessarily involves computing the
/// _persistence_ of the extrema in the graph. This persistence value is 
/// essentially a quantification of how topologically important that extrema
/// is in the graph, with more "important" extrema having higher persistence.
///
/// The partitions can then be combined with the persistence values to create a 
/// sequence of simplifications of the complex. This is known as a filtration
/// sequence. When computing the filtration sequence, the partitions are merged
/// according to their extrema's persistence, starting with the least persistent
/// partition. 
///
#[derive(Debug)]
pub struct MorseComplex {
    ordered_points: Vec<MorseNode>,
    cells: PointedUnionFind,
    pub filtration: Vec<MorseFiltrationStep>,
    kind: MorseKind
}

impl MorseComplex {
    fn from_graph<T>(kind: MorseKind, graph: &UnGraph<LabeledPoint<T>, f64>) -> Result<MorseComplex, MorseError> {
        let ordered_points = MorseComplex::get_ordered_points(kind, &graph);
        let num_points = ordered_points.len();
        let cells = PointedUnionFind::new(num_points);
        let mut complex = MorseComplex{kind, ordered_points, cells, filtration: vec![]};
        complex.construct_complex(graph)?;
        Ok(complex)
    }

    pub fn to_data<T>(&self, graph: &UnGraph<LabeledPoint<T>, f64>) -> MorseComplexData {
        let lifetimes = self.get_persistence();
        let filtration = &self.filtration;
        let lifetimes: HashMap<i64, f64> = lifetimes.iter()
            .map(|(k,v)| {
                let id = graph.node_weight(*k).unwrap().id;
                (id, *v)
            })
            .collect();
        let filtration: Vec<(f64, i64, i64)> = filtration.iter()
            .map(|filtration| {
                (filtration.time, graph.node_weight(filtration.destroyed_cell).unwrap().id, graph.node_weight(filtration.owning_cell).unwrap().id)
            })
            .collect();
        let complex: Vec<(i64, i64)> = self.get_complex().iter()
            .map(|(node, ancestor)| {
                (graph.node_weight(*node).unwrap().id, graph.node_weight(*ancestor).unwrap().id)
            })
            .collect();
        MorseComplexData{lifetimes, filtration, complex}
    }

    fn get_ordered_points<T>(kind: MorseKind, graph: &UnGraph<LabeledPoint<T>, f64>) -> Vec<MorseNode> {
        let mut nodes: Vec<NodeIndex> = graph.node_indices().collect();
        nodes.sort_by(|a, b| {
                // FIXME: how do i handle errors during the sort?
                let a_node = graph.node_weight(*a).expect("Node a wasn't in graph");
                let b_node = graph.node_weight(*b).expect("Node b wasn't in graph");
                match kind {
                    MorseKind::Descending => b_node.value.partial_cmp(&a_node.value).expect("Nan in the values"),
                    MorseKind::Ascending => a_node.value.partial_cmp(&b_node.value).expect("Nan in the values")
                }
            });
        nodes.iter().enumerate().map(|(_, n)| MorseNode::new(*n)).collect()
    }

    fn compute_filtration(&self) -> Vec<MorseFiltrationStep> {
        let mut filtration = self.ordered_points.iter() 
            .filter_map(|point| {
                match point.data.as_ref() {
                    Some(data) => {
                        if let Some(parent) = data.merge_parent {
                            Some(MorseFiltrationStep{time: data.lifetime, destroyed_cell: point.node, owning_cell: parent})
                        } else {
                            None
                        }
                    }
                    None => None
                }
             })
             .collect::<Vec<_>>();
        filtration.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        filtration
    }

    /// Returns a HashMap mapping nodex to their Morse cell extrema
    pub fn get_complex(&self) -> HashMap<NodeIndex, NodeIndex> {
        self.ordered_points.iter() 
            .filter_map(|point| {
                match point.data.as_ref() {
                    Some(data) => Some((point.node, data.ancestor)),
                    None => None
                }
             })
             .collect()
    }

    /// Returns a mapping of NodeIndices to persistence values.
    ///
    /// Note that, by definition, global extrema have infinite persistence, and non-extrema have 0
    /// persistence
    pub fn get_persistence(&self) -> HashMap<NodeIndex, f64> {
        let mut result = HashMap::with_capacity(self.ordered_points.len());
        for morse_node in self.ordered_points.iter() {
            if let Some(data) = &morse_node.data {
                result.insert(morse_node.node, data.lifetime);
            }         
        }
        result
    }

    fn construct_complex<T>(&mut self, graph: &UnGraph<LabeledPoint<T>, f64>) -> Result<&Self, MorseError>{
        // We iterate through the points in descending (or ascending, depends on self.kind) 
        // order, which means we are essentially building the morse complex at the same time
        // that we compute persistence.

        let inverse_lookup: HashMap<NodeIndex, usize> = self.ordered_points.iter().enumerate()
            .map(|x| (x.1.node, x.0))
            .collect();

        for i in 0..self.ordered_points.len() {
            // find all *already processed* points that we have an edge to
            let this_value = graph.node_weight(self.ordered_points[i].node).unwrap().value;
            let higher_indices: Vec<usize> = graph.neighbors(self.ordered_points[i].node)
                .filter(|n| { match self.kind {
                        MorseKind::Ascending => graph.node_weight(*n).unwrap().value <= this_value,
                        MorseKind::Descending => graph.node_weight(*n).unwrap().value >= this_value
                    }
                })
                .map(|n| *inverse_lookup.get(&n).unwrap())
                .filter(|&n_idx| n_idx < i)
                .collect();

            // Nothing to do if we have no neighbors, but if we do then we
            // have to merge the correspond morse crystals
            let lifetime = if higher_indices.is_empty () {
                f64::INFINITY  
            } else {
                0.
            };
            let ancestor = self.merge_crystals(i, &higher_indices, graph)?;

            // this is not a maximum so it has no lifetime
            self.ordered_points[i].data = Some(MorseData{lifetime, ancestor, merge_parent: None});
        }
        self.filtration = self.compute_filtration();
        Ok(self)
    }

    // FIXME: I don't like this signature. Not at all clear what this returned nodeindex means
    fn merge_crystals<T>(&mut self, ordered_index: usize, ascending_neighbors: &[usize],
                      graph: &UnGraph<LabeledPoint<T>, f64>) -> Result<NodeIndex, MorseError> {
        // If there are no neighbors, there's nothing to merge
        if ascending_neighbors.is_empty() {
            return Ok(self.ordered_points[ordered_index].node);
        }

        // one neighbor is easy, just union this point in to that neighbor's crystal
        if ascending_neighbors.len() == 1 {
            let neighbor_index = ascending_neighbors[0];
            self.cells.union(neighbor_index, ordered_index);

            return Ok(self.ordered_points[neighbor_index].data.as_ref().ok_or_else(|| {
                MorseError::MissingData{}})?.ancestor);
        }

        // for multiple neighbors, first figure out if all neighbors are in the same crystal
        let connected_crystals: HashSet<_> = ascending_neighbors.iter()
            .map(|&idx| self.cells.find(idx))
            .collect();

        // If they are all in the same crystal, it's the same as if there was just one neighbor
        if connected_crystals.len() == 1 {
            let neighbor_index = ascending_neighbors[0];
            self.cells.union(neighbor_index, ordered_index);

            return Ok(self.ordered_points[neighbor_index].data.as_ref().ok_or_else(|| {
                MorseError::MissingData{}})?.ancestor);
        }

        // And if we're here then we're merging crystals
        // first figure out what the global max is
        let (_, max_crystal) = connected_crystals.iter()
            .map(|&idx| {
                let node = &self.ordered_points[idx];
                // FIXME: this expect should be be an error
                let value = graph.node_weight(node.node).expect("max wasn't in the graph").value;
                (value, idx)
            })
            .max_by(|a, b| match self.kind {
                //TODO: would be nice to pull this logic out
                    MorseKind::Descending => a.0.partial_cmp(&b.0).expect("Nan in the values"),
                    MorseKind::Ascending => b.0.partial_cmp(&a.0).expect("Nan in the values")
                }
            )
            .ok_or(MorseError::NoMaximum{})?;

        let joining_node = &self.ordered_points[ordered_index];

        let (_, steepest_neighbor) = ascending_neighbors.iter()
            .map(|&idx| {
                let node = &self.ordered_points[idx];
                let value = graph.node_weight(node.node).unwrap().value;
                let edge = graph.find_edge(joining_node.node, node.node).expect("A neighbor wasn't really a neighbor");
                // unwrap_or here because the persistence calculation is still meaningful
                // even we we aren't in a metric space, so lack of grade information shouldn't
                // block the computation
                // FIXME: i'm not sure that i entirely believe the above comment
                (value / *graph.edge_weight(edge).unwrap_or(&1.), idx)
            })
            .max_by(|a, b| match self.kind {
                    MorseKind::Descending => a.0.partial_cmp(&b.0).expect("Nan in the values"),
                    MorseKind::Ascending => b.0.partial_cmp(&a.0).expect("Nan in the values")
                }
            )
            .ok_or(MorseError::NoMaximum{})?;

        // now we need to update the lifetimes and merge the other crystals
        let joining_value = graph.node_weight(joining_node.node).ok_or(MorseError::MissingNode{})?.value;
        let merge_parent = self.ordered_points[max_crystal].node;
        self.cells.union(max_crystal, ordered_index);
        for crystal in connected_crystals {
            if crystal != max_crystal {
                let crystal_node = &self.ordered_points[crystal];
                let crystal_value = graph.node_weight(crystal_node.node).ok_or(MorseError::MissingNode{})?.value;
                let ancestor = self.ordered_points[crystal].data.as_ref().ok_or(MorseError::MissingData{})?.ancestor;

                // abs here so that the math works for ascending or descending
                let lifetime = (crystal_value - joining_value).abs();
                self.ordered_points[crystal].data = Some(MorseData{ancestor, lifetime, 
                    merge_parent: Some(merge_parent)});
                self.cells.union(max_crystal, crystal);
            }
        }
        Ok(self.ordered_points[steepest_neighbor].data.as_ref().ok_or(MorseError::MissingData{})?.ancestor)
    }
}
//
// FIXME: still need more types here
/// A struct that captures the important data about a MorseSmaleComplex
pub struct MorseComplexData {
    pub lifetimes: HashMap<i64, f64>,
    pub filtration: Vec<(f64, i64, i64)>,
    pub complex: Vec<(i64, i64)>
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single() {
        let mut graph = UnGraph::new_undirected();
        let points = [
            LabeledPoint{id: 0, value: -1., point: vec![0., 0.]},
            LabeledPoint{id: 1, value: 1., point: vec![1., 0.]},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 0.);
        let complex = MorseComplex::from_graph(MorseKind::Descending, &graph);
        let lifetimes = complex.get_persistence();
        assert_eq!(lifetimes[&node_lookup[0]], 0.);
        assert_eq!(lifetimes[&node_lookup[1]], f64::INFINITY);
    }

    #[test]
    fn test_triangle() {
        let mut graph = UnGraph::new_undirected();
        let points = [
            LabeledPoint{id: 0, value: -1., point: vec![0., 0.]},
            LabeledPoint{id: 1, value: 0., point: vec![1., 1.]},
            LabeledPoint{id: 2, value: 1., point: vec![1., 0.]},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 0.);
        graph.add_edge(node_lookup[0], node_lookup[2], 0.);
        graph.add_edge(node_lookup[1], node_lookup[2], 0.);
        let complex = MorseComplex::from_graph(MorseKind::Descending, &graph);
        let lifetimes = complex.get_persistence();
        assert_eq!(lifetimes[&node_lookup[0]], 0.);
        assert_eq!(lifetimes[&node_lookup[1]], 0.);
        assert_eq!(lifetimes[&node_lookup[2]], f64::INFINITY);
    }

    #[test]
    fn test_square() {
        let mut graph = UnGraph::new_undirected();
        let points = [
            LabeledPoint{id: 0, value: 1., point: vec![0., 0.]},
            LabeledPoint{id: 1, value: -1., point: vec![1., 0.]},
            LabeledPoint{id: 2, value: 0., point: vec![0., 1.]},
            LabeledPoint{id: 3, value: 2., point: vec![1., 1.]},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 0.);
        graph.add_edge(node_lookup[0], node_lookup[2], 0.);
        graph.add_edge(node_lookup[1], node_lookup[3], 0.);
        graph.add_edge(node_lookup[2], node_lookup[3], 0.);
        let complex = MorseComplex::from_graph(MorseKind::Descending, &graph);
        let lifetimes = complex.get_persistence();
        assert_eq!(lifetimes[&node_lookup[0]], 1.);
        assert_eq!(lifetimes[&node_lookup[1]], 0.);
        assert_eq!(lifetimes[&node_lookup[2]], 0.);
        assert_eq!(lifetimes[&node_lookup[3]], f64::INFINITY);
    }

    #[test]
    fn test_all_equal_values() {
        let mut graph = UnGraph::new_undirected();
        let points = [
            LabeledPoint{id: 0, value: 0., point: vec![0., 0.]},
            LabeledPoint{id: 1, value: 0., point: vec![1., 0.]},
            LabeledPoint{id: 2, value: 0., point: vec![0., 1.]},
            LabeledPoint{id: 3, value: 0., point: vec![1., 1.]},
            LabeledPoint{id: 4, value: 1., point: vec![1., 1.]},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 1.);
        graph.add_edge(node_lookup[0], node_lookup[2], 1.);
        graph.add_edge(node_lookup[1], node_lookup[3], 1.);
        graph.add_edge(node_lookup[2], node_lookup[3], 1.);
        graph.add_edge(node_lookup[2], node_lookup[4], 1.);
        let complex = MorseComplex::from_graph(MorseKind::Descending, &graph);
        let lifetimes = complex.get_persistence();
        println!("{:?}", lifetimes);
        assert_eq!(lifetimes[&node_lookup[0]], 0.);
        assert_eq!(lifetimes[&node_lookup[1]], 0.);
        assert_eq!(lifetimes[&node_lookup[2]], 0.);
        assert_eq!(lifetimes[&node_lookup[3]], 0.);
        assert_eq!(lifetimes[&node_lookup[4]], f64::INFINITY);
    }

    #[test]
    fn test_big_square_morse_smale() {
        let mut graph = UnGraph::new_undirected();
        let points = [
            LabeledPoint{id: 0, value: 6., point: vec![0., 0.]},
            LabeledPoint{id: 1, value: 2., point: vec![1., 0.]},
            LabeledPoint{id: 2, value: 3., point: vec![2., 0.]},
            LabeledPoint{id: 3, value: 5., point: vec![0., 1.]},
            LabeledPoint{id: 4, value: 4., point: vec![1., 1.]},
            LabeledPoint{id: 5, value: -5., point: vec![1., 2.]},
            LabeledPoint{id: 6, value: 0., point: vec![0., 2.]},
            LabeledPoint{id: 7, value: 1., point: vec![1., 2.]},
            LabeledPoint{id: 8, value: 10., point: vec![2., 2.]},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 1.);
        graph.add_edge(node_lookup[1], node_lookup[2], 1.);
        graph.add_edge(node_lookup[0], node_lookup[3], 1.);
        graph.add_edge(node_lookup[1], node_lookup[4], 1.);
        graph.add_edge(node_lookup[2], node_lookup[5], 1.);
        graph.add_edge(node_lookup[3], node_lookup[4], 1.);
        graph.add_edge(node_lookup[4], node_lookup[5], 1.);
        graph.add_edge(node_lookup[3], node_lookup[6], 1.);
        graph.add_edge(node_lookup[4], node_lookup[7], 1.);
        graph.add_edge(node_lookup[5], node_lookup[8], 1.);
        graph.add_edge(node_lookup[6], node_lookup[7], 1.);
        graph.add_edge(node_lookup[7], node_lookup[8], 1.);
        let complex = MorseSmaleComplex::from_graph(&graph);
        let lifetimes = complex.descending_complex.get_persistence();
        assert_eq!(lifetimes[&node_lookup[0]], 5.);
        assert_eq!(lifetimes[&node_lookup[1]], 0.);
        assert_eq!(lifetimes[&node_lookup[2]], 1.);
        assert_eq!(lifetimes[&node_lookup[3]], 0.);
        assert_eq!(lifetimes[&node_lookup[4]], 0.);
        assert_eq!(lifetimes[&node_lookup[5]], 0.);
        assert_eq!(lifetimes[&node_lookup[6]], 0.);
        assert_eq!(lifetimes[&node_lookup[7]], 0.);
        assert_eq!(lifetimes[&node_lookup[8]], f64::INFINITY);

        let lifetimes = complex.ascending_complex.get_persistence();
        println!("{:?}", lifetimes);
        assert_eq!(lifetimes[&node_lookup[0]], 0.);
        assert_eq!(lifetimes[&node_lookup[1]], 1.);
        assert_eq!(lifetimes[&node_lookup[2]], 0.);
        assert_eq!(lifetimes[&node_lookup[3]], 0.);
        assert_eq!(lifetimes[&node_lookup[4]], 0.);
        assert_eq!(lifetimes[&node_lookup[5]], f64::INFINITY);
        assert_eq!(lifetimes[&node_lookup[6]], 4.);
        assert_eq!(lifetimes[&node_lookup[7]], 0.);
        assert_eq!(lifetimes[&node_lookup[8]], 0.);
    }

    #[test]
    fn test_filtration() {
        let mut graph = UnGraph::new_undirected();
        let points = [
            LabeledPoint{id: 0, value: 3., point: vec![0., 0.]},
            LabeledPoint{id: 1, value: -1., point: vec![1., 0.]},
            LabeledPoint{id: 2, value: 10., point: vec![0., 1.]},
            LabeledPoint{id: 3, value: 2., point: vec![1., 1.]},
            LabeledPoint{id: 4, value: 7., point: vec![1., 1.]},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 0.);
        graph.add_edge(node_lookup[0], node_lookup[3], 0.);
        graph.add_edge(node_lookup[1], node_lookup[2], 0.);
        graph.add_edge(node_lookup[1], node_lookup[4], 0.);
        graph.add_edge(node_lookup[3], node_lookup[4], 0.);
        let complex = MorseComplex::from_graph(MorseKind::Descending, &graph);
        let lifetimes = complex.get_persistence();
        println!("{:?}", lifetimes);
        assert_eq!(lifetimes[&node_lookup[0]], 1.);
        assert_eq!(lifetimes[&node_lookup[1]], 0.);
        assert_eq!(lifetimes[&node_lookup[2]], f64::INFINITY);
        assert_eq!(lifetimes[&node_lookup[3]], 0.);
        assert_eq!(lifetimes[&node_lookup[4]], 8.);

        let filtration = complex.filtration;
        let expected = [(1., node_lookup[0], node_lookup[4]), (8., node_lookup[4], node_lookup[2])];
        for (actual, expected) in filtration.iter().zip(expected.iter()) {
            assert_eq!(actual.time, expected.0);
            assert_eq!(actual.destroyed_cell, expected.1);
            assert_eq!(actual.owning_cell, expected.2);
        }
    }
}
