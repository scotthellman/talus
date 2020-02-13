//! Algorithms for constructing graphs from sets of points
use std::hash::{Hash, Hasher};
use petgraph::graph::UnGraph;
use kdtree::KdTree;
use kdtree::distance::squared_euclidean;
use std::collections::HashSet;
use std::f64;
use rand::prelude::*;

use super::LabeledPoint;

#[derive(Debug, Clone, Copy)]
enum NeighborState {
    New,
    Old
}

impl NeighborState {
    fn is_new(self) -> bool {
        match self {
            NeighborState::New => true,
            NeighborState::Old => false
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NeighborData {
    distance: f64,
    idx: usize,
    state: NeighborState
}

impl Hash for NeighborData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.idx.hash(state);
    }
}

impl PartialEq for NeighborData {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
    }
}

impl Eq for NeighborData {}


fn sample_neighbors(neighbors: &mut Vec<NeighborData>, sample_rate: f64) -> HashSet<NeighborData> {
    let mut rng = rand::thread_rng();
    let mut targets = HashSet::with_capacity(neighbors.len());
    for neighbor in neighbors.iter_mut() {
        match neighbor.state {
            NeighborState::New => {
                if rng.gen_range(0., 1.) < sample_rate {
                    targets.insert(*neighbor);
                    neighbor.state = NeighborState::Old;
                }
            },
            NeighborState::Old => {
                targets.insert(*neighbor);
            }
        }
    }
    targets
}

fn add_reversed_targets(targets: &mut Vec<HashSet<NeighborData>>) {
    // TODO: i don't love this
    let acc: Vec<(usize, NeighborData)> = targets.iter().enumerate()
        .flat_map(|(i, neighbors)| {
            neighbors.iter()
                .map(move |neighbor| {
                    (neighbor.idx, NeighborData{ distance: neighbor.distance, idx: i, state: neighbor.state})
                })
            })
        .collect();
    for (idx, data) in acc {
        targets[idx].insert(data);
    }
}

fn update_neighbors(data: &mut Vec<NeighborData>, data_idx: usize, neighbor: usize, distance: f64, k: usize) -> bool {
    // this is the only time we touch the vec of data, so if we end up sorted then we can 
    // assume we started sorted
    // TODO: this should really bisect but linear is harder to mess up for now
    //   (let the record show that this was where the major bug in my impl was anyway)

    if data_idx == neighbor {return false};
    let mut index = None;
    for (i, n_data) in data.iter().enumerate() {
        if n_data.idx == neighbor {
            return false;
        }
        if distance < n_data.distance {
            index = Some(i);
            break
        }
    }
    if let Some(index) = index {

        data.insert(index, NeighborData{distance, idx: neighbor, state: NeighborState::New});
        data.truncate(k);
        return true;
    }
    false
}

fn rejection_sample(count: usize, range: usize, rng: &mut ThreadRng) -> Vec<usize> {
    if range < count {
        // FIXME yeah this can't stay like this
        panic!();
    }
    let mut sample: HashSet<usize> = HashSet::with_capacity(count);
    while sample.len() < count {
        sample.insert(rng.gen_range(0, range));
    }
    sample.iter().copied().collect()
}


/// Constructs an approximate `k`-NN graph from a set of `points`.
///
/// `sample_rate` controls how many new neighbors are considered at each step. Lower values will
/// reduce runtime but reduce accuracy.
///
/// `precision` controls early stopping of the computation. Lower values will increase runtime
/// but increase accuracy.
///
/// This is an implementation of the algoirhtm described in [Efficient K-Nearest Neighbor Graph
/// Construction for Generic Similarity
/// Measures](https://www.cs.princeton.edu/cass/papers/www11.pdf). How much faster this is than the
/// equivalent exact kNN computation depends on how slow the similarity function is to evaluate.
/// For very fast distance calculations, this can be slower than the exact computation.
/// Note that this _does not_ require the similarity function to be a distance metric.
pub fn build_knn_approximate(points: &[LabeledPoint], k: usize, sample_rate: f64, precision: f64) 
    -> UnGraph<LabeledPoint, f64> {
    // https://www.cs.princeton.edu/cass/papers/www11.pdf

    let mut rng = rand::thread_rng();
    let mut approximate_neighbors: Vec<Vec<NeighborData>> = (0..points.len())
        .map(|_| {
            let points = rejection_sample(k, points.len(), &mut rng);
            points.iter()
                .map(|&j| NeighborData{distance: f64::INFINITY, idx:j, state: NeighborState::New})
                .collect()
        })
        .collect();
    let mut done = false;
    let mut iters = 0;
    while !done {
        iters += 1;
        // I'm making targets be a vec of hashsets
        // because redudnant targets is a very real possibility and i need an efficient solution
        // to account for those
        let mut targets: Vec<HashSet<NeighborData>> = approximate_neighbors.iter_mut()
            .map(|neighbors| sample_neighbors(neighbors, sample_rate))
            .collect();
        add_reversed_targets(&mut targets);

        let mut counter = 0;

        for targ in targets.iter() {
            for (i, target) in targ.iter().enumerate() {
                if let NeighborState::New = target.state {
                    for (j, other) in targ.iter().enumerate() {
                        if j < i || !other.state.is_new() {
                            let distance = points[target.idx].distance(&points[other.idx]);
                            let changed = update_neighbors(&mut approximate_neighbors[target.idx], target.idx, other.idx, distance, k);
                            if changed {counter += 1};
                            let changed = update_neighbors(&mut approximate_neighbors[other.idx], other.idx, target.idx, distance, k);
                            if changed {counter += 1};
                        }
                    }
                }
            }
        }

        done = counter <= (precision * points.len() as f64 * k as f64) as i64;
        if iters > 2000 {
            // FIXME: This should probably be more graceful than full-on panicking 
            // TODO: A timeout would be nice - there's a perfectly usable graph at every step after
            // all
            panic!();
        }
    }

    graph_from_neighbordata(points, approximate_neighbors)
}

fn graph_from_neighbordata(points: &[LabeledPoint], neighbors: Vec<Vec<NeighborData>>)
    -> UnGraph<LabeledPoint, f64> {
    let mut neighbor_graph = UnGraph::new_undirected();
    let mut node_lookup = Vec::with_capacity(points.len());
    for point in points {
        let node = neighbor_graph.add_node(point.to_owned());
        node_lookup.push(node);
    }

    for (i, data) in neighbors.iter().enumerate() {
        for neighbor in data{
            neighbor_graph.update_edge(node_lookup[i], node_lookup[neighbor.idx], neighbor.distance);
        }
    }

    neighbor_graph
}


/// Constructs an exact `k`-NN graph from a set of `points`.
///
/// This implementation uses a KD-tree for efficient nearest neighbor querying.
pub fn build_knn(points: &[LabeledPoint], k: usize) -> UnGraph<LabeledPoint, f64> {
    let dim = points[0].point.len();
    let mut tree = KdTree::new(dim);
    for (i, point) in points.iter().enumerate() {
        tree.add(&point.point, i).unwrap();
    }
    let mut neighbor_graph = UnGraph::new_undirected();
    let mut node_lookup = Vec::with_capacity(points.len());
    for point in points {
        let node = neighbor_graph.add_node(point.to_owned());
        node_lookup.push(node);
    }
    for (i, point) in points.iter().enumerate() {
        tree.iter_nearest(&point.point, &squared_euclidean)
            .unwrap()
            .skip(1)  // always returns itself as the first one
            .take(k)
            .for_each(|(dist, &j)| {
                neighbor_graph.update_edge(node_lookup[i], node_lookup[j], dist);
            })
    }
    neighbor_graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_knn() {
        let points = [
            LabeledPoint{id: 0, value: 6., point: vec![0., 0.]},
            LabeledPoint{id: 1, value: 2., point: vec![1., 0.]},
            LabeledPoint{id: 2, value: 3., point: vec![1.5, 0.]},
            LabeledPoint{id: 3, value: 5., point: vec![0., 0.7]},
            LabeledPoint{id: 4, value: 4., point: vec![1., 1.]},
            LabeledPoint{id: 5, value: -5., point: vec![0., 2.]},
            LabeledPoint{id: 6, value: 0., point: vec![2., 3.]}
        ];
        let mut expected_adjacencies = HashMap::with_capacity(7);
        expected_adjacencies.insert(0, vec![1, 3]);
        expected_adjacencies.insert(1, vec![0, 2, 4]);
        expected_adjacencies.insert(2, vec![1, 4]);
        expected_adjacencies.insert(3, vec![0, 4, 5]);
        expected_adjacencies.insert(4, vec![1, 2, 3, 5, 6]);
        expected_adjacencies.insert(5, vec![3, 4, 6]);
        expected_adjacencies.insert(6, vec![4, 5]);

        let g = build_knn(&points, 2);
        for node in g.node_indices() {
            let id = g.node_weight(node).unwrap().id;
            let adj_ids: HashSet<i64> = g.neighbors(node)
                .map(|n| g.node_weight(n).unwrap().id)
                .collect();
            let expected = expected_adjacencies.get(&id).unwrap();
            assert_eq!(expected.len(), adj_ids.len());
            for exp in expected {
                assert!(adj_ids.contains(exp));
            }

        }
    }

    #[test]
    fn test_knn_approximate() {
        // Strictly speaking I'm not sure that these assertions are guaranteed to be correct
        let points = [
            LabeledPoint{id: 0, value: 6., point: vec![0., 0.]},
            LabeledPoint{id: 1, value: 2., point: vec![1., 0.]},
            LabeledPoint{id: 2, value: 3., point: vec![1.5, 0.]},
            LabeledPoint{id: 3, value: 5., point: vec![0., 0.7]},
            LabeledPoint{id: 4, value: 4., point: vec![1., 1.]},
            LabeledPoint{id: 5, value: -5., point: vec![0., 2.]},
            LabeledPoint{id: 6, value: 0., point: vec![2., 3.]}
        ];
        let mut expected_adjacencies = HashMap::with_capacity(7);
        expected_adjacencies.insert(0, vec![1, 3]);
        expected_adjacencies.insert(1, vec![0, 2, 4]);
        expected_adjacencies.insert(2, vec![1, 4]);
        expected_adjacencies.insert(3, vec![0, 4, 5]);
        expected_adjacencies.insert(4, vec![1, 2, 3, 5, 6]);
        expected_adjacencies.insert(5, vec![3, 4, 6]);
        expected_adjacencies.insert(6, vec![4, 5]);

        let g = build_knn_approximate(&points, 2, 0.8, 0.01);
        println!("{:?}", g);
        for node in g.node_indices() {
            let id = g.node_weight(node).unwrap().id;
            let adj_ids: HashSet<i64> = g.neighbors(node)
                .map(|n| g.node_weight(n).unwrap().id)
                .collect();
            let expected = expected_adjacencies.get(&id).unwrap();
            assert_eq!(expected.len(), adj_ids.len());
            for exp in expected {
                assert!(adj_ids.contains(exp));
            }

        }
    }
}
