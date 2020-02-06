use ndarray::prelude::*;
use std::hash::{Hash, Hasher};
use std::cmp::Ord;
use itertools::Itertools;
use petgraph::graph::{Graph, NodeIndex};
use kdtree::KdTree;
use kdtree::distance::squared_euclidean;
use kdtree::ErrorKind;
use std::collections::{HashSet, HashMap};
use std::f64;
use rand::prelude::*;
use std::cmp::Ordering;

use super::LabeledPoint;

#[derive(Debug, Clone, Copy)]
enum NeighborState {
    New,
    Old
}

impl NeighborState {
    fn is_new(&self) -> bool {
        match self {
            NeighborState::New => true,
            NeighborState::Old => false
        }
    }
}

// FIXME: there is 0 reason this should be specific to NeighorData
#[derive(Debug)]
struct MorseHeap {
    heap: Vec<NeighborData>,
    num_rows: usize,
    num_cols: usize
}

impl MorseHeap {
    fn new(num_rows: usize, num_cols: usize) -> MorseHeap {
        // TODO: this doesn't feel great. These should probably be options, but how much
        // does that cost?
        let heap = (0..num_rows*num_cols)
            .map(|_| NeighborData{distance: f64::INFINITY, idx: 0, state: NeighborState::New})
            .collect();
        MorseHeap{heap, num_rows, num_cols}
    }

    fn new_with_indices(indices: &[usize], num_rows: usize, num_cols: usize) -> MorseHeap {
        // TODO: this doesn't feel great. These should probably be options, but how much
        // does that cost?
        let heap = (0..num_rows*num_cols)
            .map(|i| NeighborData{distance: f64::INFINITY, idx: indices[i], state: NeighborState::New})
            .collect();
        MorseHeap{heap, num_rows, num_cols}
    }

    // this should really be an impl of Index for MorseHeap
    fn get_heap(&self, i: usize) -> &[NeighborData] {
        &self.heap[i*self.num_cols.. i*self.num_cols + self.num_cols]
    }

    fn get_heap_mut(&mut self, i: usize) -> &mut [NeighborData] {
        &mut self.heap[i*self.num_cols.. i*self.num_cols + self.num_cols]
    }

    fn insert(&mut self, data: NeighborData, i: usize) {
        let heap = self.get_heap_mut(i);
        // should probably guard here if data > the root
        heap[0] = data;
        MorseHeap::heapify(heap, 0);
    }

    fn heapify(heap: &mut [NeighborData], i: usize) {
        let left = i*2 + 1;
        let right = i*2 + 2;
        let mut largest = i;

        if left < heap.len() && heap[left].distance > heap[largest].distance {
            largest = left;
        }
        if right < heap.len() && heap[right].distance > heap[largest].distance {
            largest = right;
        }

        if largest != i {
            heap.swap(i, largest);
            MorseHeap::heapify(heap, largest);
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


// FIXME: ugh this heap row thing is bad
fn sample_neighbors(neighbors: &mut [NeighborData], sample_rate: f64, heap: &mut MorseHeap, heap_row: usize) {
    let mut rng = rand::thread_rng(); //FIXME: This should probably be threaded through the call
    for neighbor in neighbors.iter_mut() {
        let weight: f64 = rng.gen();
        match neighbor.state {
            NeighborState::New => {
                if rng.gen_range(0., 1.) < sample_rate {
                    let target = NeighborData{distance: weight, idx:neighbor.idx, state: neighbor.state};
                    heap.insert(target, heap_row);
                    neighbor.state = NeighborState::Old;
                }
            },
            NeighborState::Old => {
                let neighbor = NeighborData{distance: weight, idx:neighbor.idx, state: neighbor.state};
                heap.insert(neighbor, heap_row);
            }
        }
    }
}

fn add_reversed_targets(targets: &mut MorseHeap) {
    // TODO: i don't love this
    let acc: Vec<(usize, NeighborData)> = (0..targets.num_rows)
        .flat_map(|i| {
            targets.get_heap(i).iter()
                .map(move |neighbor| {
                    (neighbor.idx, NeighborData{ distance: neighbor.distance, idx: i, state: neighbor.state})
                })
            })
        .collect();
    for (idx, data) in acc {
        targets.insert(data, idx);
    }
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

pub fn build_knn_approximate(points: &[LabeledPoint], k: usize, sample_rate: f64, precision: f64) 
    -> Graph<LabeledPoint, f64, petgraph::Undirected> {
    // https://www.cs.princeton.edu/cass/papers/www11.pdf

    let max_candidates = 30;
    let mut rng = rand::thread_rng();
    let starting_indices: Vec<usize> = (0..points.len())
        .flat_map(|_| {
            rejection_sample(k, points.len(), &mut rng)
        })
        .collect();
    let mut approximate_neighbors = MorseHeap::new_with_indices(&starting_indices, points.len(), k);
    let mut done = false;
    let mut iters = 0;
    while !done {
        iters += 1;

        //cribbing this from the pynndescent impl
        let mut targets = MorseHeap::new(points.len(), max_candidates);
        // FIXME: MorseHeap should be iter
        for i in 0..points.len() {
            sample_neighbors(approximate_neighbors.get_heap_mut(i), sample_rate, &mut targets, i);
        }
        add_reversed_targets(&mut targets);

        let mut counter = 0;

        println!("{:?}", approximate_neighbors);

        for row in 0..points.len() {
            let target_heap = targets.get_heap(row);
            for (i, target) in target_heap.iter().enumerate() {
                if let NeighborState::New = target.state {
                    for (j, other) in target_heap.iter().enumerate() {
                        // FIXME: None of this works of ascending.
                        if j < i || !other.state.is_new() {
                            let distance = points[target.idx].distance(&points[other.idx]);
                            if update_neighbors(&mut approximate_neighbors, target.idx, other.idx, distance) {
                                counter += 1;
                            }
                            if update_neighbors(&mut approximate_neighbors, other.idx, target.idx, distance) {
                                counter += 1;
                            }
                        }
                    }
                }
            }
        }

        done = counter <= (precision * points.len() as f64 * k as f64) as i64;
        if iters > 2000 {
            // FIXME: This should probably be more graceful than full-on panicking 
            panic!();
        }
    }

    graph_from_neighbordata(points, &approximate_neighbors)
}

fn update_neighbors(heap: &mut MorseHeap, data_idx: usize, neighbor: usize, distance: f64) -> bool {
    if data_idx == neighbor {return false};
    let row = heap.get_heap_mut(data_idx);
    if row[0].distance < distance {
        return false;
    }
    for n_data in row.iter() {
        if n_data.idx == neighbor {
            return false;
        }
    }
    println!("OK! At this point, {} is not in {:?}", neighbor, row);

    heap.insert(NeighborData{distance, idx: neighbor, state: NeighborState::New}, data_idx);
    true
}

fn graph_from_neighbordata(points: &[LabeledPoint], neighbors: &MorseHeap)
    -> Graph<LabeledPoint, f64, petgraph::Undirected> {
    let mut neighbor_graph = Graph::new_undirected();
    let mut node_lookup = Vec::with_capacity(points.len());
    for point in points {
        let node = neighbor_graph.add_node(point.to_owned());
        node_lookup.push(node);
    }

    for row in 0..points.len(){
        let data = neighbors.get_heap(row);
        for neighbor in data {
            neighbor_graph.update_edge(node_lookup[row], node_lookup[neighbor.idx], neighbor.distance);
        }
    }

    neighbor_graph
}


pub fn build_knn(points: &[LabeledPoint], k: usize) -> Graph<LabeledPoint, f64, petgraph::Undirected> {
    let dim = points[0].point.len();
    let mut tree = KdTree::new(dim);
    for (i, point) in points.iter().enumerate() {
        tree.add(&point.point, i).unwrap();
    }
    let mut neighbor_graph = Graph::new_undirected();
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
