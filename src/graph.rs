use ndarray::prelude::*;
use std::hash::{Hash, Hasher};
use std::cmp::Ord;
use itertools::Itertools;
use petgraph::graph::{Graph, NodeIndex};
use kdtree::KdTree;
use kdtree::distance::squared_euclidean;
use kdtree::ErrorKind;
use std::collections::HashSet;
use std::f64;
use rand::prelude::*;
use std::cmp::Ordering;

use super::LabeledPoint;

#[derive(Debug, Clone, Copy)]
enum NeighborState {
    New,
    Old
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

#[derive(Debug)]
struct TargetNeighbors {
    old: Vec<NeighborData>,
    new: Vec<NeighborData>
}


fn sample_neighbors(neighbors: &mut Vec<NeighborData>, sample_rate: f64) -> HashSet<NeighborData> {
    let mut rng = rand::thread_rng(); //FIXME: This should probably be threaded through the call
    let mut targets = HashSet::with_capacity(neighbors.len());
    for neighbor in neighbors.iter_mut() {
        match neighbor.state {
            NeighborState::New => {
                if rng.gen_range(0., 1.) < sample_rate {
                    neighbor.state = NeighborState::Old;
                    targets.insert(*neighbor);
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
    // FIXME: I shouldn't have to build this. Just add on to target neighbors in some
    // principled way?
    for (i, neighbors) in targets.iter().enumerate() {
        for neighbor in neighbors.iter() {

        }
    }
    let capacity = targets[0].old.len();
    let mut reversed: Vec<_> = (0..targets.len())
        .map(|_| TargetNeighbors{old: Vec::with_capacity(capacity), new: Vec::with_capacity(capacity)})
        .collect();
    for (i, target) in targets.iter().enumerate() {
        for neighbor in target.old.iter() {
            reversed[neighbor.idx].old.push(NeighborData{
                distance: neighbor.distance,
                idx: i,
                state: neighbor.state
            });
        }
        for neighbor in target.new.iter() {
            reversed[neighbor.idx].new.push(NeighborData{
                distance: neighbor.distance,
                idx: i,
                state: neighbor.state
            });
        }
    }
    reversed
}

impl TargetNeighbors {
    fn sample_neighbors(neighbors: &mut Vec<NeighborData>, sample_rate: f64) -> TargetNeighbors {
        let mut rng = rand::thread_rng(); //FIXME: This should probably be threaded through the call
        let mut old = Vec::with_capacity(neighbors.len());
        let mut new = Vec::with_capacity(neighbors.len());
        for neighbor in neighbors.iter_mut() {
            match neighbor.state {
                NeighborState::New => {
                    if rng.gen_range(0., 1.) < sample_rate {
                        neighbor.state = NeighborState::Old;
                        new.push(*neighbor);
                    }
                },
                NeighborState::Old => {
                    old.push(*neighbor);
                }
            }
        }
        TargetNeighbors{old, new}
    }

    fn reversed_targets(targets: &Vec<TargetNeighbors>) -> Vec<TargetNeighbors> {
        // FIXME: I shouldn't have to build this. Just add on to target neighbors in some
        // principled way?
        let capacity = targets[0].old.len();
        let mut reversed: Vec<_> = (0..targets.len())
            .map(|_| TargetNeighbors{old: Vec::with_capacity(capacity), new: Vec::with_capacity(capacity)})
            .collect();
        for (i, target) in targets.iter().enumerate() {
            for neighbor in target.old.iter() {
                reversed[neighbor.idx].old.push(NeighborData{
                    distance: neighbor.distance,
                    idx: i,
                    state: neighbor.state
                });
            }
            for neighbor in target.new.iter() {
                reversed[neighbor.idx].new.push(NeighborData{
                    distance: neighbor.distance,
                    idx: i,
                    state: neighbor.state
                });
            }
        }
        reversed
    }

    fn sample_from_other(&mut self, other: &TargetNeighbors, sample_rate: f64) {
        let mut rng = rand::thread_rng(); //FIXME: This should probably be threaded through the call
        for neighbor in other.old.iter() {
            if rng.gen_range(0., 1.) < sample_rate {
                self.old.push(*neighbor)
            }
        }
        for neighbor in other.new.iter() {
            if rng.gen_range(0., 1.) < sample_rate {
                self.new.push(*neighbor)
            }
        }
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

pub fn build_knn_approximate(points: &[LabeledPoint], k: usize, sample_rate: f64, precision: f64) 
    -> Graph<LabeledPoint, f64, petgraph::Undirected> {
    // https://www.cs.princeton.edu/cass/papers/www11.pdf
    // TODO: it's not entirely clear to me why i didn't just use petgraph for hte underlying
    // datastructure here

    let mut rng = rand::thread_rng();
    // This should be a vector of heaps, but rust's heap won't quite do it so
    println!("data structures building");
    let mut approximate_neighbors: Vec<Vec<NeighborData>> = (0..points.len())
        .map(|i| {
            let points = rejection_sample(k, points.len(), &mut rng);
            points.iter()
                .map(|&j| NeighborData{distance: f64::INFINITY, idx:j, state: NeighborState::New})
                .collect()
        })
        .collect();
    println!("data structures built");
    let mut done = false;
    let mut iters = 0;
    while !done {
        iters += 1;
        println!("1");
        // FIXME: I'm making targets be a vec of hashsets
        // because redudnant targets is a very real possibility and i need an O(1) solution to it
        let mut targets: Vec<HashSet<NeighborData>> = approximate_neighbors.iter_mut()
            .map(|neighbors| sample_neighbors(neighbors, sample_rate))
            .collect();
        println!("2");
        let reversed_targets = add_reversed_targets(&mut targets);
        println!("4");

        let mut counter = 0;

        for targ in targets.iter() {
            for (i, new_target) in targ.iter().enumerate() {
                for (j, other_new) in targ.iter().enumerate() {
                    // FIXME: This is where you stopped
                    // We need to check that either it's or new-old or new-new and j > i 
                    // and some of that check needs to happen outside of this inner loop
                    let distance = points[new_target.idx].distance(&points[other_new.idx]);
                    let changed = update_neighbors(&mut approximate_neighbors[new_target.idx], new_target.idx, other_new.idx, distance, k);
                    if changed {counter += 1};
                    let changed = update_neighbors(&mut approximate_neighbors[other_new.idx], other_new.idx, new_target.idx, distance, k);
                    if changed {counter += 1};
                }
                for other_old in targ.old.iter(){
                    let distance = points[new_target.idx].distance(&points[other_old.idx]);
                    let changed = update_neighbors(&mut approximate_neighbors[new_target.idx], new_target.idx, other_old.idx, distance, k);
                    if changed {counter += 1};
                    let changed = update_neighbors(&mut approximate_neighbors[other_old.idx], other_old.idx, new_target.idx, distance, k);
                    if changed {counter += 1};
                }
            }
        }
        println!("5");

        //println!("-------");
        //println!("{:?}", approximate_neighbors);
        println!("{} < {}", counter, (precision * points.len() as f64 * k as f64) as i64);
        done = counter <= (precision * points.len() as f64 * k as f64) as i64;
        if iters > 2000 {
            // FIXME: This should probably be more graceful than full-on panicking 
            panic!();
        }
    }

    graph_from_neighbordata(points, approximate_neighbors)
}

fn graph_from_neighbordata(points: &[LabeledPoint], neighbors: Vec<Vec<NeighborData>>)
    -> Graph<LabeledPoint, f64, petgraph::Undirected> {
    let mut neighbor_graph = Graph::new_undirected();
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
