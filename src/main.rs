use ndarray::prelude::*;
use itertools::Itertools;
use petgraph::graph::Graph;
use petgraph::dot::Dot;

fn pairwise_distance(points: ArrayView2<f64>) -> Array2<f64> {
    let mut pairwise = Array2::zeros((points.shape()[0], points.shape()[0]));
    for (i, row) in points.outer_iter().enumerate() {
        for (j, other) in points.slice(s![i.., ..]).outer_iter().enumerate() {
            let j = j+i;
            let distance = if i == j {
                0.
            } else {
                let diff = &row - &other;
                diff.dot(&diff).sqrt()
            };
            pairwise[[i,j]] = distance;
            pairwise[[j,i]] = distance;
        }
    }
    pairwise
}

fn build_knn(points: ArrayView2<f64>, k: usize) -> Graph<Array1<f64>, f64, petgraph::Undirected> {
    let mut neighbor_graph = Graph::new_undirected();
    let mut node_lookup = Vec::with_capacity(points.shape()[0]);
    for point in points.outer_iter() {
        let node = neighbor_graph.add_node(point.to_owned());
        node_lookup.push(node);
    }
    let pairwise = pairwise_distance(points);
    for (i, _) in points.outer_iter().enumerate() {
        pairwise.slice(s![i, ..]).into_iter().enumerate()
            .filter(|(j, _)| i != *j)
            .sorted_by(|(_, val), (_, other)| val.partial_cmp(other).unwrap())
            .take(k)
            .for_each(|(j, val)| {
                neighbor_graph.add_edge(node_lookup[i], node_lookup[j], *val);
            });
    }
    neighbor_graph
}

fn main() {
    let points = arr2(&[[1., 2.],
                  [3., 3.],
                  [0., 0.],
                  [-1., 1.]]);
    let pairwise = pairwise_distance(points.view());
    println!("Pairwise is {:?}", pairwise);
    let graph = build_knn(points.view(), 2);
    println!("Graph is {:?}", graph);
    println!("{:?}", Dot::with_config(&graph, &[]));
}
