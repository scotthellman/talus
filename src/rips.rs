use super::simplex::{Simplex, Dimension, CNS, SimplexConverter};
use std::collections::{HashMap, HashSet};


// FIXME: notation issue, i say lifetime when birthtime is more accurate

#[derive(Clone, Copy, Debug, PartialEq)]
struct RichSimplex {
    simplex: CNS,
    dimension: Dimension,
    lifetime: f64
}

fn construct_rich_simplex(vertices: &[usize], lifetime: f64, converter: SimplexConverter) -> RichSimplex {
    let simplex = converter.simplex_to_cns(Simplex::construct_simplex(vertices, lifetime));
    let dimension = Dimension::from(vertices.len());
    RichSimplex{simplex, dimension, lifetime}
}

fn higher_order_edge_creation(vertices: &[usize], lifetime: f64, converter: SimplexConverter,
                              neighbors: HashMap<CNS, HashSet<CNS>>) -> Vec<RichSimplex> {
    // TODO: throttle this according to max dim
    let mut previous_neighbors = None;
    loop {
        let mut common_neighbors: HashSet<CNS> = vertices.iter()
            .map(|v| neighbors[v])
            .fold(None, |acc, x| {
                match acc {
                    None => x.clone(),
                    Some(candidates) => candidates.intersection(x)
                }
            })
            .collect();
        if let Some(previous) = previous_neighbors{
            if previous == common_neighbors {
                break;
            }
        }
        previous_neighbors = Some(common_neighbors.clone());
    }
    // TODO: filter those down to all subfaces
}

fn rips(distances: Vec<Vec<f64>>, max_dim: Dimension) -> Vec<RichSimplex> {
    // TODO: there's a faster algorithm out there

    // We're going to iterate over ever cell in distances in ascending distance order
    let mut labeled_indices: Vec<(f64, usize, usize)> = distances.iter().enumerate()
        .map(|(i, row)| {
            row.iter().enumerate()
               .skip(i+1)
               .map(|(j, &val)| {
                   (val, i, j)
               })
        })
        .flatten()
        .collect();
    labeled_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // First we need a converter to do simplex->cns conversion
    let converter = SimplexConverter(distances.len(), max_dim);

    // build 0-dimensional simplices
    let mut simplices: Vec<RichSimplex> = (0..distances.len())
        .map(|i| {
            let simplex = CNS::from(i);
            let dimension = Dimension::from(0);
            let lifetime = 0.0;
            RichSimplex{simplex, dimension, lifetime}
        })
        .collect();

    // General idea: 
    // add one-simplices in ascending distance order
    // a one simplex connects two zero simplices
    // if both of those simplices are connected to other vertices,
    // higher-order simplices are created
    // So we need to be able to quickly track neighbors of each vertex
    // (One could say a graph library could do this for us, but that feels heavyweight)

    let neighbor_lookup: HashMap<CNS, HashSet<CNS>> = simplices.iter()
        .map(|s| {
            (s.simplex, HashSet::with_capacity(distances.len()))
        })
        .collect();

    for (distance, row, col) in labeled_indices {
        // the (row, col) edge has been created
        neighbor_lookup[row].insert(col);
        neighbor_lookup[col].insert(row);
        let rich_simplex = construct_rich_simplex(&[col, row], distance, converter);
        simplices.push(rich_simplex);


        // now deal with the higher-order simplices

    }

    for distance in unique_distances {
        if distance == 0 {continue};

        let new_lines = distances.iter().enumerate()
            .map(|i, row| {
                row.iter().enumerate().skip(i+1).map {|j, pairwise_dist|
                    if pairwise_dist == distance {
                        construct_simplex(&[i, j+i+1], pairwise_distance, converter)
                    } else {
                        None
                    }
                }
            })
           .flatten()
           .filter_map()
           .collect();
    }

}
