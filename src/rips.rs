use super::simplex::{Simplex, Dimension, CNS, SimplexConverter};
use std::collections::{HashMap, HashSet};
use itertools::Itertools;


// FIXME: notation issue, i say lifetime when birthtime is more accurate

#[derive(Debug)]
struct RichSimplex {
    simplex: CNS,
    dimension: Dimension,
    lifetime: f64
}

impl RichSimplex {
    fn from_vertices(vertices: &[usize], lifetime: f64, converter: SimplexConverter) -> RichSimplex {
        let simplex = converter.simplex_to_cns(&Simplex::construct_simplex(vertices, lifetime));
        let dimension = Dimension::from(vertices.len());
        RichSimplex{simplex, dimension, lifetime}
    }
}

fn find_all_cofaces(vertices: &[usize], lifetime: f64, converter: SimplexConverter, max_dim: Dimension,
                       neighbors: HashMap<usize, HashSet<usize>>) -> Vec<RichSimplex> {
    // TODO: throttle this according to max dim
    let mut previous_neighbors: Option<HashSet<usize>> = None;
    loop {
        let common_neighbors: HashSet<usize> = vertices.iter()
            .filter_map(|v| neighbors.get(v))
            .fold(None, |acc, x| {
                match acc {
                    None => Some(x.clone()),
                    Some(candidates) => Some(candidates.intersection(x).copied().collect())
                }
            })
            .unwrap_or_else(HashSet::new);
        if let Some(previous) = previous_neighbors{
            if previous == common_neighbors {
                break;
            }
        }
        previous_neighbors = Some(common_neighbors.clone());
    }
    match previous_neighbors {
        None => vec![], // FIXME: should this ever happen? or is it an error?
        Some(clique) => {
            let original_set: HashSet<usize> = vertices.iter().copied().collect();
            let neighborhood = clique.difference(&original_set);
            (0..usize::from(max_dim)-vertices.len()).map(|k| {
                neighborhood.permutations(k)
                    .map(|ns| {
                        let mut face_vertices: Vec<usize> = ns.into_iter().copied().collect();
                        face_vertices.extend(vertices);
                        RichSimplex::from_vertices(&face_vertices, lifetime, converter)
                    })
                    .collect::<Vec<RichSimplex>>()
            })
            .flatten()
            .collect()
        }
    }
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
    let converter = SimplexConverter::construct_for_vertex_count_and_dim(distances.len(), usize::from(max_dim));

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

    let neighbor_lookup: HashMap<usize, HashSet<usize>> = simplices.iter()
        .map(|s| {
            (usize::from(s.simplex), HashSet::with_capacity(distances.len()))
        })
        .collect();

    for (distance, row, col) in labeled_indices {
        // the (row, col) edge has been created
        neighbor_lookup[&row].insert(col);
        neighbor_lookup[&col].insert(row);
        let rich_simplex = RichSimplex::from_vertices(&[col, row], distance, converter);
        simplices.push(rich_simplex);


        // now deal with the higher-order simplices
        simplices.extend(find_all_cofaces(&[col, row], distance, converter, max_dim, neighbor_lookup));
    }
    simplices
}
