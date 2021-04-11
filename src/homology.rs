use itertools::Itertools;
use itertools::FoldWhile::{Continue, Done};
use super::simplex::{Simplex, Dimension, CNS, SimplexConverter};
use std::collections::{HashSet, HashMap};
use super::rips::RichSimplex;

struct HomologyData {
    simplex: Simplex,
    marked: bool,
    value: Option<usize>,
    assignment: Simplex
}


struct TransformationMatrix {
    columns: HashMap<usize, HashSet<usize>>
}

impl TransformationMatrix {
    fn new(num_columns: usize) -> Self {
        let columns = HashMap::with_capacity(num_columns);
        TransformationMatrix{columns}
    }

    fn get_column(&mut self, col: usize) -> &HashSet<usize> {
        self.ensure_column_exists(col);
        self.columns.get(&col).unwrap()
    }

    fn ensure_column_exists(&mut self, col: usize) {
        if !self.columns.contains_key(&col){
            let mut new_col = HashSet::with_capacity(5); // FIXME better capacity
            new_col.insert(col);
            self.columns.insert(col, new_col);
        }
    }

    fn add_first_to_second(&mut self, col: usize, other: usize) {
        // TODO: check on how fast this is vs mutating
        self.ensure_column_exists(col);
        self.ensure_column_exists(other);

        let this_col = self.columns.get(&col).unwrap();
        let other_col = self.columns.get(&other).unwrap();


        let new_col: HashSet<usize> = this_col.symmetric_difference(other_col).copied().collect();
        self.columns.insert(other, new_col);
    }
}

struct InterimPersistenceResult {
    transformation: TransformationMatrix,
    pairs: Vec<(usize, usize)>,
    essentials: Vec<usize>
}


fn find_persistent_pairs(faces: &[RichSimplex], cofaces: &[RichSimplex],
                         converter: &SimplexConverter) -> InterimPersistenceResult {
    // TODO: better capacities
    let coface_indices: HashMap<CNS, usize> = cofaces.iter().enumerate().map(|(i, c)| (*&c.simplex, i)).collect();
    let mut pivots: HashMap<usize, usize> = HashMap::with_capacity(cofaces.len());
    let mut essentials: Vec<usize> = Vec::with_capacity(cofaces.len());
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(cofaces.len());
    let mut transformation: TransformationMatrix = TransformationMatrix::new(cofaces.len());
    for (i, face) in faces.iter().enumerate() {
        let mut pivot_result = find_pivot(face, cofaces, &coface_indices, &pivots, converter);
        if let Some(pivot) = pivot_result.pivot_index {
            pairs.push((i, pivot));
            pivots.insert(pivot, i);
            continue;
        }
        let mut pivot = pivot_result.coface_indices.iter().max().unwrap(); // FIXME: should probably be a heap or something
        while pivot_result.coface_indices.len() > 0 {
            match pivots.get(pivot) {
                None => {break},
                Some(owner) => {
                    apply_transformation(&mut pivot_result.coface_indices, faces, &coface_indices,
                                         converter, &mut transformation, i, *owner);
                }
            }
            pivot = pivot_result.coface_indices.iter().max().unwrap(); // FIXME: should probably be a heap or something, also the duplication isn't great
        }
        if pivot_result.coface_indices.len() == 0 {
            essentials.push(i);
        } else {
            pairs.push((i, *pivot));
        }
    }
    InterimPersistenceResult{transformation, pairs, essentials}
}

fn apply_transformation(cofaces: &mut HashSet<usize>, face_simplices: &[RichSimplex],
                        coface_indices: &HashMap<CNS, usize>, converter: &SimplexConverter,
                        transformation: &mut TransformationMatrix, target_column: usize, pivot_owner: usize) {
    let current_transformation = transformation.get_column(pivot_owner);
    for i in 0..pivot_owner { // Could instead check all values in transformation, esp if they were ordered somehow
        if current_transformation.contains(&i) {
            for coface in face_simplices[i].cofacets(converter) {
                let coface_index = coface_indices[&coface.simplex];
                if !cofaces.remove(&coface_index) {
                    cofaces.insert(coface_index);
                }
            }
        }
    }
    // face-th column of transformations needs to have the current transformation added to it
    transformation.add_first_to_second(pivot_owner, target_column);
}

struct PivotResult {
    pivot_index: Option<usize>,
    maximal: bool,
    coface_indices: HashSet<usize>
}


fn find_pivot(face: &RichSimplex, cofaces: &[RichSimplex], coface_indices: &HashMap<CNS, usize>,
              pivots: &HashMap<usize, usize>, converter: &SimplexConverter) -> PivotResult {

    face.cofacets(converter)
        .filter(|c| !coface_indices.contains_key(&c.simplex))
        .fold_while(PivotResult{pivot_index: None, maximal: false, coface_indices: HashSet::new()}, |mut acc, c| {
            let coface_index = *coface_indices.get(&c.simplex).unwrap();
            acc.coface_indices.insert(coface_index);
            if acc.maximal {
                return Continue(acc)
            } else {
                if cofaces[coface_index].lifetime == face.lifetime{
                    acc.maximal = true;
                    if !pivots.contains_key(&coface_index) {
                        acc.pivot_index = Some(coface_index);
                        return Done(acc)
                    }
                }
            }
            Continue(acc)
        }).into_inner()
}

/*
 * function find_pairs_with_clearing(face_simplices::Vector{Simplex}, coface_simplices::Vector{Simplex},
                                  face_pairs::Vector{Tuple{Int, Int}}, face_columns::Dict{Int, Vector{Int}},
                                  total_points::Int)
    # This is Algorithm 2
    # i know i'm doing coboundaries but the pairs are of the form (face, coface) still
    # of course in this case thats (subface, face)
    pivots = Set(j for (i,j) in face_pairs)
    original_index_lookup = Int[]
    filtered_face_simplices = Simplex[]
    for (i,s) in enumerate(face_simplices)
        if s in pivots
            continue
        end
        push!(original_index_lookup, i)
        push!(filtered_face_simplices, s)
    end

    face_V, coface_pairs, face_essentials = find_persistent_pairs(filtered_face_simplices, coface_simplices,
        total_points)

    # Lines 4-5 in Alg 2 are about fixing our indexing
    fixed_V = zeros(Int, length(face_simplices), length(face_simplices))
    fixed_V[original_index_lookup, original_index_lookup] = face_V
    fixed_pairs = [(original_index_lookup[f], c) for (f,c) in coface_pairs]
    fixed_essentials = [original_index_lookup[f] for f in face_essentials]

    for (s, f) in face_pairs
        fixed_V[:, f] = face_columns[s]
    end
    fixed_V, fixed_pairs, fixed_essentials
*/

fn find_pairs_with_clearing(faces: &[RichSimplex], cofaces: &[RichSimplex], face_pairs: &[(usize, usize)],
                            face_columns: &HashMap<usize, Vec<usize>>, converter: &SimplexConverter) -> PivotResult {
    let pivots: HashSet<usize> = face_pairs.iter().map(|(f, c)| *c).collect();
    let mut index_lookup: Vec<usize> = Vec::with_capacity(faces.len() - pivots.len());
    // TODO: Maybe don't need to copy here?
    let mut filtered_faces: Vec<RichSimplex> = Vec::with_capacity(faces.len() - pivots.len());
    for (i, s) in faces.iter().enumerate() {
        if pivots.contains(s) {
            continue;
        }
        index_lookup.push(i);
        filtered_faces.push(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::rips;
    use proptest::prelude::*;

    #[test]
    fn test_transformation_matrix() {
        let mut t = TransformationMatrix::new(3);

        // verify we start at I
        for i in 0..3 {
            let expected: HashSet<usize> = [i].iter().copied().collect();
            let actual = t.get_column(i);
            assert!(expected.is_subset(actual));
            assert!(expected.is_superset(actual));
        }

        // Add 2 to 1
        t.add_first_to_second(1,2);
        for i in 0..3 {
            let mut expected: HashSet<usize> = [i].iter().copied().collect();
            if i == 2 {
                expected.insert(1);
            }
            let actual = t.get_column(i);
            assert!(expected.is_subset(actual));
            assert!(expected.is_superset(actual));
        }

        // We're working mod 2 so add 2 to 1 again to get I
        t.add_first_to_second(1,2);
        for i in 0..3 {
            let expected: HashSet<usize> = [i].iter().copied().collect();
            let actual = t.get_column(i);
            assert!(expected.is_subset(actual));
            assert!(expected.is_superset(actual));
        }
    }
}
