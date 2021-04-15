use itertools::Itertools;
use itertools::FoldWhile::{Continue, Done};
use super::simplex::{Simplex, CNS, SimplexConverter};
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

    fn remap_rows(&mut self, index_list: &[usize]) {
        let new_columns: HashMap<usize, HashSet<usize>> = self.columns.iter()
            .map(|(&i, col)| {
                let new_col: HashSet<usize> = col.iter()
                    .map(|&row| index_list[row])
                    .collect();
                (i, new_col)
            })
            .collect();
        self.columns = new_columns;
    }
}

struct InterimPersistenceResult {
    transformation: TransformationMatrix,
    pairs: Vec<(usize, usize)>,
    essentials: Vec<usize>
}



fn apply_transformation(cofaces: &mut HashSet<usize>, face_simplices: &[RichSimplex],
                        coface_indices: &HashMap<CNS, usize>, converter: &SimplexConverter,
                        transformation: &mut TransformationMatrix, target_column: usize, pivot_owner: usize) {
    let current_transformation = transformation.get_column(pivot_owner);
    for i in 0..pivot_owner { // Could instead check all values in transformation, esp if they were ordered somehow
        if current_transformation.contains(&i) {
            for coface in face_simplices[i].cofacets(converter) {
                match coface_indices.get(&coface.simplex){
                    None => continue,
                    Some(coface_index) => {
                        if !cofaces.remove(coface_index) {
                            cofaces.insert(*coface_index);
                        }
                    }
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

fn find_pairs_with_clearing(faces: &[RichSimplex], cofaces: &[RichSimplex], face_pairs: &[(usize, usize)],
                            face_columns: &HashMap<usize, Vec<usize>>, converter: &SimplexConverter) -> InterimPersistenceResult {
    let pivots: HashSet<usize> = face_pairs.iter().map(|(f, c)| *c).collect();
    let mut index_lookup: Vec<usize> = Vec::with_capacity(faces.len() - pivots.len());
    // TODO: Maybe don't need to copy here?
    let mut filtered_faces: Vec<RichSimplex> = Vec::with_capacity(faces.len() - pivots.len());
    for (i, s) in faces.iter().enumerate() {
        // NOTE/TODO:deviating from the julia code here, but i think the julia code was wrong?
        if pivots.contains(&i) {
            continue;
        }
        index_lookup.push(i);
        filtered_faces.push(s.clone());
    }

    let mut result = find_persistent_pairs(&filtered_faces, cofaces, converter);

    // Now we need to fix our indexing to correspond to the full faces
    result.transformation.remap_rows(&index_lookup);
    result.pairs.iter_mut().for_each(|pair| pair.0 = index_lookup[pair.0]);
    for i in 0..result.essentials.len() {
        result.essentials[i] = index_lookup[result.essentials[i]];
    }

    // TODO: i don't remember why we do this
    for (f, c) in face_pairs {
        result.transformation.columns.insert(*c, face_columns[f].iter().copied().collect());
    }

    result
}

fn compute_barcodes(complex: &[RichSimplex], converter: &SimplexConverter) -> Vec<Vec<(f64, f64)>> {
    let max_dimension = complex.iter().map(|s| usize::from(s.dimension)).max().unwrap();
    let mut lifetimes: Vec<Vec<(f64, f64)>> = Vec::with_capacity(max_dimension);
    let mut dimension_pairs: Vec<Vec<(usize, usize)>> = Vec::with_capacity(max_dimension);
    let mut dimension_essentials: Vec<Vec<usize>> = Vec::with_capacity(max_dimension);
    let mut dimension_partitions: Vec<Vec<RichSimplex>> = (0..max_dimension).map(|_| {
            //FIXME: this capacity is pretty off
            Vec::with_capacity(complex.len() / max_dimension)
        }).collect();

    // iterate in reverse because ripser uses cohomology, so we need to go backwards
    // (but all the other machinery is built for homology, so to go backwards we need to flip
    // everything around)
    for s in complex.iter().rev() {
        dimension_partitions[usize::from(s.dimension)].push(s.clone());
    }


    // H0 has to be calculated normally 
    // TODO: it can be done faster than the general homology algorithm

    let InterimPersistenceResult {transformation, pairs, essentials} = find_persistent_pairs(&dimension_partitions[0], &dimension_partitions[1], converter);
    dimension_pairs.push(pairs);
    dimension_essentials.push(essentials);
    let mut current_transformation = transformation;

    for d in 2..max_dimension {
        let subfaces = &dimension_partitions[d-2];
        let faces = &dimension_partitions[d-1];
        let cofaces = &dimension_partitions[d];
        let pairs = &dimension_pairs[d-2];

        let available_faces: HashSet<CNS> = faces.iter().map(|f| f.simplex).collect();
        let mut face_columns: HashMap<usize, Vec<usize>> = HashMap::with_capacity(10); //FIXME again, more capacity things
        for (s, _f) in pairs.iter() {
            // TODO: Split this into a function?
            let selected = current_transformation.columns.get(&s).unwrap(); // FIXME why s instead of f?
            let mut reduced: HashSet<CNS> = HashSet::with_capacity(10); // FIXME: I can definitely know this better a priori

            // TODO: This was the moment that I realized the columns might need to be ordered
            for row_idx in selected.iter().filter(|&&row| row < *s) {
                // NOTE: dropping a conditional in julia that i think was never satisfied
                for face_simplex in subfaces[*row_idx].cofacets(converter) {
                    if !available_faces.contains(&face_simplex.simplex) {
                        continue
                    }
                    if !reduced.remove(&face_simplex.simplex) {
                        reduced.insert(face_simplex.simplex);
                    }
                }
            }
            let filtered_column: Vec<usize> = faces.iter().enumerate()
                .filter_map(|(i, f)| {
                    if reduced.contains(&f.simplex) {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();
            face_columns.insert(*s, filtered_column);
        }
        let InterimPersistenceResult {transformation, pairs, essentials} = find_pairs_with_clearing(&faces, &cofaces, &pairs, &face_columns, converter);
        current_transformation = transformation;
        lifetimes.push(pairs.iter().map(|(b, d)| (faces[*b].lifetime, cofaces[*d].lifetime)).collect());
        dimension_pairs.push(pairs);
        dimension_essentials.push(essentials);
    }

    // TODO: how does the julia code do the barcodes?
    lifetimes
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
