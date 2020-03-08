
struct HomologyData {
    simplex: Simplex,
    marked: bool,
    value: Option<usize>,
    assignment: Simplex
}

fn compute_intervals(data: &mut [HomologyData]) {
    //https://geometry.stanford.edu/papers/zc-cph-05/zc-cph-05.pdf
    let mut lifetimes = [];
    for i in 0..data.len() {
        match remove_pivot_rows(data[i].simplex, data) {
            None => data[i].marked = true,
            Some(boundary) => {
                let boundary_idx = boundary.max_index;
                let k = data[boundary_idx].simplex.dimension;
                data[boundary_idx].value = Some(i);
                data[boundary_idx].assignment = Some(boundary);
                // FIXME: obviously this won't work, nothing is in lifetimes
                lifetimes[k].insert((data[boundary_idx].simplex.lifetime, data[i].simplex.lifetime));
            }
        }
    }
    for simplex in data {
        if simplex.marked && simplex.value.is_none() {
            let k = simplex.dimension; // FIXME: lifetime is the wrong word. more like "appearancetime"
            lifetimes[k].insert((simplex.lifetime, usize::max_value())); //maybe i need an enum here to handle infinity
        }
    }
}


fn remove_pivot_rows(simplex: &Simplex, data: &[HomologyData]) {
    let k = simplex.dimension;
    let mut boundary = simplex.boundary(k);
    boundary.remove_unmarked(data);
    while boundary.len() > 0 {
        let i = boundary.max_index(data);
        match data[i].value {
            None => break;
            Some(value) => {
                // FIXME: this is the part i definitely do not understand
                let q = SOMETHING;
                d = d - q.inverse() * data[i].value();
            }
        }
    }
    return d;
}


