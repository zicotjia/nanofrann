// DistType = f64
// IndexType = usize

// Version 3: unsafe rust

// Version 2: use array
// struct KNNResultSet<const N: usize> {
//     indices: [usize; N],
//     dists: [f64; N], // dists might still need Vec for dynamic resizing
//     capacity: usize,
//     count: usize,
// }
//
// impl<const N: usize> crate::KNNResultSet<N> {
//     fn new(capacity: usize) -> Self {
//         Self {
//             indices: [0; N],  // Initialized to 0
//             dists: [f64::MAX; N],
//             capacity,
//             count: 0,
//         }
//     }
//
//     fn size(&self) -> usize {
//         self.count
//     }
//
//     fn is_full(&self) -> bool {
//         self.count == self.capacity
//     }
//
//     fn add_point(dist: f64, index: usize) {
//
//     }
// }

// Version 1: used vectors
struct KNNResultSet {
    indices: Vec<usize>,
    dists: Vec<f64>,
    capacity: usize,
    count: usize,
}

impl KNNResultSet {

    // Since size is fixed, we can just offload all the allocation work in the initialization
    // Remove overhead of calling Vec::push()
    fn new_with_capacity(capacity: usize) -> Self {
        unsafe {
            // Fastest way to allocate without default value that i know of
            let indices = Vec::from_raw_parts(
                std::alloc::alloc(std::alloc::Layout::array::<usize>(capacity).unwrap()) as *mut usize,
                capacity,
                capacity,
            );
            let dists = Vec::from_raw_parts(
                std::alloc::alloc(std::alloc::Layout::array::<f64>(capacity).unwrap()) as *mut f64,
                capacity,
                capacity,
            );
            Self {
                indices,
                dists,
                capacity,
                count: 0,
            }
        }
    }

    fn size(&self) -> usize {
        self.count
    }

    fn is_full(&self) -> bool {
        self.count == self.capacity
    }

    fn add_point(&mut self, dist: f64, index: usize) {
        let mut i = self.count;
        while i > 0 {
            if self.dists[i - 1] > dist {
                if i < self.capacity {
                    self.dists[i] = self.dists[i - 1];
                    self.indices[i] = self.indices[i - 1];
                }
            } else {
                break;
            }
            i -= 1;
        }
        if i < self.capacity {
            self.dists[i] = dist;
            self.indices[i] = index;
        }
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    fn worst_dist(&self) -> f64 {
        self.dists[self.capacity - 1]
    }
}

struct RadiusResultSet {
    radius: f64,
    indices_dists: Vec<(usize, f64)>, // (Index, Distance)
}

impl RadiusResultSet {
    pub fn new_with_radius(radius: f64) -> Self {
        Self {
            radius,
            indices_dists: vec![]
        }
    }

    pub fn clear(&mut self) {
        self.indices_dists.clear();
    }

    pub fn add_point(&mut self, dist: f64, index: usize) {
        if dist < self.radius { self.indices_dists.push((index, dist)) };
    }

    pub fn set_radius_and_clear(&mut self, radius: f64) {
        self.radius = radius;
        self.clear();
    }

    pub fn worst_item(&self) -> (usize, f64) {
        assert!(self.indices_dists.len() > 0);
        self.indices_dists.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .copied().unwrap() // To return an owned (usize, f64) instead of a reference
    }
}


// Equivalent to IndexDist_Sorter in C++ nanoflann. If same dist, return the earliest index
fn sort_by_index_distance<T: Ord, U: PartialOrd>(vec: &mut Vec<(T, U)>) {
    vec.sort_by(|a, b| {
        match a.1.partial_cmp(&b.1) {
            Some(std::cmp::Ordering::Equal) => a.0.cmp(&b.0),
            Some(ordering) => ordering,
            None => std::cmp::Ordering::Greater, // Treat NaN as larger
        }
    });
}

fn main() {
    let mut result = KNNResultSet::new_with_capacity(10);
    println!("{:?}", result.indices);
    result.add_point(10.0, 10);
    result.add_point(11.0, 11);
    result.add_point(1.0, 12);
    println!("{:?}", result.indices);
    println!("{:?}", result.dists);
}

#[cfg(test)]
mod knnresult_set_tests {
    use approx::relative_eq;
    use super::*;

    #[test]
    fn test_results_new() {
        let sut = KNNResultSet::new_with_capacity(10);
        assert_eq!(sut.indices.len(), 10);
        assert_eq!(sut.dists.len(), 10);
    }

    #[test]
    fn test_add_point() {
        let mut sut = KNNResultSet::new_with_capacity(10);
        sut.add_point(1.0, 1);
        sut.add_point(3.0, 2);
        sut.add_point(2.0, 3);
        sut.add_point(5.0, 4);
        sut.add_point(4.0, 5);
        sut.add_point(7.0, 6);
        sut.add_point(6.0, 7);
        sut.add_point(8.0, 8);
        sut.add_point(9.0, 9);
        sut.add_point(10.0, 10);
        assert_eq!(sut.indices, [1, 3, 2, 5, 4, 7, 6, 8, 9, 10])
    }

    #[test]
    fn test_worst_dist() {
        let mut sut = KNNResultSet::new_with_capacity(10);
        sut.add_point(1.0, 1);
        sut.add_point(2.0, 2);
        sut.add_point(3.0, 3);
        sut.add_point(4.0, 4);
        sut.add_point(5.0, 5);
        sut.add_point(6.0, 6);
        sut.add_point(7.0, 7);
        sut.add_point(8.0, 8);
        sut.add_point(9.0, 9);
        sut.add_point(10.0, 10);
        relative_eq!(sut.worst_dist(),  10.0);
    }
}

#[cfg(test)]
mod radius_result_set_tests {
    use approx::relative_eq;
    use super::*;

    #[test]
    fn test_results_new() {
        let sut = RadiusResultSet::new_with_radius(10.0);
        assert_eq!(sut.indices_dists, []);
        relative_eq!(sut.radius, 10.0);
    }

    #[test]
    fn test_add_point() {
        let mut sut = RadiusResultSet::new_with_radius(5.0);
        sut.add_point(1.0, 1);
        sut.add_point(2.0, 2);
        sut.add_point(3.0, 3);
        sut.add_point(4.0, 4);
        sut.add_point(5.0, 5);
        sut.add_point(6.0, 6);
        relative_eq!(sut.indices_dists[0].1, 1.0);
        relative_eq!(sut.indices_dists[1].1, 2.0);
        relative_eq!(sut.indices_dists[2].1, 3.0);
        relative_eq!(sut.indices_dists[3].1, 4.0);
        assert_eq!(sut.indices_dists.len(), 4);
    }

    #[test]
    fn test_sort() {
        let mut sut = RadiusResultSet::new_with_radius(5.0);
        sut.add_point(2.0, 1);
        sut.add_point(1.0, 2);
        sut.add_point(2.0, 3);
        sut.add_point(3.0, 4);
        sort_by_index_distance::<usize, f64>(&mut sut.indices_dists);
        assert_eq!(sut.indices_dists[0].0, 2);
        assert_eq!(sut.indices_dists[1].0, 1);
        assert_eq!(sut.indices_dists[2].0, 3);
        assert_eq!(sut.indices_dists[3].0, 4);
        relative_eq!(sut.indices_dists[0].1, 1.0);
        relative_eq!(sut.indices_dists[1].1, 2.0);
        relative_eq!(sut.indices_dists[2].1, 2.0);
        relative_eq!(sut.indices_dists[3].1, 3.0);
    }

    #[test]
    fn test_clear() {
        let mut sut = RadiusResultSet::new_with_radius(5.0);
        sut.add_point(1.0, 1);
        sut.add_point(2.0, 2);
        sut.add_point(3.0, 3);
        sut.add_point(4.0, 4);
        sut.clear();
        assert_eq!(sut.indices_dists.len(), 0);
    }

    #[test]
    fn test_radius_and_clear() {
        let mut sut = RadiusResultSet::new_with_radius(5.0);
        sut.add_point(1.0, 1);
        sut.add_point(2.0, 2);
        sut.add_point(3.0, 3);
        sut.add_point(4.0, 4);
        sut.set_radius_and_clear(10.0);
        assert_eq!(sut.indices_dists.len(), 0);
        relative_eq!(sut.radius, 10.0);
    }

    #[test]
    fn test_worst_item() {
        let mut sut = RadiusResultSet::new_with_radius(11.0);
        sut.add_point(1.0, 1);
        sut.add_point(10.0, 2);
        sut.add_point(2.0, 3);
        sut.add_point(5.0, 4);
        sut.add_point(4.0, 5);
        assert_eq!(sut.worst_item().0, 2);
        relative_eq!(sut.worst_item().1, 10.0);
    }
}