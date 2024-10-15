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
mod tests {
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
        sut.add_point(3.0, 2);
        sut.add_point(2.0, 3);
        sut.add_point(5.0, 4);
        sut.add_point(4.0, 5);
        sut.add_point(7.0, 6);
        sut.add_point(6.0, 7);
        sut.add_point(8.0, 8);
        sut.add_point(9.0, 9);
        sut.add_point(10.0, 10);
        relative_eq!(sut.worst_dist(),  10.0);
    }
}