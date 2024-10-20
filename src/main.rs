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

struct SearchParams {
    checks: usize,
    eps: f64,
    sorted: bool,
}

impl SearchParams {
    pub fn new() -> Self {
        Self {
            checks: 32,
            eps: 0.0,
            sorted: true,
        }
    }
}

struct KDTree {
    vind: Vec<usize>,
    leaf_size: usize,
}

struct Point {
    x: f64,
    y: f64,
    z: f64,
}

// Hard code a data source for the KDTree
struct DataSource {
    vec: Vec<Point>
}

impl DataSource {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            vec: Vec::with_capacity(capacity)
        }
    }
    pub fn add_point(&mut self, x: f64, y: f64, z: f64) {
        self.vec.push(Point { x, y, z });
    }
    pub fn size(&self) -> usize {
        self.vec.len()
    }

    pub fn get_point_count(&self) -> usize {
        self.vec.len()
    }

    pub fn get_point(&self, index: usize, dim: usize) -> f64 {
        match dim {
            0 => self.vec[index].x,
            1 => self.vec[index].y,
            2 => self.vec[index].z,
            _ => panic!("Invalid dimension")
        }
    }
}

struct KDTreeSingleIndexParams {
    leaf_max_size: usize,
}

impl KDTreeSingleIndexParams {
    pub fn new() -> Self {
        Self {
            leaf_max_size: 10
        }
    }
}

pub struct Node {
    node_type: NodeType,         // Replaces the union
    child1: Option<Box<Node>>,   // Replaces Node* child1
    child2: Option<Box<Node>>,   // Replaces Node* child2
}

pub enum NodeType {
    Leaf {
        left: usize,
        right: usize,
    },
    NonLeaf {
        div_feat: i32,      // Dimension used for subdivision
        div_low: f64,      // The low value for subdivision
        div_high: f64,     // The high value for subdivision
    },
}

struct Interval {
    low: f64,
    high: f64,
}

struct BoundingBox {
    bounds: Vec<Interval>,
}

impl BoundingBox {
    pub fn new(dim: usize) -> Self {
        let mut bounds = Vec::with_capacity(dim);
        for _ in 0..dim {
            bounds.push(Interval { low: 0.0, high: 0.0 });
        }
        Self { bounds }
    }
}

struct KDTreeSingleIndex {
    // Indices to points in the dataset
    vind: Vec<usize>,
    leaf_size: usize,
    dataset: DataSource,
    // nanoflann use a pointer to the root node
    root: Option<Box<Node>>,
    size: usize,
    size_at_index_build: usize,
    dim: usize,
    root_BBox: BoundingBox,
}

impl KDTreeSingleIndex {
    pub fn new(dataset: DataSource, params: KDTreeSingleIndexParams) -> Self {
        let dim = 3;
        let size = dataset.size();
        let mut vind = Vec::with_capacity(size);
        for i in 0..size {
            vind.push(i);
        }
        let root = None;
        let size_at_index_build = size;
        let mut tree = Self {
            vind,
            leaf_size: params.leaf_max_size,
            dataset,
            root,
            size,
            size_at_index_build,
            dim,
            root_BBox: BoundingBox::new(dim),
        };
        tree.init_vind();
        tree
    }

    pub fn init_vind(&mut self) {
        self.vind.clear();
        // Can be optimized to use unsafe code
        self.vind.resize(self.size, 0);
        for i in 0..self.size {
            self.vind.push(i);
        }
    }

    pub fn compute_bounding_box(&self, bounding_box: BoundingBox) {
        let N = self.size;
        for i in 0..self.dim {
            bounding_box.bounds[i].low = self.dataset.get_point(0, i);
            bounding_box.bounds[i].high = self.dataset.get_point(0, i);
        }
        for k in 1..N {
            for i in 0..self.dim {
                let val = self.dataset.get_point(k, i);
                if val < bounding_box.bounds[i].low {
                    bounding_box.bounds[i].low = val;
                }
                if val > bounding_box.bounds[i].high {
                    bounding_box.bounds[i].high = val;
                }
            }
        }
    }

    // Return index of new root
    pub fn divide_tree(&self, left: usize, right: usize, bounding_box: &BoundingBox) -> Node {
        let mut node = Node {
            node_type: NodeType::NonLeaf {
                div_feat: 0,
                div_low: 0.0,
                div_high: 0.0,
            },
            child1: None,
            child2: None,
        };
        if (right - left) <= self.leaf_size {
            // Explicitly give the node a Leaf type
            node.child1 = Some(Box::new(Node {
                node_type: NodeType::Leaf { left, right },
                child1: None,
                child2: None,
            }));
            for i in 0..self.dim {
                let val = self.dataset.get_point(self.vind[left], i);
                bounding_box.bounds[i].low = val;
                bounding_box.bounds[i].high = val;
            }
            for k in (left + 1)..right {
                for i in 0..self.dim {
                    let val = self.dataset.get_point(self.vind[k], i);
                    if val < bounding_box.bounds[i].low {
                        bounding_box.bounds[i].low = val;
                    }
                    if val > bounding_box.bounds[i].high {
                        bounding_box.bounds[i].high = val;
                    }
                }
            }
        } else {
            let index;
            let cut_feat;
            let cut_val;
        }
        node
    }

    fn middle_split(&self, ind: usize, count: usize, index: usize,
                    cut_feat: &mut i32, cut_val: &mut f64, bounding_box: &BoundingBox) {
        let eps = 1e-5;
        let mut max_span = bounding_box.bounds[0].high - bounding_box.bounds[0].low;
        for i in 1..self.dim {
            let span = bounding_box.bounds[i].high - bounding_box.bounds[i].low;
            if span > max_span {
                max_span = span;
            }
        }
        let mut max_spread = -1.0;
        *cut_feat = 0;
        for i in 0..self.dim {
            let span = bounding_box.bounds[i].high - bounding_box.bounds[i].low;
            if span > (1.0 - eps) * max_span {
                let mut min_element: f64 = 0.0;
                let mut max_element: f64 = 0.0;
                self.compute_min_max(ind, count, i, &mut min_element, &mut max_element);
                let spread = max_element - min_element;
                if spread > max_spread {
                    max_spread = spread;
                    *cut_feat = i as i32;
                }
            }
        }
        let split_val = 0.5 * (bounding_box.bounds[*cut_feat as usize].low + bounding_box.bounds[*cut_feat as usize].high);
        let mut min_elememt: f64 = 0.0;
        let mut max_element: f64 = 0.0;
        self.compute_min_max(ind, count, *cut_feat as usize, &mut min_elememt, &mut max_element);

        if split_val < min_elememt {
            *cut_val = min_elememt;
        } else if split_val > max_element {
            *cut_val = max_element;
        } else {
            *cut_val = split_val;
        }

        let lim1: usize;
        let lim2: usize;
    }

    // Element is equivalent to dim
    fn compute_min_max(&self, ind: usize, count: usize, cut_feat: usize, min_element: &mut f64, max_element: &mut f64) {
        *min_element = self.dataset.get_point(self.vind[ind], cut_feat);
        *max_element = self.dataset.get_point(self.vind[ind], cut_feat);
        for i in 1..count {
            let val = self.dataset.get_point(self.vind[ind + i], cut_feat);
            if val < *min_element {
                *min_element = val;
            }
            if val > *max_element {
                *max_element = val;
            }
        }
    }

    /**
    *  Subdivide the list of points by a plane perpendicular on axe corresponding
    *  to the 'cutfeat' dimension at 'cutval' position.
    *
    *  On return:
    *  dataset[ind[0..lim1-1]][cutfeat]<cutval
    *  dataset[ind[lim1..lim2-1]][cutfeat]==cutval
    *  dataset[ind[lim2..count]][cutfeat]>cutval
    */
    fn plane_split(&self, ind: usize, count: usize, cut_feat: usize, cut_val: &mut f64, lim1: &mut usize, lim2: &mut usize) {
        let mut left = 0;
        let mut right = count - 1;
        // This is a variation of the Dutch National Flag problem
        loop {
            while left <= right && self.dataset.get_point(self.vind[ind + left], cut_feat) < *cut_val {
                left += 1;
            }
            while left <= right && self.dataset.get_point(self.vind[ind + right], cut_feat) >= *cut_val {
                right -= 1;
            }
            if left > right  {
                break;
            }
            let temp = self.vind[ind + left];
            self.vind[ind + left] = self.vind[ind + right];
            self.vind[ind + right] = temp;
            left += 1;
            right -= 1;
        }
        // By this point, all points less than cut_val are on the left side of
        *lim1 = left;
        right = count - 1;
        loop {
            while left <= right && self.dataset.get_point(self.vind[ind + left], cut_feat) <= *cut_val {
                left += 1;
            }
            while left <= right && self.dataset.get_point(self.vind[ind + right], cut_feat) > *cut_val {
                right -= 1;
            }
            if left > right {
                break;
            }
            let temp = self.vind[ind + left];
            self.vind[ind + left] = self.vind[ind + right];
            self.vind[ind + right] = temp;
            left += 1;
            right -= 1;
        }
    }

    fn dataset_get(&self, index: usize, dim: usize) -> f64 {
        self.dataset.get_point(index, dim)
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