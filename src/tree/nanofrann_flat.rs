// DistType = FloatType
// IndexType = usize

use std::ops::AddAssign;
use std::rc::Rc;
use num_traits::Float;
// Version 3: unsafe rust
use crate::common::common::*;
use crate::common::min_max::MinMax;

// Custom implementation. Use index to locate nodes.
// Index could point to two types of nodes: LeafNode or SplitNode
// Let say size is n. Then [0..n-1] is split node anything above n - 1 is leaf node
// i,e: for n = 100, index = 102 represent LeafNode[2]
#[derive(Clone, Debug)]
pub(crate) struct LeafNode {
    pub(crate) left: usize,
    pub(crate) right: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct SplitNode<DistType> {
    pub(crate) child1: usize,
    pub(crate) child2: usize,
    pub(crate) div_feat: i32,
    pub(crate) div_low: DistType,
    pub(crate) div_high: DistType,
}

#[derive(Debug)]
pub(crate) struct KDTreeSingleIndexFlat<'a, DistType> {
    // Indices to points in the dataset
    pub(crate) vind: Vec<usize>,
    pub(crate) leaf_nodes: Vec<LeafNode>,
    pub(crate) split_nodes: Vec<SplitNode<DistType>>,
    pub(crate) max_num_split_nodes: usize,
    pub(crate) leaf_size: usize,
    pub(crate) dataset: &'a DataSource<DistType>,
    pub(crate) root_index: usize,
    pub(crate) size: usize,
    pub(crate) size_at_index_build: usize,
    pub(crate) dim: usize,
    pub(crate) root_bounding_box: BoundingBox<DistType>,
}

impl<'a, FloatType> KDTreeSingleIndexFlat<'a, FloatType>
where FloatType: Float + MinMax + AddAssign + Copy {
    #[inline]
    pub fn new(dataset: &'a DataSource<FloatType>, params: KDTreeSingleIndexParams) -> Self {
        let size = dataset.size();
        let mut vind = Vec::with_capacity(size);
        unsafe {
            vind.set_len(size);
            for i in 0..size {
                vind[i] = i;
            }
        }
        let leaf_node_count = size.div_ceil(params.leaf_max_size);
        let split_node_count = leaf_node_count.next_power_of_two() * 3;
        let size_at_index_build = size;
        let mut tree = Self {
            vind,
            leaf_size: params.leaf_max_size,
            dataset,
            root_index: 0,
            leaf_nodes: Vec::with_capacity(leaf_node_count),
            split_nodes: Vec::with_capacity(split_node_count),
            max_num_split_nodes: split_node_count,
            size_at_index_build,
            dim: 3,
            root_bounding_box: BoundingBox::new(3),
            size,
        };
        tree
    }

    #[inline]
    pub fn build_index(&mut self) {
        self.compute_bounding_box();
        let mut bounding_box = self.root_bounding_box.clone();
        self.root_index = self.divide_tree(0, self.size, &mut bounding_box);
    }


    #[inline]
    pub fn compute_bounding_box(&mut self) {
        let size = self.size;
        for i in 0..self.dim {
            self.root_bounding_box.bounds[i].low = self.dataset.get_point(0, i);
            self.root_bounding_box.bounds[i].high = self.dataset.get_point(0, i);
        }
        for k in 1..size {
            for i in 0..self.dim {
                let val = self.dataset.get_point(k, i);
                if val < self.root_bounding_box.bounds[i].low {
                    self.root_bounding_box.bounds[i].low = val;
                }
                if val > self.root_bounding_box.bounds[i].high {
                    self.root_bounding_box.bounds[i].high = val;
                }
            }
        }
    }

    #[inline]
    pub fn divide_tree(&mut self, left: usize, right: usize, bounding_box: &mut BoundingBox<FloatType>) -> usize {
        let index;
        if (right - left) <= self.leaf_size {
            let node = LeafNode { left, right };
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
            index = self.leaf_nodes.len() + self.max_num_split_nodes;
            self.leaf_nodes.push(node);
        } else {
            let mut mid = 0;
            let mut cut_feat: usize = 0;
            let mut cut_val = FloatType::zero();
            self.middle_split(left, right - left, &mut mid, &mut cut_feat, &mut cut_val, bounding_box);

            let mut left_bounding_box = bounding_box.clone();
            left_bounding_box.bounds[cut_feat].high = cut_val;
            let child1 = self.divide_tree(left, left + mid, &mut left_bounding_box);

            let mut right_bounding_box = bounding_box.clone();
            right_bounding_box.bounds[cut_feat].low = cut_val;
            let child2 = self.divide_tree(left + mid, right, &mut right_bounding_box);

            for i in 0..self.dim {
                bounding_box.bounds[i].low = left_bounding_box.bounds[i].low.min(right_bounding_box.bounds[i].low);
                bounding_box.bounds[i].high = left_bounding_box.bounds[i].high.max(right_bounding_box.bounds[i].high);
            }

            let node = SplitNode {
                child1,
                child2,
                div_feat: cut_feat as i32,
                div_low: left_bounding_box.bounds[cut_feat].high,
                div_high: right_bounding_box.bounds[cut_feat].low,
            };
            index = self.split_nodes.len();
            self.split_nodes.push(node);
        }
        index
    }

    #[inline]
    fn middle_split(&mut self, ind: usize, count: usize, index: &mut usize,
                    cut_feat: &mut usize, cut_val: &mut FloatType, bounding_box: &BoundingBox<FloatType>) {
        let eps = FloatType::from(1e-5).unwrap();
        let mut max_span = bounding_box.bounds[0].high - bounding_box.bounds[0].low;
        for i in 1..self.dim {
            let span = bounding_box.bounds[i].high - bounding_box.bounds[i].low;
            if span > max_span {
                max_span = span;
            }
        }
        let mut max_spread = FloatType::from(-1.0).unwrap();
        *cut_feat = 0;
        let mut min_element = FloatType::zero();
        let mut max_element = FloatType::zero();
        for i in 0..self.dim {
            let span = bounding_box.bounds[i].high - bounding_box.bounds[i].low;
            if span > (FloatType::from(1.0).unwrap() - eps) * max_span {
                let mut min_element_: FloatType = FloatType::zero();
                let mut max_element_: FloatType = FloatType::zero();
                self.compute_min_max(ind, count, i, &mut min_element_, &mut max_element_);
                let spread = max_element_ - min_element_;
                if spread > max_spread {
                    max_spread = spread;
                    *cut_feat = i;
                    min_element = min_element_;
                    max_element = max_element_;
                }
            }
        }
        let split_val = FloatType::from(0.5).unwrap() * (bounding_box.bounds[*cut_feat].low + bounding_box.bounds[*cut_feat].high);

        if split_val < min_element {
            *cut_val = min_element;
        } else if split_val > max_element {
            *cut_val = max_element;
        } else {
            *cut_val = split_val;
        }

        // lim1 is index of last points that are less than cut_val
        // lim2 is index of last points that are equal to cut_val
        let mut lim1: usize = 0;
        let mut lim2: usize = 0;
        self.plane_split(ind, count, *cut_feat, cut_val, &mut lim1, &mut lim2);
        // Balancing purpose
        if lim1 > count / 2 { *index = lim1; }
        else if lim2 < count / 2 { *index = lim2; }
        else { *index = count / 2; }
    }

    #[inline]
    fn compute_min_max(&self, ind: usize, count: usize, cut_feat: usize, min_element: &mut FloatType, max_element: &mut FloatType) {
        *min_element = self.dataset.get_point(self.vind[ind], cut_feat);
        *max_element = *min_element;
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
    fn plane_split(&mut self, ind: usize, count: usize, cut_feat: usize, cut_val: &mut FloatType, lim1: &mut usize, lim2: &mut usize) {
        let mut left = 0;
        let mut right = count - 1;
        // This is a variation of the Dutch National Flag problem
        loop {
            while left <= right && self.dataset.get_point(self.vind[ind + left], cut_feat) < *cut_val {
                left += 1;
            }
            while right != 0 && left <= right && self.dataset.get_point(self.vind[ind + right], cut_feat) >= *cut_val {
                right -= 1;
            }
            if left > right || right == 0 {
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
            while right != 0 && left <= right && self.dataset.get_point(self.vind[ind + right], cut_feat) > *cut_val {
                right -= 1;
            }
            if left > right || right == 0 {
                break;
            }
            let temp = self.vind[ind + left];
            self.vind[ind + left] = self.vind[ind + right];
            self.vind[ind + right] = temp;
            left += 1;
            right -= 1;
        }
        *lim2 = left;
    }

    // Squared Euclidean
    #[inline]
    fn accum_dist(a: FloatType, b: FloatType) -> FloatType {
        let diff = a - b;
        diff * diff
    }

    // Compute how far the point is from the bounding box
    #[inline]
    fn compute_initial_distance(&self, point: &Point<FloatType>, dists: &mut Vec<FloatType>) -> FloatType {
        let mut dist_square: FloatType = FloatType::zero();
        for i in 0..self.dim {
            if point[i] < self.root_bounding_box.bounds[i].low {
                dists[i] = Self::accum_dist(point[i], self.root_bounding_box.bounds[i].low);
                dist_square = dist_square + dists[i];
            }
            if point[i] > self.root_bounding_box.bounds[i].high {
                dists[i] = Self::accum_dist(point[i], self.root_bounding_box.bounds[i].high);
                dist_square = dist_square + dists[i];
            }
        }
        dist_square
    }

    #[inline]
    fn search_level(&self, result: &mut dyn ResultSet<FloatType>, point: &Point<FloatType>,
                    node_index: usize, mut min_dists_square: FloatType, dists: &mut Vec<FloatType>, eps_error: FloatType) -> bool {
        if node_index >= self.max_num_split_nodes {
            let node = &self.leaf_nodes[node_index - self.max_num_split_nodes];
            let worst_dist = result.worst_dist();
            for i in node.left..node.right {
                let index = self.vind[i];
                let dist_square = self.dataset.get_squared_distance(point, index);
                if dist_square < worst_dist {
                    if !result.add_point(dist_square, index) {
                        return false;
                    }
                }
            }
            return true;
        }
        let node = &self.split_nodes[node_index];
        let div_feat = node.div_feat;
        let div_low = node.div_low;
        let div_high = node.div_high;
        let index = div_feat as usize;
        let val = point[index];
        let diff1 = val - div_low;
        let diff2 = val - div_high;

        let (best_child, other_child, cut_dists) =
            if diff1 + diff2 < FloatType::zero() {
                (&node.child1, &node.child2, Self::accum_dist(val, div_high))
            } else {
                (&node.child2, &node.child1, Self::accum_dist(val, div_low))
            };

        if !self.search_level(result, point, *best_child, min_dists_square, dists, eps_error) {
            return false;
        }
        let dist = dists[index];
        min_dists_square = min_dists_square + cut_dists - dist;
        dists[index] = cut_dists;
        if min_dists_square * eps_error <= result.worst_dist() {
            if !self.search_level(result, point, *other_child, min_dists_square, dists, eps_error) {
                return false;
            }
        }
        dists[index] = dist;
        true
    }
    #[inline]
    pub(crate) fn knn_search(&self, point: &Point<FloatType>, num_closest: usize, result: &mut dyn ResultSet<FloatType>)  {
        let mut dists = vec![FloatType::zero(); self.dim];
        let eps_error = FloatType::from(1.0 + 1e-5).unwrap();
        let min_dists_square = self.compute_initial_distance(point, &mut dists);
        self.search_level(result, point, self.root_index, min_dists_square, &mut dists, eps_error);
    }
    #[inline]
    fn dataset_get(&self, index: usize, dim: usize) -> FloatType {
        self.dataset.get_point(index, dim)
    }
}