Blazingly Fast 3d KD Tree implementation on based on nanoflann https://github.com/jlblancoc/nanoflann on Rust. 

Optimised to do approximate nearest neighbour queries with millions of points.

Benchmarking with [kd-tree crate](https://github.com/mrhooray/kdtree-rs) with the following settings: 
- Using one frame from longdress point cloud file.
- About 760000 points
- Find approximate 10 nearest neighbours for each points

This kd-tree implementation is about 1.8-2x faster.

```
kd_tree_construction/nanofrann_query_resample
                        time:   [794.17 ms 868.52 ms 958.46 ms]
kd_tree_construction/kd_tree_query_resample
                        time:   [1.4154 s 1.5954 s 1.8034 s]
```

To-Do:
- Refactor DataSource to be a trait so nanofrann can be used with custom data structure.
- Add example code and how to use.
