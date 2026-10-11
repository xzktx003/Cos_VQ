# Third-party notices

## NoWag

`third_party/NoWag/` vendors the subset of NoWag used by the baseline quantization pipeline. The copied source was based on revision `07638f7ff3b7c929b036707ca49355b5737c9883` and includes local changes for chunked k-means assignment and dtype-preserving centroid/update operations. The vendored files carry the GNU GPL version 3 license; see `third_party/NoWag/LICENSE`.

The root `LICENSE` does not replace or relicense this third-party component. Review the obligations that apply to the intended distribution of the combined package before publishing it. Model weights and datasets are not included and retain their respective upstream terms.
