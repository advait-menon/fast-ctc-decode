use std::time;

use super::SearchError;
use crate::tree::{SuffixTree, ROOT_NODE};
use ndarray::{Array1, ArrayBase, ArrayView1, Axis, Data, FoldWhile, Ix1, Ix2, Ix3, Zip};
use ndarray_stats::QuantileExt;

const EPSILON: f32 = 1e-6;

/// A node in the labelling tree to build from.
#[derive(Clone, Copy, Debug)]
struct SearchPoint {
    /// The node search should progress from.
    node: i32,
    /// The transition state for crf.
    state: usize,
    /// The cumulative probability of the labelling so far for paths without any leading blank
    /// labels.
    label_prob: f32,
    /// The cumulative probability of the labelling so far for paths with one or more leading
    /// blank labels.
    gap_prob: f32,

    length: i32, // Length of search point

    counts: [usize; 4],
}

impl SearchPoint {
    /// The total probability of the labelling so far.
    ///
    /// This sums the probabilities of the paths with and without leading blank labels.
    fn probability(&self) -> f32 {
        self.label_prob + self.gap_prob
    }
}

/// Convert probability into an ASCII encoded phred quality score between 0 and 40.
pub fn phred(prob: f32, qscale: f32, qbias: f32) -> char {
    let max = 1e-4;
    let p = if 1.0 - prob < max { max } else { 1.0 - prob };
    let q = -10.0 * p.log10() * qscale + qbias;
    std::char::from_u32(q.round() as u32 + 33).unwrap()
}

pub fn crf_beam_search<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix3>,
    init_state: &ArrayBase<D, Ix1>,
    alphabet: &[String],
    beam_size: usize,
    beam_cut_threshold: f32,
) -> Result<(String, Vec<usize>), SearchError> {
    assert!(!alphabet.is_empty());
    assert!(!network_output.is_empty());
    assert_eq!(network_output.ndim(), 3);
    assert_eq!(network_output.shape()[2], alphabet.len());

    let n_state = network_output.shape()[1];
    let n_base = network_output.shape()[2] - 1;

    let mut suffix_tree = SuffixTree::new(n_base);
    let mut beam = vec![SearchPoint {
        node: ROOT_NODE,
        label_prob: *init_state.max().unwrap(),
        gap_prob: init_state[0],
        state: init_state.argmax().unwrap(),
        length: 0,
        counts: [0; 4]
    }];
    let mut next_beam = Vec::new();

    for (idx, probs) in network_output.axis_iter(Axis(0)).enumerate() {
        next_beam.clear();

        for &SearchPoint {
            node,
            state,
            label_prob,
            gap_prob,
            length,
            counts
        } in &beam
        {
            let pr = probs.slice(s![state, ..]);

            // add N to beam
            if pr[0] > beam_cut_threshold {
                next_beam.push(SearchPoint {
                    node: node,
                    state: state,
                    label_prob: 0.0,
                    gap_prob: (label_prob + gap_prob) * pr[0],
                    length: length,
                    counts: counts
                });
            }

            for (label, &pr_b) in pr.iter().skip(1).enumerate() {
                if pr_b < beam_cut_threshold {
                    continue;
                }

                let new_node_idx = suffix_tree
                    .get_child(node, label)
                    .unwrap_or_else(|| suffix_tree.add_node(node, label, idx));

                let mut new_counts = counts;
                new_counts[label] += 1;
                next_beam.push(SearchPoint {
                    node: new_node_idx,
                    gap_prob: 0.0,
                    label_prob: (label_prob + gap_prob) * pr_b,
                    state: (state * n_base) % n_state + (label),
                    length: length + 1,
                    counts: new_counts
                });
            }
        }

        std::mem::swap(&mut beam, &mut next_beam);

        const DELETE_MARKER: i32 = i32::min_value();
        beam.sort_by_key(|x| x.node);
        let mut last_key = DELETE_MARKER;
        let mut last_key_pos = 0;
        for i in 0..beam.len() {
            let beam_item = beam[i];
            if beam_item.node == last_key {
                beam[last_key_pos].label_prob += beam_item.label_prob;
                beam[last_key_pos].gap_prob += beam_item.gap_prob;
                beam[i].node = DELETE_MARKER;
            } else {
                last_key_pos = i;
                last_key = beam_item.node;
            }
        }

        beam.retain(|x| x.node != DELETE_MARKER);
        let mut has_nans = false;
        beam.sort_unstable_by(|a, b| {
            (b.probability())
                .partial_cmp(&(a.probability()))
                .unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal // don't really care
                })
        });
        if has_nans {
            return Err(SearchError::IncomparableValues);
        }
        beam.truncate(beam_size);
        if beam.is_empty() {
            // we've run out of beam (probably the threshold is too high)
            return Err(SearchError::RanOutOfBeam);
        }
        let top = beam[0].probability();
        for x in &mut beam {
            x.label_prob /= top;
            x.gap_prob /= top;
        }
    }

    let mut path = Vec::new();
    let mut sequence = String::new();

    if beam[0].node != ROOT_NODE {
        for (label, &time) in suffix_tree.iter_from(beam[0].node) {
            path.push(time);
            sequence.push_str(&alphabet[label + 1]);
        }
    }

    path.reverse();
    Ok((sequence.chars().rev().collect::<String>(), path))
}

pub fn shannon_entropy(probs: ArrayView1<f32>) -> f32 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

fn kl_divergence(counts: &[usize]) -> f32 {
    // assumes TOTAL count is > 0
    let total: usize = counts.iter().sum();
    let uniform = 0.25;
    let mut kl = 0.0;
    let mut have_updated = false;

    for &count in counts {
        if count > 0 {
            have_updated = true;
            let p = count as f32 / total as f32;
            kl += p * (p / uniform).ln();
        }
    }
    if !have_updated {
        return f32::MAX; // return max if we havent updated
    } else {
        return kl
    }
}

pub fn beam_search<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    beam_size: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
    no_repeats: bool,
    entropy_threshold: f32,
    length: i32,
) -> Result<(String, Vec<usize>), SearchError> {
    // alphabet size minus the blank label
    let alphabet_size = alphabet.len() - 1;
    let time_steps = network_output.len_of(Axis(0));
    let expected_bases_per_signal = length as f32/time_steps as f32;

    let mut suffix_tree = SuffixTree::new(alphabet_size);
    let mut beam = vec![SearchPoint {
        node: ROOT_NODE, //starting node to progress from
        state: 0, // crf transition state
        gap_prob: 1.0, // cum prob of labelling so far for paths with one or more leading blank labels
        label_prob: 0.0, // cum prob of labelling so far for paths without leading blank label
        length: 0,
        counts: [0;4]
    }]; //vector of search points for current beam (initialised with starting search point)
    let mut next_beam = Vec::new(); // vector of search points for next beam

    // pr is the probabilities at time given by idx
    for (idx, pr) in network_output.outer_iter().enumerate() {
        next_beam.clear(); //empty the list of next search points
        let entropy = shannon_entropy(pr);
        // for each searchpoint in the current beam
        for &SearchPoint {
            node,
            label_prob,
            gap_prob,
            state,
            length,
            counts
        } in &beam
        {   
            let mut underrepresented = Vec::new();
            // if entropy is > threshold then let us count the number of total A/C/G/T

            // if entropy_threshold >= 0.0 && entropy > entropy_threshold && length > 0 {
            //     // find proportion of each base
            //     if total > 0 {
            //         if (a_count as f32 / total as f32) < 0.25 {
            //             underrepresented.push(1); // A
            //         }
            //         if (c_count as f32 / total as f32) < 0.25 {
            //             underrepresented.push(2); // C
            //         }
            //         if (g_count as f32 / total as f32) < 0.25 {
            //             underrepresented.push(3); // G
            //         }
            //         if (t_count as f32 / total as f32) < 0.25 {
            //             underrepresented.push(4); // T
            //         }
            //     }
            // }

            // now find KL and if its too big only look at underrepresented bases :D
            if length > 0 {
                let kl = kl_divergence(&counts);
                if kl > 0.005 && entropy > 1.15 {
                    for (i, &count) in counts.iter().enumerate() {
                        let freq = count as f64 / length as f64;
                        if freq < 0.25 {
                            underrepresented.push(i);
                        }
                    }
                }
            }

            // tip_label is the final label of the branch i.e. label of the last node which is the search point node
            let tip_label = suffix_tree.label(node);            
            // note that this is because the blank label is always the first
            // add N to beam if probability is above
            if pr[0] > beam_cut_threshold {
                next_beam.push(SearchPoint {
                    node: node, // from same node 
                    state: state, // same state transistion
                    label_prob: 0.0, // label prob is 0 because we do have a leading blank
                    gap_prob: (label_prob + gap_prob) * pr[0], // (cum prob of paths with leading blank + without) * probability of blank at this step
                    length: length,
                    counts: counts
                });
            }
            // note we dont add anything to the suffix tree for the blank case
            
            // for all other labels,probs except the blank
            for (label, &pr_b) in pr.iter().skip(1).enumerate() {
                // if probability is less than threshold then ignore
                if pr_b < beam_cut_threshold {
                    continue;
                }

                // if entropy > 1.0 && !underrepresented.contains(&label) && !underrepresented.is_empty() {
                //     continue;
                // }

                if !underrepresented.contains(&label) && !underrepresented.is_empty() {
                    continue;
                }
                
                // if collapse repeats (true for CTC and the current label we consider is same as the tip label)
                // then add the next search point (gap prob is 0 because no leading blank, multiply label prob by next label prob)
                if collapse_repeats && Some(label) == tip_label {
                    next_beam.push(SearchPoint {
                        node: node,
                        label_prob: label_prob * pr_b,
                        gap_prob: 0.0,
                        state: state,
                        length: length,
                        counts: counts
                    });

                    // new node is child of current node or if it doesnt exist 
                    // otherwise if no child and its possible for path to have a blank
                    // add a node as child of current node and make that new node idx
                    // we do this so that we can have repeats e.g. ANA -> AA. since we merge the first AN -> A
                    // for repeats we need it to have a new node idx
                    let new_node_idx = suffix_tree.get_child(node, label).or_else(|| {
                        if gap_prob > 0.0 {
                            Some(suffix_tree.add_node(node, label, idx))
                        } else {
                            None
                        }
                    });
                    
                    // if new_node_idx is not None then set idx to new node idx
                    // next search point is the new node, and we update label_prob,

                    // if no_repeats == True then we don't want this search point
                    let mut new_counts = counts;
                    new_counts[label] += 1;
                    if let Some(idx) = new_node_idx{
                        next_beam.push(SearchPoint {
                            node: idx,
                            state: state,
                            label_prob: if no_repeats {0.0} else {gap_prob * pr_b},
                            gap_prob: 0.0,
                            length: length+1,
                            counts: new_counts
                        });
                    }
                } else {
                    // if not collapsing repeats or if parent is root
                    // we get the child of the node or if it doesnt exist we add that node
                    let new_node_idx = suffix_tree
                        .get_child(node, label)
                        .unwrap_or_else(|| suffix_tree.add_node(node, label, idx));
                    let mut new_counts = counts;
                    new_counts[label] += 1;
                    // next beam is here
                    next_beam.push(SearchPoint {
                        node: new_node_idx,
                        state: state,
                        label_prob: (label_prob + gap_prob) * pr_b,
                        gap_prob: 0.0,
                        length: length+1,
                        counts: new_counts
                    });
                }
            }
        }
        // beam, next_beam = next_beam, beam. Note that we clear next_beam each loop so it doesnt matter if we swap the,
        std::mem::swap(&mut beam, &mut next_beam);

        // del marker is smallest i32, sort the search points in beam (which is the search points for next iteration by the index of the node)
        const DELETE_MARKER: i32 = i32::min_value();
        beam.sort_by_key(|x| x.node);
        // set a last key and last_key_pos to keep track of last seen key and its position
        let mut last_key = DELETE_MARKER;
        let mut last_key_pos = 0;

        // iterate through all search points in beam
        for i in 0..beam.len() {
            let beam_item = beam[i];

            //i.e. if two consecutive items have the same node index, merge their probabilities
            // otherwise set the last key to that search point
            // this only happens for the N branch and the branch with the same label from a node
            if beam_item.node == last_key {
                beam[last_key_pos].label_prob += beam_item.label_prob;
                beam[last_key_pos].gap_prob += beam_item.gap_prob;
                beam[i].node = DELETE_MARKER;
            } else {
                last_key_pos = i;
                last_key = beam_item.node;
            }
        }

        // filter out all "duplicate" nodes (see above for loop) that we merged
        beam.retain(|x| x.node != DELETE_MARKER);

        // sorts search points in descending order of probabilities and if any are NaN set has_nans to true
        // partial cmp between b and a is descending, unstable sort means in place, if we have nans we set them to equal
        // as we dont care about them
        let mut has_nans = false;
    
        if length >= 1 {
            beam.sort_unstable_by(|a, b| {
                let prob_a = a.probability();
                let prob_b = b.probability();

                // handle cases where either search point is impossible
                let mut a_zero: bool = false;
                let mut b_zero: bool = false;

                if prob_a < EPSILON{
                    a_zero = true;
                }
                if prob_b < EPSILON {
                    b_zero = true;
                }
                
                if a_zero && b_zero {
                    return std::cmp::Ordering::Equal
                } else if b_zero {
                    return std::cmp::Ordering::Less
                } else if a_zero {
                    return std::cmp::Ordering::Greater
                }


                // let dist_a = (a.length as f32 - length as f32).abs();
                // let dist_b = (b.length as f32 - length as f32).abs();
            

                // let bases_per_signal_a = a.length as f32/(idx+1) as f32;
                // let bases_per_signal_b = b.length as f32/(idx+1) as f32;

                // let length_inverse = 1.0/(length as f32);

                // let a_kl = kl_divergence(&a.counts);
                // let b_kl = kl_divergence(&b.counts);

                let score_a = prob_a;
                let score_b = prob_b;

                // Combine probability with length proximity
                // score_a = prob_a * (1.0 - (1.0/(length as f32)) * dist_a as f32) * idx as f32 * (1.0/time_steps as f32);
                // score_b = prob_b * (1.0 - (1.0/(length as f32)) * dist_b as f32) * idx as f32 * (1.0/time_steps as f32);


                // if (idx+1) as f32 > (1.0/expected_bases_per_signal) {
                //     if (bases_per_signal_a - expected_bases_per_signal).abs() > 0.1 {
                //         score_a = 0.0;
                //     }

                //     if (bases_per_signal_b - expected_bases_per_signal).abs() > 0.1 {
                //         score_b = 0.0;
                //     }
                // }
            
                score_b.partial_cmp(&score_a).unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal
                })
            });
        } else {
            beam.sort_unstable_by(|a, b| {
                (b.probability())
                    .partial_cmp(&(a.probability()))
                    .unwrap_or_else(|| {
                        has_nans = true;
                        std::cmp::Ordering::Equal // don't really care
                    })
            });
        }

        // if we have nans raise error
        if has_nans {
            return Err(SearchError::IncomparableValues);
        }

        // keep the top beam_size probabilities (beam width)
        beam.truncate(beam_size);

        // if beam empty then threshold too high
        if beam.is_empty() {
            // we've run out of beam (probably the threshold is too high)
            return Err(SearchError::RanOutOfBeam);
        }
        // highest probability is top
        let top = beam[0].probability();
        // normalise probabilities by highest one for numerical stability
        for x in &mut beam {
            x.label_prob /= top;
            x.gap_prob /= top;
        }
    }

    let mut path = Vec::new();
    let mut sequence = String::new();

    // if the first node in beam is not root then its valid path with highest prob
    // iterate from that backwards and add time to path and label to sequence
    if beam[0].node != ROOT_NODE {
        for (label, &time) in suffix_tree.iter_from(beam[0].node) {
            path.push(time);
            sequence.push_str(&alphabet[label + 1]);
        }
    }

    // reverse the path and sequence, return them
    path.reverse();
    Ok((sequence.chars().rev().collect::<String>(), path))
}

fn find_max(
    acc: Option<(usize, f32)>,
    elem_idx: usize,
    elem_val: &f32,
) -> FoldWhile<Option<(usize, f32)>> {
    match acc {
        Some((_, val)) => {
            if *elem_val > val {
                FoldWhile::Continue(Some((elem_idx, *elem_val)))
            } else {
                FoldWhile::Continue(acc)
            }
        }
        None => FoldWhile::Continue(Some((elem_idx, *elem_val))),
    }
}

pub fn viterbi_search<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    qstring: bool,
    qscale: f32,
    qbias: f32,
    collapse_repeats: bool,
) -> Result<(String, Vec<usize>), SearchError> {
    assert!(!alphabet.is_empty());
    assert!(!network_output.is_empty());
    assert_eq!(network_output.ndim(), 2);
    assert_eq!(alphabet.len(), network_output.shape()[1]);

    let mut path = Vec::new();
    let mut quality = String::new();
    let mut sequence = String::new();

    let mut last_label = None;
    let mut label_prob_count = 0;
    let mut label_prob_total = 0.0;

    for (idx, pr) in network_output.outer_iter().enumerate() {
        let (label, prob) = Zip::indexed(pr)
            .fold_while(None, find_max)
            .into_inner()
            .unwrap(); // only an empty network_output could give us None

        if label != 0 && (!collapse_repeats || last_label != Some(label)) {
            if label_prob_count > 0 {
                quality.push(phred(
                    label_prob_total / (label_prob_count as f32),
                    qscale,
                    qbias,
                ));
                label_prob_total = 0.0;
                label_prob_count = 0;
            }

            sequence.push_str(&alphabet[label]);
            path.push(idx);
        }

        if label != 0 {
            label_prob_total += prob;
            label_prob_count += 1;
        }

        last_label = Some(label);
    }

    if label_prob_count > 0 {
        quality.push(phred(
            label_prob_total / (label_prob_count as f32),
            qscale,
            qbias,
        ));
    }

    if qstring {
        sequence.push_str(&quality);
    }

    Ok((sequence, path))
}

pub fn crf_greedy_search<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix3>,
    init_state: &ArrayBase<D, Ix1>,
    alphabet: &[String],
    qstring: bool,
    qscale: f32,
    qbias: f32,
) -> Result<(String, Vec<usize>), SearchError> {
    assert!(!alphabet.is_empty());
    assert!(!network_output.is_empty());
    assert_eq!(network_output.ndim(), 3);
    assert_eq!(network_output.shape()[2], alphabet.len());

    let n_state = network_output.shape()[1] as i32;
    let n_base = network_output.shape()[2] as i32 - 1;

    let mut path = Vec::new();
    let mut quality = String::new();
    let mut sequence = String::new();
    let mut state = init_state.argmax().unwrap() as i32;

    for (idx, pr) in network_output.axis_iter(Axis(0)).enumerate() {
        let label = pr.slice(s![state, ..]).argmax().unwrap();

        if label > 0 {
            path.push(idx);
            sequence.push_str(&alphabet[label]);
            let prob = *pr.slice(s![state, ..]).max().unwrap();
            quality.push(phred(prob, qscale, qbias));
            state = (state * n_base) % n_state + (label as i32 - 1);
        }
    }

    if qstring {
        sequence.push_str(&quality);
    }

    Ok((sequence, path))
}

#[cfg(test)]
mod tests {
    use super::*;
    //use test::Bencher;

    #[test]
    fn crf_test_greedy() {
        let alphabet = vec![
            String::from("N"),
            String::from("A"),
            String::from("C"),
            String::from("G"),
            String::from("T"),
        ];

        let network_output = array![
            [
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [1f32, 0.000, 0.000, 0.000, 0.000], // N 2
                [0f32, 0.000, 0.000, 0.000, 0.000],
            ],
            [
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.900, 0.000, 0.000], // C 2
                [0f32, 0.000, 0.000, 0.000, 0.000],
            ],
            [
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.700], // T 1
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
            ],
            [
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [1f32, 0.000, 0.000, 0.000, 0.000], // N 3
            ],
            [
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.990, 0.000, 0.000, 0.000], // A 3
            ],
            [
                [0f32, 0.900, 0.000, 0.000, 0.000], // A 0
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
            ],
            [
                [0f32, 0.000, 0.000, 0.999, 0.000], // G 0
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
            ],
        ];
        let init = array![0f32, 0., 1., 0., 0.];
        let (sequence, path) =
            crf_greedy_search(&network_output, &init, &alphabet, false, 1.0, 0.0).unwrap();

        assert_eq!(sequence, "CTAAG");
        assert_eq!(path, vec![1, 2, 4, 5, 6]);

        let (sequence, path) =
            crf_greedy_search(&network_output, &init, &alphabet, true, 1.0, 0.0).unwrap();

        assert_eq!(sequence, "CTAAG+&5+?");
        assert_eq!(path, vec![1, 2, 4, 5, 6]);

        let beam_size = 5;
        let beam_cut_threshold = 0.01;
        let (sequence, path) = crf_beam_search(
            &network_output,
            &init,
            &alphabet,
            beam_size,
            beam_cut_threshold,
        )
        .unwrap();

        assert_eq!(sequence, "CTAAG");
        assert_eq!(path, vec![1, 2, 4, 5, 6]);
    }

    #[test]
    fn test_phred_scores() {
        let qbias = 0.0;
        let qscale = 1.0;
        assert_eq!('!', phred(0.0, qscale, qbias));
        assert_eq!('$', phred(0.5, qscale, qbias));
        assert_eq!('+', phred(1.0 - 1e-1, qscale, qbias));
        assert_eq!('5', phred(1.0 - 1e-2, qscale, qbias));
        assert_eq!('?', phred(1.0 - 1e-3, qscale, qbias));
        assert_eq!('I', phred(1.0 - 1e-4, qscale, qbias));
        assert_eq!('I', phred(1.0 - 1e-5, qscale, qbias));
        assert_eq!('I', phred(1.0 - 1e-6, qscale, qbias));
        assert_eq!('I', phred(1.0, qscale, qbias));
    }

    #[test]
    fn test_viterbi() {
        let qbias = 0.0;
        let qscale = 1.0;
        let alphabet = vec![String::from("N"), String::from("A"), String::from("G")];
        let network_output = array![
            [0.0f32, 0.4, 0.6], // G
            [0.0f32, 0.3, 0.7], // G
            [0.3f32, 0.3, 0.4], // G
            [0.4f32, 0.3, 0.3], // N
            [0.4f32, 0.3, 0.3], // N
            [0.3f32, 0.3, 0.4], // G
            [0.1f32, 0.4, 0.5], // G
            [0.1f32, 0.5, 0.4], // A
            [0.8f32, 0.1, 0.1], // N
            [0.1f32, 0.1, 0.8], // G
        ];

        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, false, qscale, qbias, true).unwrap();
        assert_eq!(seq, "GGAG");
        assert_eq!(starts, vec![0, 5, 7, 9]);

        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, true, qscale, qbias, true).unwrap();
        assert_eq!(seq, "GGAG%$$(");
        assert_eq!(starts, vec![0, 5, 7, 9]);
    }

    #[test]
    fn test_viterbi_blank_bounds() {
        let qbias = 0.0;
        let qscale = 1.0;
        let alphabet = vec![String::from("N"), String::from("A"), String::from("G")];
        let network_output = array![
            [0.6f32, 0.2, 0.2], // N
            [0.6f32, 0.2, 0.2], // N
            [0.0f32, 0.4, 0.6], // G
            [0.0f32, 0.3, 0.7], // G
            [0.3f32, 0.3, 0.4], // G
            [0.4f32, 0.3, 0.3], // N
            [0.4f32, 0.3, 0.3], // N
            [0.3f32, 0.3, 0.4], // G
            [0.1f32, 0.4, 0.5], // G
            [0.1f32, 0.5, 0.4], // A
            [0.8f32, 0.1, 0.1], // N
            [0.1f32, 0.1, 0.8], // G
            [0.4f32, 0.3, 0.3], // N
        ];
        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, false, qscale, qbias, true).unwrap();
        assert_eq!(seq, "GGAG");
        assert_eq!(starts, vec![2, 7, 9, 11]);

        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, true, qscale, qbias, true).unwrap();
        assert_eq!(seq, "GGAG%$$(");
        assert_eq!(starts, vec![2, 7, 9, 11]);

        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, false, qscale, qbias, false).unwrap();
        assert_eq!(seq, "GGGGGAG");
        assert_eq!(starts, vec![2, 3, 4, 7, 8, 9, 11]);

        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, true, qscale, qbias, false).unwrap();
        assert_eq!(seq, "GGGGGAG%&##$$(");
        assert_eq!(starts, vec![2, 3, 4, 7, 8, 9, 11]);

        let (seq, _starts) = beam_search(&network_output, &alphabet, 5, 0.0, true, false, -1.0,-1).unwrap();
        assert_eq!(seq, "GAGAG");

        let (seq, _starts) = beam_search(&network_output, &alphabet, 5, 0.0, false, false, -1.0,-1).unwrap();
        assert_eq!(seq, "GGGAGAG");
    }

    /*
    // This one is all blanks, and so returns no sequence (which means we're not benchmarking the
    // construction of the results).
    #[bench]
    fn benchmark_trivial_viterbi(b: &mut Bencher) {
        use ndarray::Array2;
        let qbias = 0.0;
        let qscale = 1.0;
        let alphabet = vec![String::from("N"), String::from("A"), String::from("G")];
        let network_output = Array2::from_shape_fn((1000, 3), |p| match p {
            (_, 0) => 1.0f32,
            (_, _) => 0.0f32,
        });
        b.iter(|| viterbi_search(&network_output, &alphabet, false, qscale, qbias, true));
    }

    // This one changes label at every data point, so result contruction has the maximum possible
    // impact on run time.
    #[bench]
    fn benchmark_unstable_viterbi(b: &mut Bencher) {
        use ndarray::Array2;
        let qbias = 0.0;
        let qscale = 1.0;
        let alphabet = vec![String::from("N"), String::from("A"), String::from("G")];
        let network_output = Array2::from_shape_fn((1000, 3), |p| match p {
            (n, 1) if n % 2 == 0 => 0.0f32,
            (n, 1) if n % 2 != 0 => 1.0f32,
            (n, 2) if n % 2 == 0 => 1.0f32,
            (n, 2) if n % 2 != 0 => 0.0f32,
            _ => 0.0f32,
        });
        b.iter(|| viterbi_search(&network_output, &alphabet, false, qscale, qbias, true));
    }
     */
}
