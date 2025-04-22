use crate::vec2d::Vec2D;

/// An element in a possible labelling.
#[derive(Clone, Copy, Debug)]
struct LabelNode<T> {
    /// The index into the alphabet of this label.
    ///
    /// Note that blanks are not represented by a LabelNode - this is an actual label.
    label: usize,
    /// The index of the parent LabelNode.
    parent: i32,
    /// Extra data attached to the node
    data: T,
}

/// A tree of labelling suffixes (partial labellings pinned to the end of the network output).
pub struct SuffixTree<T> {
    // Invariants:
    //
    // nodes[i].parent < nodes.len() for all i
    // nodes[i].parent < 0 => nodes[i].parent == ROOT_NODE for all i
    // children[i][j]] < nodes.len() for all i, j
    //
    // nodes.len() == children.len()
    //
    // For all n, l where suffix_children[n][l] != ROOT_NODE:
    //     nodes[children[n][l]].label == l (child edge label matches child label)
    //     nodes[children[n][l]].parent == n  (child's parent pointer is correct)
    //
    // Also:
    //     nodes[root_children[ROOT_NODE][l]].label == l (child edge label matches child label)
    //     nodes[root_children[ROOT_NODE][l]].parent == ROOT_NODE  (child's parent pointer is correct)
    //
    // For all n > 0:
    //     nodes[n].parent != ROOT_NODE => children[nodes[n].parent][nodes[n].label] == n
    //     nodes[n].parent == ROOT_NODE => root_children[nodes[n].label] == n
    //     (the parent node has a child edge back to this node labelled correctly)
    nodes: Vec<LabelNode<T>>,
    children: Vec2D<i32>,
    // We don't actually store the root node in `nodes`, because it has no associated label, data
    // or parent. In order to keep `nodes` and `children` in line (so they could be zipped), we
    // store the root's children here.
    root_children: Vec<i32>,
}

pub struct SuffixTreeIter<'a, T> { 
    nodes: &'a Vec<LabelNode<T>>, //reference to nodes of suffixtree
    next: i32, // node to start traversing from
}

impl<'a, T> Iterator for SuffixTreeIter<'a, T> {
    type Item = (usize, &'a T); // returns usize and the type of the data of a LabelNode

    fn next(&mut self) -> Option<Self::Item> {
        // if not root node, get the nodes parent as next node and return the nodes label and data
        if self.next >= 0 {
            // NB: we could use an unsafe deref here as we maintain the invariant that
            // next <= nodes.len()
            let node = &self.nodes[self.next as usize];
            self.next = node.parent;
            Some((node.label, &node.data))
        } else {
            None
        }
    }
}

pub struct SuffixTreeIterNoData<'a, T> {
    nodes: &'a Vec<LabelNode<T>>, //reference to the nodes vector of suffix tree
    next: i32, //index of starting node to traverse from
}

impl<'a, T> Iterator for SuffixTreeIterNoData<'a, T> {
    type Item = usize; // we return a label

    fn next(&mut self) -> Option<Self::Item> {
        // if the node is not root we set its parent as the next node and return its label
        if self.next >= 0 {
            // NB: we could use an unsafe deref here as we maintain the invariant that
            // next <= nodes.len()
            let node = &self.nodes[self.next as usize];
            self.next = node.parent;
            Some(node.label)
        } else {
            None
        }
    }
}

pub const ROOT_NODE: i32 = -1;

#[derive(Clone, Copy, Debug)]
pub struct NodeInfo {
    pub parent: i32,
    pub label: usize,
}

impl<T> SuffixTree<T> {
    pub fn new(alphabet_size: usize) -> Self {
        Self {
            nodes: Vec::new(), // empty vector
            children: Vec2D::new(alphabet_size), // 0 rows, alphabet_size cols
            root_children: vec![-1; alphabet_size], // vector of -1
        }
    }

    pub fn label(&self, node: i32) -> Option<usize> {
        if node >= 0 { //if node index is valid (and not root) return its label
            Some(self.nodes[node as usize].label)
        } else {
            None
        }
    }

    pub fn info(&self, node: i32) -> Option<NodeInfo> {
        // gets the label and parent index of given valid node
        if node >= 0 {
            let node = &self.nodes[node as usize];
            Some(NodeInfo {
                parent: node.parent,
                label: node.label,
            })
        } else {
            None
        }
    }

    pub fn add_node(&mut self, parent: i32, label: usize, data: T) -> i32 {
        // check label is in bounds and we havent overflowed
        assert!(label < self.root_children.len());
        assert!(self.nodes.len() < (i32::max_value() as usize));
        
        // new node index is number of nodes we have
        let new_node_idx = self.nodes.len() as i32;
        // if parent is root make sure that we havent created a child for that label yet
        // and then set the index
        if parent == ROOT_NODE {
            assert_eq!(self.root_children[label], -1);
            self.root_children[label] = new_node_idx;
        } else {
        // make sure the parent is not the root node and the parent doest have a child for this label
            assert!(parent >= 0);
            assert_eq!(self.children[(parent as usize, label)], -1);
            self.children[(parent as usize, label)] = new_node_idx;
        }
        // append a LabelNode to the list of nodes
        self.nodes.push(LabelNode {
            label,
            parent,
            data,
        });
        // add a row in children (with alphabet_size - 1 columns) filled with -1 
        //i.e. this new node has no children
        self.children.add_row_with_value(-1); 

        // return the new node index
        new_node_idx
    }

    pub fn get_child(&self, node: i32, label: usize) -> Option<i32> {
        // if node is root then get the child from root_children
        if node == ROOT_NODE {
            let idx = self.root_children[label];
            if idx >= 0 {
                return Some(idx);
            }
        } else {
            // else make sure the node is not root and take the child node 
            // from children
            assert!(node >= 0);
            let idx = self.children[(node as usize, label)];
            if idx >= 0 {
                return Some(idx);
            }
        }
        None
    }

    pub fn get_data_ref(&self, node: i32) -> Option<&T> {
        // if node is within bounds and not root, get its data
        if node >= 0 && (node as usize) < self.nodes.len() {
            Some(&self.nodes[node as usize].data)
        } else {
            None
        }
    }

    pub fn get_data_ref_mut(&mut self, node: i32) -> Option<&mut T> {
        // uses a mutable reference and returns a mutable reference to the data
        if node >= 0 && (node as usize) < self.nodes.len() {
            Some(&mut self.nodes[node as usize].data)
        } else {
            None
        }
    }

    pub fn iter_from_no_data(&self, node: i32) -> SuffixTreeIterNoData<T> {
        // returns a no data iterator starting from the fiven node
        assert!((node as usize) < self.nodes.len());
        SuffixTreeIterNoData {
            nodes: &self.nodes,
            next: node,
        }
    }

    pub fn iter_from(&self, node: i32) -> SuffixTreeIter<T> {
        // returns a interator from the given node that does go over data
        assert!((node as usize) < self.nodes.len());
        SuffixTreeIter {
            nodes: &self.nodes,
            next: node,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_assembly() {
        let mut tree = SuffixTree::new(2);
        assert_eq!(tree.get_child(-1, 0), None);
        assert_eq!(tree.get_child(-1, 1), None);
        assert_eq!(tree.get_data_ref(-1), None);
        assert_eq!(tree.label(-1), None);

        assert_eq!(tree.add_node(-1, 0, 100), 0);
        assert_eq!(tree.get_child(-1, 0), Some(0));
        assert_eq!(tree.get_data_ref(0), Some(&100));
        assert_eq!(tree.label(0), Some(0));

        assert_eq!(tree.add_node(-1, 1, 101), 1);
        assert_eq!(tree.get_child(-1, 1), Some(1));
        assert_eq!(tree.get_data_ref(1), Some(&101));
        assert_eq!(tree.label(1), Some(1));

        assert_eq!(tree.add_node(0, 0, 102), 2);
        assert_eq!(tree.get_child(0, 0), Some(2));
        assert_eq!(tree.get_data_ref(2), Some(&102));
        assert_eq!(tree.label(2), Some(0));

        assert_eq!(tree.add_node(0, 1, 103), 3);
        assert_eq!(tree.get_child(0, 1), Some(3));
        assert_eq!(tree.get_data_ref(3), Some(&103));
        assert_eq!(tree.label(3), Some(1));

        assert_eq!(tree.add_node(3, 1, 104), 4);
        assert_eq!(tree.get_child(3, 1), Some(4));
        assert_eq!(tree.get_data_ref(4), Some(&104));
        assert_eq!(tree.label(4), Some(1));

        assert_eq!(tree.add_node(1, 0, 105), 5);
        assert_eq!(tree.get_child(1, 0), Some(5));
        assert_eq!(tree.get_data_ref(5), Some(&105));
        assert_eq!(tree.label(5), Some(0));

        // everything still unchanged
        assert_eq!(tree.label(-1), None);
        assert_eq!(tree.get_child(-1, 0), Some(0));
        assert_eq!(tree.get_data_ref(0), Some(&100));
        assert_eq!(tree.label(0), Some(0));
        assert_eq!(tree.get_child(-1, 1), Some(1));
        assert_eq!(tree.get_data_ref(1), Some(&101));
        assert_eq!(tree.label(1), Some(1));
        assert_eq!(tree.get_child(0, 0), Some(2));
        assert_eq!(tree.get_data_ref(2), Some(&102));
        assert_eq!(tree.label(2), Some(0));
        assert_eq!(tree.get_child(0, 1), Some(3));
        assert_eq!(tree.get_data_ref(3), Some(&103));
        assert_eq!(tree.label(3), Some(1));
        assert_eq!(tree.get_child(1, 0), Some(5));
        assert_eq!(tree.get_data_ref(5), Some(&105));
        assert_eq!(tree.label(5), Some(0));
        assert_eq!(tree.get_child(1, 1), None);
        assert_eq!(tree.get_child(2, 0), None);
        assert_eq!(tree.get_child(2, 1), None);
        assert_eq!(tree.get_child(3, 0), None);
        assert_eq!(tree.get_child(3, 1), Some(4));
        assert_eq!(tree.get_data_ref(4), Some(&104));
        assert_eq!(tree.label(4), Some(1));

        let ancestor_labels: Vec<usize> = tree.iter_from_no_data(4).collect();
        assert_eq!(ancestor_labels, vec![1, 1, 0]);

        let ancestor_label_and_data: Vec<(usize, i32)> =
            tree.iter_from(4).map(|(x, &y)| (x, y)).collect();
        assert_eq!(ancestor_label_and_data, vec![(1, 104), (1, 103), (0, 100)]);
    }
}
